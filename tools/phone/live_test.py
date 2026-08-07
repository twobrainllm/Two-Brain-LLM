#!/usr/bin/env python3
"""End-to-end live test of the phone fast brain, one step at a time.

Run with the S25 attached over USB (or paired via wireless debugging):

    python tools/phone/live_test.py

Every step reports PASS / FAIL / SKIP and keeps going where it safely can, so
a half-finished device -- server up, model binary not pushed yet -- still
produces a useful report instead of one opaque traceback.

Nothing here writes to the device. It reads properties, forwards a port, and
sends prompts.

    --mock   skip the adb steps and test a local server instead
             (src/phone_brain/mock_phone_brain_server.py). That is how you verify
             this harness itself without hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

ADB_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Android/Sdk/platform-tools/adb.exe",
    Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
    Path("C:/platform-tools/adb.exe"),
]

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
_results: list[tuple[str, str, str]] = []


def report(status: str, step: str, detail: str = "") -> None:
    _results.append((status, step, detail))
    mark = {PASS: "[ OK ]", FAIL: "[FAIL]", SKIP: "[SKIP]", INFO: "[ .. ]"}[status]
    print(f"{mark} {step}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def find_adb() -> str | None:
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    for candidate in ADB_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def adb(adb_path: str, *args: str, timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run([adb_path, *args], capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {timeout}s"


def http_post(url: str, payload: dict, timeout: int = 120) -> tuple[dict | None, float, str]:
    """Returns (body, latency_ms, error). Never raises."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body, (time.perf_counter() - start) * 1000, ""
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:  # noqa: BLE001
            pass
        return None, (time.perf_counter() - start) * 1000, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return None, (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------


def step_device(adb_path: str) -> str | None:
    code, out = adb(adb_path, "devices", "-l")
    if code != 0:
        report(FAIL, "adb devices", out)
        return None

    lines = [ln for ln in out.splitlines()[1:] if ln.strip()]
    devices = [ln for ln in lines if "\tdevice" in ln or " device " in ln]
    unauthorized = [ln for ln in lines if "unauthorized" in ln]
    offline = [ln for ln in lines if "offline" in ln]

    if unauthorized:
        report(FAIL, "device authorized", "Unlock the phone and tap 'Allow USB debugging'.")
        return None
    if offline:
        report(FAIL, "device online", f"{offline}\nTry: adb kill-server && adb start-server")
        return None
    if not devices:
        report(
            FAIL,
            "device attached",
            "No device. In order of likelihood:\n"
            "  1. USB preferences -> 'USB controlled by' must be 'Connected device'.\n"
            "     Set to 'This device' the phone acts as USB host and never\n"
            "     enumerates as a peripheral -- it charges, host sees nothing.\n"
            "  2. USB mode -> 'Transferring files (MTP)'. NOT USB tethering:\n"
            "     that shares mobile data, needs a SIM, and is unrelated to adb.\n"
            "  3. Developer options -> USB debugging ON, phone unlocked.\n"
            "  4. Wireless works too:\n"
            "       adb pair <ip>:<pair-port> && adb connect <ip>:<port>",
        )
        return None
    if len(devices) > 1:
        report(INFO, "multiple devices attached", out)

    serial = devices[0].split()[0]
    report(PASS, "device attached", devices[0])

    for prop, label in [
        ("ro.product.model", "model"),
        ("ro.soc.model", "soc"),
        ("ro.build.version.release", "android"),
    ]:
        code, value = adb(adb_path, "-s", serial, "shell", "getprop", prop)
        if code == 0 and value:
            report(INFO, f"device {label}", value)
    return serial


def step_forward(adb_path: str, serial: str, port: int) -> bool:
    """adb forward, not adb reverse. `forward` opens a port on the HOST that
    tunnels to the device -- the server is on the phone and the caller is here.
    (L_INTERFACE_CONTRACT.md says `reverse`, which is backwards; see
    docs/phone-wiring.md.)"""
    code, out = adb(adb_path, "-s", serial, "forward", f"tcp:{port}", f"tcp:{port}")
    if code != 0:
        report(FAIL, f"adb forward tcp:{port}", out)
        return False
    report(PASS, f"adb forward tcp:{port} -> device tcp:{port}", "")
    return True


def step_server_process(adb_path: str, serial: str, port: int) -> None:
    """Distinguishes 'server not started' from 'server up, model not loaded'.
    Different fixes, and the second is the expected state until the binary
    lands."""
    code, out = adb(adb_path, "-s", serial, "shell", f"cat /proc/net/tcp | grep -i ':{port:04X}'")
    if code == 0 and out.strip():
        report(PASS, f"something is listening on device port {port}", "")
    else:
        report(
            INFO,
            f"no listener found on device port {port}",
            "The phone-side server may not be running yet. Not conclusive --\n"
            "/proc/net/tcp is restricted on some builds.",
        )


def step_endpoint(base_url: str, model: str) -> bool:
    body, latency_ms, error = http_post(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 16,
            "temperature": 0.2,
        },
    )
    if error:
        report(
            FAIL,
            "phone endpoint answers",
            f"{error}\n"
            "If the server is running but this fails, the model binary is most\n"
            "likely not pushed/loaded yet -- the expected state right now.",
        )
        return False
    try:
        text = body["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001
        report(FAIL, "response matches the contract", f"unexpected shape: {str(body)[:200]}")
        return False

    report(PASS, "phone endpoint answers", f"{latency_ms:.0f}ms -- {text[:120]!r}")

    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens and latency_ms > 0:
        tok_s = completion_tokens / (latency_ms / 1000)
        report(
            INFO,
            "throughput",
            f"{tok_s:.1f} tok/s ({completion_tokens} completion tokens).\n"
            "Public benchmarks put 3B/w4a16 on this chipset class near 10-13 tok/s "
            "-- a rough sanity bound, not a guarantee.",
        )
    return True


def step_router(base_url: str) -> bool:
    """The whole point: does TwoBrainRouter use the phone, and is what crosses
    the wire actually masked?"""
    os.environ["TWO_BRAIN_PHONE_BRAIN"] = "1"
    os.environ["TWO_BRAIN_PHONE_URL"] = base_url

    from two_brain_router.routing.router import TwoBrainRouter

    router = TwoBrainRouter(tier="mobile")
    brain = type(router.fast_brain).__name__
    if brain != "PhoneFastBrain":
        report(FAIL, "router selected the phone brain", f"got {brain}")
        return False
    report(PASS, "router selected the phone brain", f"{brain} -> {router.fast_brain.base_url}")

    # Capture exactly what leaves for the device. PhoneFastBrain builds the
    # request inside _post_chat_completion(prompt), so the prompt string is
    # what crosses the wire.
    sent: list = []
    original_post = router.fast_brain._post_chat_completion
    router.fast_brain._post_chat_completion = lambda prompt: (
        sent.append(prompt) or original_post(prompt)
    )

    probe = "email bob@example.com and call 555-123-4567 about lunch"
    try:
        decision = router.route(probe)
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "routed query completed", f"{type(exc).__name__}: {exc}")
        return False

    if not sent:
        report(FAIL, "the phone was actually called", "no request was captured")
        return False

    payload = str(sent[0])
    leaked = [s for s in ("bob@example.com", "555-123-4567") if s in payload]
    if leaked:
        report(FAIL, "PII masked before reaching the phone", f"LEAKED: {leaked}\n{payload}")
        return False
    report(PASS, "PII masked before reaching the phone", payload)

    report(
        PASS if decision.tier_answered == "local" else INFO,
        f"routed to: {decision.tier_answered}",
        f"masked {decision.pii_entities_masked} entities | {decision.est_latency_ms:.0f}ms\n"
        f"answer: {decision.answer[:160]}",
    )
    return True


def step_fallback() -> bool:
    """A dead phone must escalate, not crash. Needs no hardware, so it always
    runs -- it proves the demo survives the phone dropping out mid-show."""
    os.environ["TWO_BRAIN_PHONE_BRAIN"] = "1"
    os.environ["TWO_BRAIN_PHONE_URL"] = "http://127.0.0.1:9"

    from two_brain_router.routing.router import TwoBrainRouter

    router = TwoBrainRouter(tier="mobile")
    try:
        decision = router.route("what time zone is Tokyo in?")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "dead phone escalates instead of crashing", f"{type(exc).__name__}: {exc}")
        return False

    if decision.tier_answered != "cloud":
        report(FAIL, "dead phone escalates instead of crashing", f"got {decision.tier_answered}")
        return False
    report(PASS, "dead phone escalates instead of crashing", "fell through to the deep brain")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="port the phone serves on")
    parser.add_argument("--mock", action="store_true", help="skip adb; test a local server")
    parser.add_argument("--model", default="llama-3.2-3b-instruct")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    print(f"\nPhone brain live test -- target {base_url}\n" + "-" * 64)

    endpoint_ok = False
    if args.mock:
        report(SKIP, "adb / device checks", "--mock: assuming a local server")
        endpoint_ok = step_endpoint(base_url, args.model)
    else:
        adb_path = find_adb()
        if not adb_path:
            report(FAIL, "adb found", "Install platform-tools, or put adb on PATH.")
        else:
            report(PASS, "adb found", adb_path)
            serial = step_device(adb_path)
            if serial and step_forward(adb_path, serial, args.port):
                step_server_process(adb_path, serial, args.port)
                endpoint_ok = step_endpoint(base_url, args.model)

    if endpoint_ok:
        step_router(base_url)
    else:
        report(SKIP, "router integration", "the endpoint is not answering yet")

    step_fallback()
    return summarize()


def summarize() -> int:
    failed = [r for r in _results if r[0] == FAIL]
    passed = [r for r in _results if r[0] == PASS]
    print("\n" + "-" * 64)
    print(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\nFailed steps:")
        for _, step, _detail in failed:
            print(f"  - {step}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
