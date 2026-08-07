<#
.SYNOPSIS
    Launch the Two-Brain router API with every brain this machine can run.

.DESCRIPTION
    NPU at startup, all GPU models switchable from the UI's model picker, and
    the real cloud deep brain when a key is present.

    Every variable is defaulted in the CONFIG block below, so this runs with no
    prior setup. Anything already in the environment wins, so a one-off
    override is still just:

        $env:TWO_BRAIN_MODEL_ROOT = "D:\models"; .\run-server.ps1

    The one thing NOT defaulted here is INFERENCE_CLOUD_API_KEY. It is a
    credential; it belongs in secrets.txt, which is gitignored and must stay
    so. Without it the script warns and falls back to the stub deep brain.

    Separate from run.ps1 on purpose: that is the repo-convention *bootstrap*
    (creates a venv, installs deps) and is deliberately generic. This is a
    launcher, and it exists because getting this environment right by hand has
    gone wrong in several distinct ways, each guarded below.

.EXAMPLE
    .\run-server.ps1
    .\run-server.ps1 --tier mobile
    .\run-server.ps1 --port 8766 --no-trace
#>
[CmdletBinding()]
param(
    # Everything unrecognised is handed to api.py untouched (--tier, --port,
    # --host, --no-trace). Declared this way so the script never has to know
    # api.py's argument list.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ApiArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
Set-Location $Repo

function Note($msg) { Write-Host "  $msg" }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

# Assign only when the variable is absent or empty, so an exported value wins.
function Set-Default($name, $value) {
    $current = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($name, $value)
    }
}
function Get-Env($name) { [Environment]::GetEnvironmentVariable($name) }
function Clear-Env($name) { [Environment]::SetEnvironmentVariable($name, $null) }

# ==========================================================================
# CONFIG -- edit these, or set them in the environment before calling.
# Forward slashes throughout: pathlib accepts them on Windows and they avoid
# every backslash-escaping question.
# ==========================================================================

# --- where the weights and the llama.cpp build live -----------------------
# Both are gitignored and large, so they usually live in one checkout and are
# shared. MODEL_ROOT must be laid out like this repo's data/ (i.e. contain
# npu_model/ and vlm_gpu_model/).
Set-Default TWO_BRAIN_MODEL_ROOT "C:/Users/qc_de/dev/hollowbyte/Two-Brain-LLM/data"
Set-Default TWO_BRAIN_LLAMA_BIN  "C:/Users/qc_de/dev/hollowbyte/Two-Brain-LLM/.llama-cpp-opencl/extracted"

# --- which brains are real ------------------------------------------------
# NPU as the startup brain: its artifact is in this repo and loads in ~12s.
# GPU is deliberately NOT enabled here -- see the Clear-Env calls below.
Set-Default TWO_BRAIN_NPU_BRAIN "1"

# --- cloud ----------------------------------------------------------------
# Endpoint and model are configuration, not secrets. The key is not here.
Set-Default INFERENCE_CLOUD_ENDPOINT "https://aisuite.cirrascale.com/apis/v2"
Set-Default TWO_BRAIN_CLOUD_MODEL    "Llama-3.3-70B"

# --- Shape C --------------------------------------------------------------
# 1 = the local model returns {solution, confidence, unknown} and a named gap
# splits the query. 0 reverts both AI-PC brains to plain self-rated confidence.
Set-Default TWO_BRAIN_STRUCTURED "1"

# --- trace ----------------------------------------------------------------
# api.py traces by default (--no-trace to silence). REDACT=1 prints
# placeholders instead of real PII, at the cost of the thing the trace is for.
Set-Default TWO_BRAIN_GPU_LOG       (Join-Path $Repo "llama-server.log")
Set-Default TWO_BRAIN_TRACE_REDACT  "0"

# --- mobile tier (only used with --tier mobile) ---------------------------
# 127.0.0.1, never localhost: localhost resolves to ::1 first on this host
# while the servers bind IPv4 only, costing ~2s per call -- enough to blow the
# routing budget on its own. Port 8080 matches the adb forward in
# src/phone_brain/PHONE_DEPLOYMENT_GUIDE.md.
Set-Default TWO_BRAIN_PHONE_URL        "http://127.0.0.1:8080"
Set-Default TWO_BRAIN_PHONE_MODEL      "llama-3.2-3b-instruct"
Set-Default TWO_BRAIN_PHONE_MAX_TOKENS "512"

# ==========================================================================
# Guards -- each of these is a failure that actually happened.
# ==========================================================================

# UTF-8 stdout, and this is not cosmetic. trace.py prints raw user and model
# text; on Windows stdout defaults to cp1252, so one character outside that
# codepage (a maths symbol, an arrow, a curly quote -- ordinary model output)
# raises UnicodeEncodeError *inside the tracer*. It propagates out of the
# request and reaches the browser as an error, which the UI then reports as
# "the API wasn't reachable". Seen for real: 'charmap' codec can't encode
# character U+2248 turned a working image query into an apparent outage.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# TWO_BRAIN_GPU_MODEL/_MMPROJ are read *before* the path the catalogue passes
# in (os.environ.get(...) or model_path or ...), so if either is set, every GPU
# model picked in the UI silently loads that file instead. Worse on a VLM: the
# projector still comes from the catalogue, pairing one model's weights with
# another's vision encoder.
#
# TWO_BRAIN_GPU_BRAIN picks the *startup* brain, and that path resolves its
# model against this repo's own data/ -- ignoring TWO_BRAIN_MODEL_ROOT
# entirely. With the weights in another checkout it cannot succeed, and the
# server dies before it binds. Start on the NPU; the picker switches to GPU,
# and that path does honour MODEL_ROOT.
Clear-Env TWO_BRAIN_GPU_BRAIN
Clear-Env TWO_BRAIN_GPU_MODEL
Clear-Env TWO_BRAIN_GPU_MMPROJ

# Interpreter. .venv-npu first: on this machine it is the native-arm64
# environment carrying the Hexagon runtime. run.ps1 creates a plain .venv, so
# accept that too rather than assuming either exists.
$Py = $null
foreach ($candidate in @(
    (Join-Path $Repo ".venv-npu\Scripts\python.exe"),
    (Join-Path $Repo ".venv\Scripts\python.exe")
)) {
    if (Test-Path $candidate) { $Py = $candidate; break }
}
if (-not $Py) {
    Write-Host "ERROR: no virtualenv found. Run .\run.ps1 first, or create .venv-npu." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Get-Env TWO_BRAIN_MODEL_ROOT))) {
    Warn "model root does not exist: $(Get-Env TWO_BRAIN_MODEL_ROOT)"
}
if (-not (Test-Path (Join-Path (Get-Env TWO_BRAIN_LLAMA_BIN) "llama-server.exe"))) {
    Warn "llama-server.exe not found under $(Get-Env TWO_BRAIN_LLAMA_BIN)"
    Warn "  GPU models will fail when selected. The NPU model is unaffected."
}

# ==========================================================================
# Cloud credentials
# ==========================================================================
# Parsed rather than dot-sourced: secrets.txt is a KEY=value file, not a
# PowerShell script. Quotes are stripped, and so is any trailing CR -- a CRLF
# file would otherwise put a stray `r inside the key, which fails auth in a way
# that looks like a bad key rather than a bad file.
$SecretsPath = Join-Path $Repo "secrets.txt"
if (Test-Path $SecretsPath) {
    foreach ($line in Get-Content $SecretsPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $name = $Matches[1]
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Setting TWO_BRAIN_CLOUD_BRAIN=1 without both values makes
# CirrascaleDeepBrain raise in its constructor -- inside
# TwoBrainRouter.__init__, so the server would not start at all. Warning and
# falling back to the stub is strictly better than refusing to boot: every
# local tier still works and only escalations are simulated.
$HaveKey = -not [string]::IsNullOrWhiteSpace((Get-Env INFERENCE_CLOUD_API_KEY))
$HaveUrl = -not [string]::IsNullOrWhiteSpace((Get-Env INFERENCE_CLOUD_ENDPOINT))

if ($HaveKey -and $HaveUrl) {
    $env:TWO_BRAIN_CLOUD_BRAIN = "1"
    $CloudState = "real -- CirrascaleDeepBrain, $(Get-Env TWO_BRAIN_CLOUD_MODEL)"
} else {
    Clear-Env TWO_BRAIN_CLOUD_BRAIN
    $CloudState = "STUB -- escalations return canned text"
    Warn "no cloud API key; the deep brain will be the stub."
    if (Test-Path $SecretsPath) {
        Warn "  secrets.txt exists but does not define INFERENCE_CLOUD_API_KEY=<key>"
    } else {
        Warn "  create secrets.txt (gitignored) containing: INFERENCE_CLOUD_API_KEY=<key>"
    }
    Warn "  Escalations will read '[cloud:ai100 mock deep-brain response...]'."
}

# ==========================================================================
# Report, then launch
# ==========================================================================
Write-Host "Two-Brain router"
Note "python      : $Py"
Note "model root  : $(Get-Env TWO_BRAIN_MODEL_ROOT)"
Note "llama bin   : $(Get-Env TWO_BRAIN_LLAMA_BIN)"
Note "gpu log     : $(Get-Env TWO_BRAIN_GPU_LOG)"
Note "cloud       : $CloudState"
Note "structured  : $(Get-Env TWO_BRAIN_STRUCTURED) (Shape C)"

# What the picker will actually offer. Printed before launch because a model
# missing here is a missing *file*, and learning that now beats learning it 17s
# into a cold load.
Write-Host "  models      :"
$Probe = @'
from two_brain_router.routing.models import available_models
found = list(available_models())
for m in found:
    print("    {0:<12} {1}".format(m.id, m.label))
if not found:
    print("    NONE -- check TWO_BRAIN_MODEL_ROOT")
'@
try { & $Py -c $Probe } catch { Note "    (could not enumerate -- is the package installed?)" }

Write-Host ""
if ($ApiArgs) {
    & $Py -m two_brain_router.api @ApiArgs
} else {
    & $Py -m two_brain_router.api
}
exit $LASTEXITCODE
