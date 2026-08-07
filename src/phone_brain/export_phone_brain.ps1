<#
.SYNOPSIS
    Exports Llama-3.2-3B-Instruct to a Genie-ready QNN bundle for Snapdragon 8 Elite.

.DESCRIPTION
    Windows/PowerShell port of export_phone_brain.sh.

    Targets the Snapdragon 8 Elite (Galaxy S25's SoC) at the officially-supported
    w4a16 precision (4-bit weights / 16-bit activations, sensitive layers at w8a16,
    8-bit KV cache).

    Why w4a16 and not w4a8: as of this writing, Qualcomm's published Llama recipes
    for the Genie/QNN path ship at w4a16 -- that's what's benchmarked and supported
    for this chipset. True W4A8 isn't an offered preset for these LLM exports;
    getting there would mean a custom AIMET PTQ recipe outside this script. Confirm
    what your installed qai_hub_models version actually exposes before changing it:

        python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --help

.PARAMETER OutputDir
    Destination directory for the exported bundle. Default: genie_bundle_l_phone

.PARAMETER ContextLength
    Sequence context length. Trim further if you hit device memory pressure.
    Default: 2048, or the CONTEXT_LEN environment variable if set.

.PARAMETER SkipChecks
    Skip the preflight environment checks and go straight to the export.

.EXAMPLE
    .\export_phone_brain.ps1

.EXAMPLE
    .\export_phone_brain.ps1 -OutputDir genie_bundle_l_phone -ContextLength 1024

.NOTES
    PREREQS (one-time, on your dev machine -- not the phone):

      1. Python 3.10-3.13, then:
           pip install "qai-hub-models[llama-v3-2-3b-instruct]"

      2. Qualcomm AI Hub account at https://aihub.qualcomm.com, grab an API token
         from Account -> Settings -> API Token, then:
           qai-hub configure --api_token <YOUR_AI_HUB_TOKEN>

      3. Llama 3.2 weights are gated on Hugging Face. Request access to
         meta-llama/Llama-3.2-3B-Instruct, then on this machine:
           hf auth login          # older versions: huggingface-cli login

    If PowerShell refuses to run this file, either unblock it for the session:
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    or invoke it directly:
        powershell -ExecutionPolicy Bypass -File .\export_phone_brain.ps1
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $OutputDir = 'genie_bundle_l_phone',

    [int] $ContextLength = $(if ($env:CONTEXT_LEN) { [int] $env:CONTEXT_LEN } else { 2048 }),

    [switch] $SkipChecks
)

# Rough equivalent of `set -euo pipefail`. Note that $ErrorActionPreference does
# not apply to native executables in Windows PowerShell 5.1, so exit codes from
# python are checked explicitly below.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Chipset = 'qualcomm-snapdragon-8-elite'   # Galaxy S25's SoC
$Module  = 'qai_hub_models.models.llama_v3_2_3b_instruct.export'


function Write-Section {
    param([string] $Text)
    Write-Host ('=' * 66) -ForegroundColor DarkGray
    Write-Host $Text
    Write-Host ('=' * 66) -ForegroundColor DarkGray
}

function Test-PythonImport {
    <#
        Returns $true if the module imports cleanly.

        In Windows PowerShell 5.1, redirecting a native command's stderr (2>) while
        $ErrorActionPreference is 'Stop' promotes that stderr text to a terminating
        NativeCommandError -- so a failing import would kill the script instead of
        returning a value. Dropping to 'Continue' for the duration of the call, and
        merging stderr into stdout with 2>&1, avoids that.
    #>
    param([string] $ModuleName)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & python -c "import $ModuleName" 2>&1
        return [pscustomobject]@{
            Ok     = ($LASTEXITCODE -eq 0)
            Output = ($output | Out-String).Trim()
        }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Prereqs {
    Write-Host 'Running preflight checks...' -ForegroundColor Cyan

    # Python present and reachable?
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "python not found on PATH. Activate your virtualenv first (.\.venv\Scripts\Activate.ps1)."
    }
    Write-Host "  [ok] python -> $($python.Source)"

    # Warn if this looks like a system Python rather than a venv.
    if (-not $env:VIRTUAL_ENV) {
        Write-Warning '  VIRTUAL_ENV is not set -- you may be exporting against a system Python.'
    }

    # Is the export module importable?
    $import = Test-PythonImport -ModuleName 'qai_hub_models'
    if (-not $import.Ok) {
        Write-Host ''
        Write-Host '  [fail] import qai_hub_models raised:' -ForegroundColor Red
        Write-Host $import.Output
        Write-Host ''
        if ($import.Output -match 'No module named') {
            throw 'qai_hub_models is not installed. Run: pip install "qai-hub-models[llama-v3-2-3b-instruct]"'
        }
        throw 'qai_hub_models is installed but failed to import -- see the traceback above.'
    }
    Write-Host '  [ok] qai_hub_models importable'

    # AI Hub token configured? This is the ~/.qai_hub/client.ini the SDK looks for.
    $hubConfig = Join-Path $env:USERPROFILE '.qai_hub\client.ini'
    if (-not (Test-Path $hubConfig)) {
        throw "AI Hub config not found at $hubConfig. Run: qai-hub configure --api_token <YOUR_TOKEN>"
    }
    Write-Host "  [ok] AI Hub config at $hubConfig"

    # Hugging Face credentials present? Llama 3.2 is a gated repo.
    $hfHome  = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $env:USERPROFILE '.cache\huggingface' }
    $hfToken = Join-Path $hfHome 'token'
    if (-not $env:HF_TOKEN -and -not (Test-Path $hfToken)) {
        Write-Warning '  No Hugging Face token found. Llama-3.2-3B-Instruct is gated; run `hf auth login` if the export 401s.'
    }
    else {
        Write-Host '  [ok] Hugging Face credentials present'
    }

    Write-Host ''
}


if (-not $SkipChecks) { Test-Prereqs }

Write-Section @"
 Exporting Llama-3.2-3B-Instruct  ->  Genie/QNN bundle
 Chipset:        $Chipset
 Context length: $ContextLength
 Output dir:     $OutputDir
"@

python -m $Module `
    --chipset $Chipset `
    --context-length $ContextLength `
    --skip-profiling `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Error "Export failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

$resolved = (Resolve-Path $OutputDir -ErrorAction SilentlyContinue)
Write-Host ''
Write-Host "==> Export complete: $(if ($resolved) { $resolved.Path } else { $OutputDir })" -ForegroundColor Green
Write-Host ''
Write-Host 'Next steps:'
Write-Host '  1. Connect the S25 with USB debugging enabled, then:'
Write-Host "       adb push $OutputDir /data/local/tmp/genie_bundle_l"
Write-Host '  2. Install/serve via GenieX or the raw Genie/QAIRT runtime on-device.'
Write-Host '  3. Run bench_phone_brain.py against the served endpoint to sanity-check'
Write-Host '     latency and tokens/sec before wiring in the orchestrator.'