param(
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

& $ProjectPython project.py data audit
if ($LASTEXITCODE -ne 0) { throw "Data audit failed." }

$ModelNames = @("lstm", "gru")
$ExperimentSeeds = @(42, 43, 44)
foreach ($ModelName in $ModelNames) {
    foreach ($ExperimentSeed in $ExperimentSeeds) {
        $RunDirectory = "runs/$ModelName-seed$ExperimentSeed"
        $Arguments = @(
            "project.py", "train",
            "--model", $ModelName,
            "--run-dir", $RunDirectory,
            "--device", $Device,
            "--seed", $ExperimentSeed
        )
        if (Test-Path -LiteralPath "$RunDirectory/config.json") {
            $Arguments += "--resume"
        }
        & $ProjectPython @Arguments
        if ($LASTEXITCODE -ne 0 -and -not (Test-Path -LiteralPath "$RunDirectory/best.pt")) {
            throw "Training failed before producing a checkpoint: $RunDirectory"
        }
        & $ProjectPython project.py evaluate --run-dir $RunDirectory --device cpu
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $RunDirectory" }
    }
}

& $ProjectPython project.py aggregate `
    --protocol configs/experiment_protocol_v1.json `
    --output runs/multiseed_summary.json `
    --force `
    --run-dir runs/lstm-seed42 `
    --run-dir runs/lstm-seed43 `
    --run-dir runs/lstm-seed44 `
    --run-dir runs/gru-seed42 `
    --run-dir runs/gru-seed43 `
    --run-dir runs/gru-seed44
if ($LASTEXITCODE -ne 0) { throw "Aggregation failed." }
