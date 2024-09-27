$Env:CONDA_EXE = "/home/ruqi/projects/MVSFormerPlusPlus/yes/bin/conda"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "/home/ruqi/projects/MVSFormerPlusPlus/yes"
$Env:_CONDA_EXE = "/home/ruqi/projects/MVSFormerPlusPlus/yes/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs