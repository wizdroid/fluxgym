param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,
    
    [Parameter(Mandatory=$true)]
    [string]$DestinationDir
)

# Function to display usage information
function Show-Usage {
    Write-Host "Usage: .\create-safetensors-symlinks.ps1 -SourceDir <path> -DestinationDir <path>"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -SourceDir       Source directory containing .safetensors files"
    Write-Host "  -DestinationDir  Destination directory where symlinks will be created"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\create-safetensors-symlinks.ps1 -SourceDir 'C:\models\source' -DestinationDir 'C:\models\dest'"
}

# Validate that source directory exists
if (-not (Test-Path -Path $SourceDir -PathType Container)) {
    Write-Error "Source directory does not exist: $SourceDir"
    Show-Usage
    exit 1
}

# Create destination directory if it doesn't exist
if (-not (Test-Path -Path $DestinationDir -PathType Container)) {
    Write-Host "Creating destination directory: $DestinationDir"
    try {
        New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    }
    catch {
        Write-Error "Failed to create destination directory: $($_.Exception.Message)"
        exit 1
    }
}

# Convert to absolute paths
$SourceDir = (Resolve-Path -Path $SourceDir).Path
$DestinationDir = (Resolve-Path -Path $DestinationDir).Path

Write-Host "Source Directory: $SourceDir"
Write-Host "Destination Directory: $DestinationDir"
Write-Host ""

# Find all .safetensors files in source directory (including subdirectories)
$safetensorsFiles = Get-ChildItem -Path $SourceDir -Filter "*.safetensors" -Recurse -File

if ($safetensorsFiles.Count -eq 0) {
    Write-Warning "No .safetensors files found in source directory: $SourceDir"
    exit 0
}

Write-Host "Found $($safetensorsFiles.Count) .safetensors file(s)"
Write-Host ""

$successCount = 0
$errorCount = 0

foreach ($file in $safetensorsFiles) {
    # Calculate relative path from source directory
    $relativePath = $file.FullName.Substring($SourceDir.Length + 1)
    
    # Destination file path
    $destFile = Join-Path -Path $DestinationDir -ChildPath $relativePath
    
    # Create subdirectories in destination if needed
    $destFileDir = Split-Path -Path $destFile -Parent
    if (-not (Test-Path -Path $destFileDir -PathType Container)) {
        try {
            New-Item -ItemType Directory -Path $destFileDir -Force | Out-Null
        }
        catch {
            Write-Error "Failed to create directory: $destFileDir - $($_.Exception.Message)"
            $errorCount++
            continue
        }
    }
    
    # Check if symlink already exists
    if (Test-Path -Path $destFile) {
        # Check if it's already a symlink pointing to the correct target
        $item = Get-Item -Path $destFile
        if ($item.LinkType -eq "SymbolicLink" -and $item.Target -eq $file.FullName) {
            Write-Host "Symlink already exists: $relativePath" -ForegroundColor Yellow
            $successCount++
            continue
        }
        else {
            Write-Warning "File already exists (not a symlink or points to different target): $relativePath"
            $errorCount++
            continue
        }
    }
    
    # Create symbolic link
    try {
        New-Item -ItemType SymbolicLink -Path $destFile -Target $file.FullName | Out-Null
        Write-Host "Created symlink: $relativePath" -ForegroundColor Green
        $successCount++
    }
    catch {
        Write-Error "Failed to create symlink for $relativePath - $($_.Exception.Message)"
        $errorCount++
    }
}

Write-Host ""
Write-Host "Summary:"
Write-Host "  Successfully created: $successCount symlinks"
if ($errorCount -gt 0) {
    Write-Host "  Errors: $errorCount" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "  All symlinks created successfully!" -ForegroundColor Green
    exit 0
}