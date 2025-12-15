# Fixed List Formatter - Proper Newline Handling
$files = @(
    "rl_course_presentation.md",
    "module2_content.md", 
    "module3_content.md",
    "modules_4_5_6_content.md",
    "modules_7_8_labs_final.md"
)

$fixCount = 0

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Processing: $file" -ForegroundColor Cyan
        
        $lines = Get-Content $file
        $newLines = @()
        $changed = $false
        
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $currentLine = $lines[$i]
            
            # Check if current line is a bold header ending with :
            if ($currentLine -match '^\*\*[^*]+:\*\*$') {
                # Check if next line exists and starts with - (bullet)
                if ($i+1 -lt $lines.Count -and $lines[$i+1] -match '^-\s') {
                    # Add current line, then blank line
                    $newLines += $currentLine
                    $newLines += ""
                    $changed = $true
                    continue
                }
            }
            
            # Also fix any literal `r`n`r`n sequences
            if ($currentLine -match '`r`n`r`n') {
                $currentLine = $currentLine -replace '`r`n`r`n', "`r`n`r`n"
                $changed = $true
            }
            
            $newLines += $currentLine
        }
        
        if ($changed) {
            $newLines | Set-Content $file
            $fixCount++
            Write-Host "  Fixed!" -ForegroundColor Green
        } else {
            Write-Host "  OK" -ForegroundColor Gray
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Fixed $fixCount file(s)" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Cyan
