# Fix Numbered Lists - Add line breaks between numbered items
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
        
        $content = Get-Content $file -Raw
        $originalContent = $content
        
        # Fix pattern: "1. Text 2. Text 3. Text" -> proper numbered list
        # Match numbered items: digit(s). followed by text, then another digit.
        $content = $content -replace '(\d+\.\s+[^0-9]+?)\s+(\d+\.)', "$1`r`n$2"
        
        if ($content -ne $originalContent) {
            Set-Content -Path $file -Value $content -NoNewline
            $fixCount++
            Write-Host "  Fixed numbered lists!" -ForegroundColor Green
        } else {
            Write-Host "  No numbered list issues" -ForegroundColor Gray
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Fixed $fixCount file(s)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Refresh browser to see changes!" -ForegroundColor Yellow
