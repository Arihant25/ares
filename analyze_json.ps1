# PowerShell script to analyze JSON structure and display hierarchy with counts

$jsonPath = "c:\Users\Arihant\Git\ares\data\finetuning.json"

# Read and parse the JSON file
$json = Get-Content $jsonPath | ConvertFrom-Json

# Build the nested structure
$structure = @{}

if ($json.Count -gt 0) {
    $firstItem = $json[0]
    
    # Level object
    $levelObj = @{}
    foreach ($field in $firstItem.PSObject.Properties.Name) {
        if ($field -eq "chapters") {
            $levelObj[$field] = @()
        } else {
            $levelObj[$field] = "value"
        }
    }
    
    # Chapter object
    if ($firstItem.chapters -and $firstItem.chapters.Count -gt 0) {
        $firstChapter = $firstItem.chapters[0]
        $chapterObj = @{}
        foreach ($field in $firstChapter.PSObject.Properties.Name) {
            if ($field -eq "concepts") {
                $chapterObj[$field] = @()
            } else {
                $chapterObj[$field] = "value"
            }
        }
        $levelObj["chapters"] = @($chapterObj)
        
        # Concept object
        if ($firstChapter.concepts -and $firstChapter.concepts.Count -gt 0) {
            $firstConcept = $firstChapter.concepts[0]
            $conceptObj = @{}
            foreach ($field in $firstConcept.PSObject.Properties.Name) {
                if ($field -eq "misconceptions") {
                    $conceptObj[$field] = @()
                } else {
                    $conceptObj[$field] = "value"
                }
            }
            $chapterObj["concepts"] = @($conceptObj)
            
            # Misconception object
            if ($firstConcept.misconceptions -and $firstConcept.misconceptions.Count -gt 0) {
                $firstMisconception = $firstConcept.misconceptions[0]
                $misconceptionObj = @{}
                foreach ($field in $firstMisconception.PSObject.Properties.Name) {
                    if ($field -eq "socratic_sequence") {
                        $misconceptionObj[$field] = @("value")
                    } else {
                        $misconceptionObj[$field] = "value"
                    }
                }
                $conceptObj["misconceptions"] = @($misconceptionObj)
            }
        }
    }
    
    $structure = @($levelObj)
}

$structure | ConvertTo-Json -Depth 10

# Initialize counters
$totalLevels = 0
$totalChapters = 0
$totalConcepts = 0
$totalMisconceptions = 0

Write-Host "=== JSON Structure Analysis ===" -ForegroundColor Cyan
Write-Host ""

# Iterate through each level
foreach ($item in $json) {
    $level = $item.level
    $title = $item.title
    $chapterCount = $item.chapters.Count
    
    $totalLevels++
    $totalChapters += $chapterCount
    
    Write-Host "Level $level - $title" -ForegroundColor Yellow
    Write-Host "  Chapters: $chapterCount" -ForegroundColor Green
    
    # Iterate through each chapter in the level
    foreach ($chapter in $item.chapters) {
        $topic = $chapter.topic
        $conceptCount = $chapter.concepts.Count
        
        $totalConcepts += $conceptCount
        
        Write-Host "    ├─ $topic" -ForegroundColor Cyan
        Write-Host "       └─ Concepts: $conceptCount" -ForegroundColor Gray
        
        # Count misconceptions within each concept
        $levelMisconceptions = 0
        foreach ($concept in $chapter.concepts) {
            if ($concept.misconceptions) {
                $misconceptionCount = $concept.misconceptions.Count
                $levelMisconceptions += $misconceptionCount
                $totalMisconceptions += $misconceptionCount
            }
        }
        
        if ($levelMisconceptions -gt 0) {
            Write-Host "          └─ Misconceptions: $levelMisconceptions" -ForegroundColor Magenta
        }
    }
    
    Write-Host ""
}

# Print summary
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Total Levels: $totalLevels" -ForegroundColor Yellow
Write-Host "Total Chapters: $totalChapters" -ForegroundColor Yellow
Write-Host "Total Concepts: $totalConcepts" -ForegroundColor Yellow
Write-Host "Total Misconceptions: $totalMisconceptions" -ForegroundColor Magenta
