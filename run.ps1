# run.ps1 — Windows one-command launcher for the Product Rescue Mission pipeline
# Usage: .\run.ps1
# Run from the Datathon repo root directory.

$ErrorActionPreference = "Stop"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Product Rescue Mission — Pipeline Launcher" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host "`n[1/6] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 2: Stream dataset (if not already present)
$dataFile = "data\clean\subcategory_sample.csv"
if (Test-Path $dataFile) {
    Write-Host "`n[2/6] Dataset found at $dataFile — skipping stream." -ForegroundColor Green
} else {
    Write-Host "`n[2/6] Streaming 50,000 reviews from HuggingFace..." -ForegroundColor Yellow
    python setup_scripts\stream_data.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Data streaming failed."; exit 1 }
}

# Step 3: Sentiment analysis
Write-Host "`n[3/6] Running RoBERTa sentiment analysis (Day 1)..." -ForegroundColor Yellow
python day1_roberta_sentiment.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 1 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 4: Translation
Write-Host "`n[4/6] Running multilingual translation (Day 1.5)..." -ForegroundColor Yellow
python day1.5_translate.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 1.5 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 5: Aspect mining + emotion detection
Write-Host "`n[5/6] Running BERTopic aspect mining + emotion detection (Day 2)..." -ForegroundColor Yellow
python day2_aspects.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 2 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 6: Business logic
Write-Host "`n[6/7] Running business logic + CEO export (Day 3)..." -ForegroundColor Yellow
python day3_business_logic.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 3 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Optional: Generate Jupyter notebook
if (Get-Command jupyter -ErrorAction SilentlyContinue) {
    Write-Host "`n[7/7] Generating Jupyter notebook..." -ForegroundColor Yellow
    pip install nbformat -q
    python notebooks\make_notebook.py
    Write-Host "      Notebook: notebooks\product_rescue_analysis.ipynb" -ForegroundColor Green
}

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host " All pipeline steps complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "`nLaunching Streamlit dashboard..." -ForegroundColor Yellow
Write-Host "   Open: http://localhost:8501" -ForegroundColor Cyan
streamlit run gui\app.py
