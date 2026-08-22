# run.ps1 — Windows one-command launcher for the Product Rescue Mission pipeline
# Usage: .\run.ps1
# Run from the Datathon repo root directory.

$ErrorActionPreference = "Stop"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Product Rescue Mission — Pipeline Launcher" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host "`n[1/10] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 2: Discover flagship product + competitors (if not already present)
$dataFile = "data\clean\subcategory_sample.csv"
if (Test-Path $dataFile) {
    Write-Host "`n[2/10] Dataset found at $dataFile — skipping discovery." -ForegroundColor Green
} else {
    Write-Host "`n[2/10] Discovering flagship product + competitors from HuggingFace..." -ForegroundColor Yellow
    python setup_scripts\stream_data.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Data discovery failed."; exit 1 }
}

# Step 3: Product identity resolution (brand/title for flagship + competitors —
# must run BEFORE Day 3.5 so the competitor benchmark picks up resolved names
# instead of "Unknown")
Write-Host "`n[3/10] Resolving product identity (Day 3.7)..." -ForegroundColor Yellow
python day3.7_product_identity.py
if ($LASTEXITCODE -ne 0) { Write-Error "Product identity resolution failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 4: Sentiment analysis
Write-Host "`n[4/10] Running RoBERTa sentiment analysis (Day 1)..." -ForegroundColor Yellow
python day1_roberta_sentiment.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 1 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 5: Translation
Write-Host "`n[5/10] Running multilingual translation (Day 1.5)..." -ForegroundColor Yellow
python day1.5_translate.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 1.5 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 6: Aspect mining + emotion detection
Write-Host "`n[6/10] Running BERTopic aspect mining + emotion detection (Day 2)..." -ForegroundColor Yellow
python day2_aspects.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 2 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 7: Business logic
Write-Host "`n[7/10] Running business logic + CEO export (Day 3)..." -ForegroundColor Yellow
python day3_business_logic.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 3 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 8: ML suite (XGBoost churn model + anomaly/forecast/semantic search)
Write-Host "`n[8/10] Running ML model suite (Day 4)..." -ForegroundColor Yellow
python day4_ml_models.py
if ($LASTEXITCODE -ne 0) { Write-Error "Day 4 failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 9: Competitor benchmark (skips gracefully if no competitor data; reads
# the resolved brand names patched in by Day 3.7 above)
Write-Host "`n[9/10] Running competitor benchmark (Day 3.5)..." -ForegroundColor Yellow
python day3.5_competitor_benchmark.py
if ($LASTEXITCODE -ne 0) { Write-Error "Competitor benchmark failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 10: AI complaint summary
Write-Host "`n[10/10] Generating AI complaint summary (Day 3.6)..." -ForegroundColor Yellow
python day3.6_ai_complaint_summary.py
if ($LASTEXITCODE -ne 0) { Write-Error "AI complaint summary failed."; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Optional: Generate Jupyter notebook
if (Get-Command jupyter -ErrorAction SilentlyContinue) {
    Write-Host "`nGenerating Jupyter notebook..." -ForegroundColor Yellow
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
