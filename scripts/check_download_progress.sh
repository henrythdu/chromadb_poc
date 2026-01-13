#!/bin/bash
# Monitor the progress of the 500 paper download

LOG_FILE="logs/download_500_papers.out"
PID_FILE="logs/download.pid"

# Get the PID if running
PID=$(ps aux | grep "download_500_papers.py" | grep -v grep | awk '{print $2}')

echo "========================================"
echo "ArXiv Download Progress Monitor"
echo "========================================"
echo "Time: $(date)"
echo ""

if [ -z "$PID" ]; then
    echo "❌ Download process is NOT running"
    echo ""
else
    echo "✅ Download process is RUNNING (PID: $PID)"
    echo ""
fi

# Count downloaded PDFs
PDF_COUNT=$(ls -1 ml_pdfs/*.pdf 2>/dev/null | wc -l)
echo "📄 Papers downloaded: $PDF_COUNT / 500"

# Check if metadata file exists
if [ -f "data/downloaded_papers_500.json" ]; then
    echo "📋 Metadata file: EXISTS"
    METADATA_COUNT=$(python3 -c "import json; print(len(json.load(open('data/downloaded_papers_500.json'))))" 2>/dev/null || echo "0")
    echo "   Records in metadata: $METADATA_COUNT"
else
    echo "📋 Metadata file: NOT YET CREATED"
fi

# Show last few log lines
echo ""
echo "📝 Recent log entries:"
echo "----------------------------------------"
tail -5 "$LOG_FILE" 2>/dev/null || echo "Log file not found yet"

echo ""
echo "========================================"

# If process completed, show summary
if [ -z "$PID" ] && [ -f "data/downloaded_papers_500.json" ]; then
    echo ""
    echo "🎉 DOWNLOAD COMPLETE!"
    echo "Total papers: $METADATA_COUNT"
    echo "PDF files: $PDF_COUNT"
    echo "Metadata: data/downloaded_papers_500.json"
fi
