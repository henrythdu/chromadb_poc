#!/bin/bash
# Wait for download to complete and show final summary

echo "Waiting for download to complete..."
echo "Press Ctrl+C to stop waiting (download will continue in background)"
echo ""

while true; do
    PID=$(ps aux | grep "download_500_papers.py" | grep -v grep | awk '{print $2}')

    if [ -z "$PID" ]; then
        echo ""
        echo "========================================"
        echo "🎉 DOWNLOAD PROCESS COMPLETE!"
        echo "========================================"
        echo ""

        # Final count
        PDF_COUNT=$(ls -1 ml_pdfs/*.pdf 2>/dev/null | wc -l)
        echo "📄 Total PDFs downloaded: $PDF_COUNT"

        # Check metadata
        if [ -f "data/downloaded_papers_500.json" ]; then
            METADATA_COUNT=$(python3 -c "import json; print(len(json.load(open('data/downloaded_papers_500.json'))))" 2>/dev/null || echo "N/A")
            echo "📋 Metadata records: $METADATA_COUNT"
            echo "📁 Metadata file: data/downloaded_papers_500.json"

            # Show summary
            echo ""
            echo "Sample paper titles:"
            echo "----------------------------------------"
            python3 -c "
import json
with open('data/downloaded_papers_500.json') as f:
    papers = json.load(f)
    for i, p in enumerate(papers[:5], 1):
        print(f'{i}. {p[\"title\"][:80]}...')
            " 2>/dev/null
        else
            echo "⚠️  Metadata file not found - checking logs..."
            echo ""
            echo "Last 10 log entries:"
            tail -10 logs/download_500_papers.out
        fi

        echo ""
        echo "========================================"
        echo "Download directory: ml_pdfs/"
        echo "Log file: logs/download_500_papers.out"
        echo "========================================"
        break
    fi

    # Show progress every 10 seconds
    PDF_COUNT=$(ls -1 ml_pdfs/*.pdf 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Downloading... $PDF_COUNT papers downloaded so far"

    sleep 10
done
