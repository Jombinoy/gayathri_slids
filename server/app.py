from flask import Flask, render_template, jsonify, send_from_directory
import os
import markdown
import re
from pathlib import Path

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Base directory for presentation files
BASE_DIR = Path(__file__).parent.parent
DIAGRAMS_DIR = BASE_DIR / 'diagrams'

def parse_markdown_to_slides(md_content):
    """Parse markdown content into individual slides"""
    # Split by horizontal rules or slide markers
    slides = re.split(r'\n---\n|\n##\s+Slide\s+\d+:', md_content)
    
    parsed_slides = []
    for slide in slides:
        if slide.strip():
            # Extract speaker notes
            speaker_notes = ""
            if '**Speaker Notes:**' in slide:
                parts = slide.split('**Speaker Notes:**')
                slide_content = parts[0]
                speaker_notes = parts[1].strip() if len(parts) > 1 else ""
            else:
                slide_content = slide
            
            # Preprocess: ensure proper spacing for numbered lists
            slide_content = re.sub(r'(\d+\.\s+\*\*[^*]+\*\*[^\n]+)\n(\d+\.)', r'\1\n\n\2', slide_content)
            
            # Convert markdown to HTML with proper extensions
            html_content = markdown.markdown(
                slide_content,
                extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists']
            )
            
            parsed_slides.append({
                'content': html_content,
                'notes': speaker_notes
            })
    
    return parsed_slides

def load_all_presentations():
    """Load all markdown presentation files"""
    files = [
        'rl_course_presentation.md',
        'module2_content.md',
        'module3_content.md',
        'modules_4_5_6_content.md',
        'modules_7_8_labs_final.md'
    ]
    
    all_slides = []
    for filename in files:
        filepath = BASE_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                slides = parse_markdown_to_slides(content)
                all_slides.extend(slides)
    
    return all_slides

@app.route('/')
def index():
    """Serve the main presentation page"""
    return render_template('index.html')

@app.route('/api/slides')
def get_slides():
    """API endpoint to get all slides"""
    slides = load_all_presentations()
    return jsonify({
        'total': len(slides),
        'slides': slides
    })

@app.route('/diagrams/<path:filename>')
def serve_diagram(filename):
    """Serve diagram images"""
    return send_from_directory(DIAGRAMS_DIR, filename)

@app.route('/api/info')
def get_info():
    """Get presentation information"""
    return jsonify({
        'title': 'BCSE432E - Reinforcement Learning',
        'subtitle': 'Complete Course Presentation',
        'modules': 8,
        'labs': 20,
        'hours': {
            'lecture': 45,
            'lab': 30
        }
    })

if __name__ == '__main__':
    # Ensure templates directory exists
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting RL Presentation Server...")
    print("📊 Loading presentation content...")
    print(f"📁 Serving diagrams from: {DIAGRAMS_DIR}")
    print("\n✅ Server ready!")
    print("🌐 Open browser to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
