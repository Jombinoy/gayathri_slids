# RL Course Presentation Server

## 🚀 Quick Start

### **Option 1: Double-Click to Start (Easiest)**
Simply double-click: **`START_SERVER.bat`**

The server will:
1. Install dependencies automatically
2. Start the Flask server
3. Open at http://localhost:5000

### **Option 2: Manual Start**
```powershell
# Install dependencies
pip install -r server/requirements.txt

# Start server
cd server
python app.py
```

---

## 🌐 **Access the Presentation**

Once the server is running:
1. Open browser to: **http://localhost:5000**
2. The presentation will load all 146 slides automatically
3. Navigate with arrow keys or spacebar

---

## ⌨️ **Controls**

| Key | Action |
|-----|--------|
| **Arrow Keys** | Navigate slides |
| **Space** | Next slide |
| **Shift + Space** | Previous slide |
| **S** | Speaker notes view |
| **O** or **Esc** | Overview mode |
| **F** | Fullscreen |
| **H** | Home (first slide) |
| **?** | Help menu |

---

## 📊 **Features**

✅ **All 146 Slides** loaded dynamically  
✅ **21 Diagrams** served properly  
✅ **Speaker Notes** on every slide  
✅ **Professional Theme** with smooth transitions  
✅ **Slide Numbers** and progress tracking  
✅ **Code Highlighting** for algorithms  
✅ **Responsive Design** works on any screen  

---

## 🔧 **API Endpoints**

### `GET /`
- Serves the main presentation page

### `GET /api/slides`
- Returns all slides as JSON
- Response:
```json
{
  "total": 146,
  "slides": [
    {
      "content": "<html content>",
      "notes": "speaker notes"
    }
  ]
}
```

### `GET /api/info`
- Returns presentation metadata
- Response:
```json
{
  "title": "BCSE432E - Reinforcement Learning",
  "modules": 8,
  "labs": 20,
  "hours": {"lecture": 45, "lab": 30}
}
```

### `GET /diagrams/<filename>`
- Serves diagram images from `diagrams/` folder

---

## 📁 **Project Structure**

```
gayathri_slids/
├── server/
│   ├── app.py              # Flask API server
│   ├── templates/
│   │   └── index.html      # Reveal.js frontend
│   └── requirements.txt    # Python dependencies
├── diagrams/               # All 21 diagram images
├── *.md                    # Markdown presentation files
├── START_SERVER.bat        # Easy launcher (Windows)
└── SERVER_README.md        # This file
```

---

## 🎯 **How It Works**

1. **Flask Server** (`app.py`):
   - Parses all 5 markdown files
   - Extracts slides and speaker notes
   - Converts markdown to HTML
   - Serves diagrams as static files
   - Provides REST API endpoints

2. **Frontend** (`index.html`):
   - Fetches slides from `/api/slides`
   - Dynamically creates reveal.js slides
   - Fixes image paths to use `/diagrams/` endpoint
   - Adds speaker notes as aside elements
   - Initializes reveal.js with all features

3. **Reveal.js**:
   - Handles slide navigation
   - Provides speaker notes view (press 'S')
   - Shows overview mode (press 'O')
   - Supports keyboard shortcuts

---

## 🔌 **Accessing from Other Devices**

To access from another computer on the same network:

1. Find your IP address:
```powershell
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)
```

2. On other device, open browser to:
```
http://YOUR_IP:5000
# Example: http://192.168.1.100:5000
```

---

## 🛠️ **Troubleshooting**

### Port 5000 already in use?
Edit `server/app.py`, change the last line:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use port 8080
```

### Images not loading?
- Ensure `diagrams/` folder contains all 21 PNG files
- Check browser console (F12) for errors
- Verify paths in markdown files use `diagrams/filename.png`

### Slides not loading?
- Check server console for errors
- Ensure all `.md` files exist in parent directory
- Open http://localhost:5000/api/slides to see raw data

---

## 🎓 **For Presenting**

### **Setup Before Class:**
1. Start server: `START_SERVER.bat`
2. Open browser to http://localhost:5000
3. Press 'F' for fullscreen
4. Press 'S' to open speaker notes view (shows notes + next slide)

### **During Class:**
- Use **Arrow Keys** or **Spacebar** to navigate
- Press **'S'** to see your speaker notes (students don't see these)
- Press **'O'** for overview to jump to specific slide
- Press **'B'** or **'.'** to pause (black screen)

### **After Class:**
- Share the URL with students if on local network
- Or export to PDF: Add `?print-pdf` to URL, then print to PDF

---

## 📤 **Sharing the Presentation**

### **Option 1: Send Files**
Zip the entire `gayathri_slids` folder and send

### **Option 2: Host Online**
Deploy to:
- **Heroku** (free tier)
- **PythonAnywhere** (free tier)
- **Render** (free tier)
- **Railway** (free tier)

### **Option 3: Export to PDF**
1. Add `?print-pdf` to URL: http://localhost:5000?print-pdf
2. Print to PDF from browser
3. Share PDF file

---

## ✨ **Advantages Over PowerPoint**

✅ **No Software Needed** - Just a web browser  
✅ **Works Anywhere** - Windows, Mac, Linux, tablets  
✅ **Easy Sharing** - Send URL to students  
✅ **Professional** - Modern, smooth transitions  
✅ **Interactive** - Live server, dynamic loading  
✅ **Portable** - Entire presentation in one folder  

---

## 🎉 **You're Ready!**

Your complete RL course presentation is now:
- ✅ Served via professional web server
- ✅ Accessible from any browser
- ✅ Including all 146 slides
- ✅ With all 21 diagrams
- ✅ With speaker notes
- ✅ Ready to present!

**Just run `START_SERVER.bat` and open http://localhost:5000** 🚀
