from fastapi import FastAPI, Response
import requests
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
from reportlab.pdfgen import canvas
from io import BytesIO

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
FIGMA_TOKEN = os.getenv("FIGMA_TOKEN")
FILE_KEY = os.getenv("FILE_KEY")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Tessa FastAPI Server is Running"}

@app.get("/models")
def list_gemini_models():
    """List available Gemini models"""
    if not GEMINI_API_KEY:
        return {"error": "Missing GEMINI_API_KEY"}
    
    try:
        models = genai.list_models()
        model_list = []
        for m in models:
            model_list.append({
                "name": m.name,
                "display_name": m.display_name,
                "supported_methods": m.supported_generation_methods
            })
        return {"models": model_list}
    except Exception as e:
        return {"error": str(e)}


@app.get("/fetch")
def get_figma_json():
    """Fetch and print the Figma MCP JSON"""
    if not FIGMA_TOKEN or not FILE_KEY:
        return {"error": "Missing FIGMA_TOKEN or FILE_KEY in .env"}

    url = f"https://api.figma.com/v1/files/{FILE_KEY}"
    headers = {"X-Figma-Token": FIGMA_TOKEN}

    print("Fetching Figma MCP JSON...")
    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        data = res.json()
        print("MCP JSON Output:")
        print(json.dumps(data, indent=2))
        return {"status": "ok", "message": "MCP JSON printed to console"}
    else:
        print(f"Error {res.status_code}: {res.text}")
        return {"error": f"Failed to fetch JSON - {res.status_code}"}


@app.get("/analysis")
def analyze_figma():
    if not FIGMA_TOKEN or not FILE_KEY or not GEMINI_API_KEY:
        return {"error": "Missing env vars"}

    # 1) fetch figma
    url = f"https://api.figma.com/v1/files/{FILE_KEY}"
    headers = {"X-Figma-Token": FIGMA_TOKEN}
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return {"error": "failed to fetch figma"}

    figma_json = res.json()

    # Extract key information from Figma JSON to reduce token usage
    def extract_figma_summary(data):
        summary = {
            "name": data.get("name", "Unknown"),
            "version": data.get("version", "Unknown"),
            "pages": []
        }
        
        # Extract document structure
        doc = data.get("document", {})
        for page in doc.get("children", []):
            page_info = {
                "name": page.get("name"),
                "type": page.get("type"),
                "frames": []
            }
            
            # Extract frames/screens
            for child in page.get("children", [])[:20]:  # Limit to first 20 frames
                frame_info = {
                    "name": child.get("name"),
                    "type": child.get("type"),
                    "background": child.get("backgroundColor"),
                }
                page_info["frames"].append(frame_info)
            
            summary["pages"].append(page_info)
        
        # Extract styles
        summary["styles"] = {
            "colors": list(data.get("styles", {}).keys())[:10] if data.get("styles") else [],
            "text_styles": list(data.get("componentSets", {}).keys())[:10] if data.get("componentSets") else []
        }
        
        return summary

    figma_summary = extract_figma_summary(figma_json)

    # 2) gemini analysis - using streaming API with summarized data
    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
    prompt = f"""
    You are a UI/UX design analyst. I will give you a summarized Figma file structure.
    Extract useful insights and provide:
    
    1. **Screens Summary**: List all pages and main frames/screens found
    2. **Design Patterns**: Identify common patterns in naming and structure
    3. **Color Usage**: Analyze color themes if present
    4. **Components**: List repeated components or frames
    5. **UX Observations**: Note any structural issues or good practices
    6. **Recommendations**: Provide 3-5 actionable improvement suggestions
    
    Respond as a structured analysis report with clear sections and bulletpoints.
    Be specific and actionable.
    
    Here is the Figma file summary:
    {json.dumps(figma_summary, indent=2)}
    """
    
    try:
        # Use streaming generation and collect the response
        response = model.generate_content(prompt, stream=True)
        analysis_text = ""
        for chunk in response:
            if chunk.text:
                analysis_text += chunk.text
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

    # 3) build pdf in RAM with better formatting
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    
    # Add title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 800, "Figma Design Analysis Report")
    
    # Add content
    pdf.setFont("Helvetica", 10)
    y_position = 770
    
    for line in analysis_text.split("\n"):
        if y_position < 50:  # Start new page if near bottom
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y_position = 800
        
        # Truncate long lines to fit page width
        if len(line) > 90:
            line = line[:87] + "..."
        
        pdf.drawString(40, y_position, line)
        y_position -= 15
    
    pdf.save()
    buffer.seek(0)

    # 4) send pdf binary
    return Response(
        content=buffer.getvalue(), 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=figma_analysis.pdf"}
    )