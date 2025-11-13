from flask import Flask, render_template, request, send_file
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from langdetect import detect, LangDetectException
import nltk
import os
import re
from datetime import datetime
from fpdf import FPDF
import language_tool_python
from PyPDF2 import PdfReader
import pandas as pd

# ✅ Grammar checker initialize
print("⏳ Initializing Grammar Checker... (first time only)")
tool = language_tool_python.LanguageTool('en-US')
print("✅ Grammar Checker Ready!")

# ✅ NLTK setup
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

app = Flask(__name__)

# ---------- Utility Functions ----------

def summarize_text(text, length):
    """Summarize given text into a specific number of sentences."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, length)
    return " ".join(str(s) for s in summary_sentences)

def extract_text_pypdf2(file_storage):
    """Extract text from normal text-based PDFs."""
    text = ""
    reader = PdfReader(file_storage)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text.strip()

def clean_text(text):
    """Improve indentation & readability."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[.?!])\s+', '.\n\n', text)
    text = re.sub(r'(?<=;)\s*', '\n', text)
    text = re.sub(r'(?<=\{)\s*', '\n', text)
    text = re.sub(r'\s*(?=\})', '\n', text)
    return text.strip()

def batch_summarize(long_text, batch_size=4000):
    """Handle long PDFs by summarizing in smaller chunks."""
    summaries = []
    for i in range(0, len(long_text), batch_size):
        chunk = long_text[i:i + batch_size]
        summary = summarize_text(chunk, 8)
        summaries.append(summary)
    return "\n".join(summaries)

# ---------- Routes ----------
# HOMEPAGE ROUTE
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


# ABOUT PAGE ROUTE
@app.route("/about")
def about():
    return render_template("about.html")


# 🧠 TEXT SUMMARIZER
@app.route("/summarize", methods=["GET", "POST"])
def text_summarizer():
    summary = None
    original_text = None
    length_option = "5"
    language = "Unknown"
    word_count = 0
    reading_time = 0

    if request.method == "POST":
        # Get text or file input
        text_input = request.form.get("text", "").strip()
        uploaded_file = request.files.get("file")
        length_option = request.form.get("length", "5")

        # If file uploaded
        if uploaded_file and uploaded_file.filename.lower().endswith(".txt"):
            text_input = uploaded_file.read().decode("utf-8")

        if not text_input:
            return render_template("summarize.html", summary="⚠️ Please enter or upload some text.", original="")

        original_text = text_input

        try:
            # Detect language
            try:
                language = detect(text_input)
            except LangDetectException:
                language = "Unknown"

            # Summarize text
            summary = summarize_text(text_input, int(length_option))
            summary = clean_text(summary)

            # Stats
            word_count = len(text_input.split())
            reading_time = round(word_count / 200, 2)  # avg 200 wpm

        except Exception as e:
            summary = f"⚠️ Error while summarizing: {e}"

    return render_template(
        "summarize.html",
        summary=summary,
        original=original_text,
        length_option=length_option,
        language=language,
        word_count=word_count,
        reading_time=reading_time
    )

# 🧾 PDF SUMMARIZER
@app.route("/pdf-summarizer", methods=["GET", "POST"])
def pdf_summarizer():
    """Main route for PDF summarization."""
    if request.method == "POST":
        uploaded_pdf = request.files.get("file")
        if not (uploaded_pdf and uploaded_pdf.filename.lower().endswith(".pdf")):
            return render_template("pdf.html", message="⚠️ Please upload a valid PDF file.", download_link=None)

        try:
            uploaded_pdf.seek(0)
            extracted = extract_text_pypdf2(uploaded_pdf)

            if not extracted.strip():
                return render_template("pdf.html", message="⚠️ No readable text found in this PDF.", download_link=None)

            summary = batch_summarize(extracted) if len(extracted) > 8000 else summarize_text(extracted, 10)
            summary = clean_text(summary)

            filename = os.path.splitext(uploaded_pdf.filename)[0]
            summary_name = f"{filename}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            path = os.path.join("static", summary_name)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Summary of {filename}", ln=True, align='C')
            pdf.ln(10)

            pdf.set_font("Arial", size=12)
            safe_summary = summary.encode("latin-1", "replace").decode("latin-1")
            for line in safe_summary.split("\n"):
                if not line.strip():
                    continue
                pdf.multi_cell(0, 8, line.strip(), align='J')
                pdf.ln(2)

            pdf.output(path)
            return render_template("pdf.html", download_link=summary_name, message="✅ Summary ready for download!")

        except Exception as e:
            return render_template("pdf.html", message=f"⚠️ Error: {e}", download_link=None)

    return render_template("pdf.html", message=None, download_link=None)

# 🧮 EXCEL ANALYZER
@app.route("/excel-analyzer", methods=["GET", "POST"])
def excel_analyzer():
    """Analyze Excel or CSV file and generate PDF summary."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("excel.html", message="⚠️ Please upload an Excel or CSV file.", download_link=None)

        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                return render_template("excel.html", message="⚠️ Unsupported file format.", download_link=None)

            filename = os.path.splitext(file.filename)[0]
            total_rows, total_cols = df.shape
            column_names = list(df.columns)
            missing_values = df.isnull().sum().to_dict()
            data_types = df.dtypes.astype(str).to_dict()

            numeric_cols = len(df.select_dtypes(include="number").columns)
            categorical_cols = len(df.select_dtypes(exclude="number").columns)
            breakdown = f"Numeric Columns: {numeric_cols}, Categorical Columns: {categorical_cols}"

            preview = df.head(5).to_string(index=False)

            summary_name = f"{filename}_ExcelSummary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            path = os.path.join("static", summary_name)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Excel File Summary: {filename}", ln=True, align='C')
            pdf.ln(10)

            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, f"File Name: {file.filename}")
            pdf.multi_cell(0, 8, f"Total Rows: {total_rows}")
            pdf.multi_cell(0, 8, f"Total Columns: {total_cols}")
            pdf.multi_cell(0, 8, f"Column Names: {', '.join(column_names)}")
            pdf.multi_cell(0, 8, f"Data Type Breakdown: {breakdown}")
            pdf.ln(8)

            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 8, "Missing Values:", ln=True)
            pdf.set_font("Courier", size=11)
            for col, val in missing_values.items():
                safe_text = f"{col}: {val}".encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 6, safe_text)
            pdf.ln(8)

            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 8, "Column Data Types:", ln=True)
            pdf.set_font("Courier", size=11)
            for col, dtype in data_types.items():
                safe_text = f"{col}: {dtype}".encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 6, safe_text)
            pdf.ln(8)

            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 8, "First 5 Rows Preview:", ln=True)
            pdf.set_font("Courier", size=10)
            safe_preview = preview.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe_preview)
            pdf.ln(5)

            pdf.output(path)
            return render_template("excel.html", message="✅ Excel summary generated successfully!", download_link=summary_name)

        except Exception as e:
            return render_template("excel.html", message=f"⚠️ Error: {e}", download_link=None)

    return render_template("excel.html", message=None, download_link=None)

# 📝 GRAMMAR CHECK
@app.route("/grammar-check", methods=["GET", "POST"])
def grammar_check():
    corrected_text = None
    original_text = None
    if request.method == "POST":
        text_input = request.form.get("text", "").strip()
        if text_input:
            try:
                corrected_text = tool.correct(text_input)
                original_text = text_input
            except Exception as e:
                corrected_text = f"⚠️ Grammar checker failed: {e}"
        else:
            corrected_text = "⚠️ Please enter some text to check grammar."
    return render_template("grammar.html", corrected=corrected_text, original=original_text)

# 🧾 DOWNLOAD HANDLER
@app.route("/download/<filename>")
def download_file(filename):
    filepath = os.path.join("static", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename, mimetype="application/pdf")
    else:
        return "⚠️ File not found", 404

# ❌ 404 PAGE
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))