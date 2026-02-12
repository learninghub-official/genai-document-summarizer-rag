import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    text = extract_text_from_pdf("/Users/mayank/Documents/AI_Projects/2531874/GenAI_Compliance_Summarizer/sample_outputs/compliance_policy.pdf")
    print(text[:500])
