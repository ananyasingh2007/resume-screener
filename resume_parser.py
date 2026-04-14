import fitz
import io

def extract_text(file):
    try:
        if file.name.endswith('.pdf'):
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()

        elif file.name.endswith('.txt'):
            return file.read().decode('utf-8', errors='ignore').strip()

        else:
            return None

    except Exception as e:
        return None