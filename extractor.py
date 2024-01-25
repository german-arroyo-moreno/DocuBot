import PyPDF2

def text_from_pdf(pdf_path):
    meta = []
    text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text.append(page.extract_text())
            meta.append({'page': page_num, 'source': pdf_path})
    return {'text': text, 'meta': meta}
