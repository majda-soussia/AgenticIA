import PyPDF2
while True:
    user_input = input("> ")
    if user_input.startswith("pdf"):
        # Handle CSV file reading
        path = user_input[4:].strip()
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            print(f"\nPDF ({len(reader.pages)} pages):\n{text[:1000]}...")
        except Exception as e:
            print(f"\nErreur: {e}")
        continue