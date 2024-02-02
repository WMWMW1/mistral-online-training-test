import fitz  # PyMuPDF
import re

def extract_text_by_page_layout(pdf_path):
    # Open the provided PDF file
    document = fitz.open(pdf_path)
    extracted_text = []

    for page_number in range(len(document)):
        page = document[page_number]

        # Analyze the layout to determine the top, left, and right regions
        # This is a placeholder: you may need to adjust the coordinates
        # depending on the actual layout of your PDF
        top_rect = fitz.Rect(0, 0, page.rect.width, 100)  # Adjust height as needed
        left_rect = fitz.Rect(0, 100, page.rect.width / 2, page.rect.height)  # Adjust coordinates as needed
        right_rect = fitz.Rect(page.rect.width / 2, 100, page.rect.width, page.rect.height)  # Adjust coordinates as needed

        # Extract text from the defined regions
        top_text = page.get_text("text", clip=top_rect)
        left_text = page.get_text("text", clip=left_rect)
        right_text = page.get_text("text", clip=right_rect)

        # Combine the text from top, left, and right regions
        page_text = top_text + "\n" + left_text + "\n" + right_text
        extracted_text.append(page_text)

    # Close the document
    document.close()
    return extracted_text

def save_text_to_file(text_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for page_text in text_list:
            file.write(page_text)
            file.write("\n\n--- End of Page ---\n\n")

# Path to your PDF file
pdf_path = 'pdf111.pdf' 
# Path to the output text file
output_txt_path = 'file.txt'  #

# Extract text from the PDF
text_by_page = extract_text_by_page_layout(pdf_path)
# Save the extracted text to a file
save_text_to_file(text_by_page, output_txt_path)
