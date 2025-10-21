import os
from PyPDF2 import PdfReader, PdfWriter

def trim_pdf(original_filepath, page_range_str):
    """
    Creates a temporary, trimmed version of a PDF based on a page range.

    Args:
        original_filepath (str): The full path to the original PDF file.
        page_range_str (str): A string representing the page range, e.g., "15-28".

    Returns:
        str: The filepath of the newly created temporary trimmed PDF.
             Returns None if there's an error.
    """
    try:
        # Parse the page range string
        start_page, end_page = map(int, page_range_str.split('-'))

        # Adjust for 0-based indexing
        start_page -= 1
        end_page -= 1

        reader = PdfReader(original_filepath)
        writer = PdfWriter()

        # Add the specified pages to the new PDF
        for i in range(start_page, end_page + 1):
            writer.add_page(reader.pages[i])

        # Create a temporary filename for the trimmed PDF
        temp_dir = 'temp_uploads'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        original_basename = os.path.basename(original_filepath)
        temp_filename = f"trimmed_{start_page + 1}-{end_page + 1}_{original_basename}"
        temp_filepath = os.path.join(temp_dir, temp_filename)

        # Write the new PDF to the temporary file
        with open(temp_filepath, 'wb') as f:
            writer.write(f)

        return temp_filepath

    except Exception as e:
        print(f"Error trimming PDF: {e}")
        return None

def cleanup_temp_file(filepath):
    """Deletes a temporary file if it exists."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up temp file {filepath}: {e}")