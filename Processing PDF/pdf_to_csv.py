import camelot
import os

# Path to the PDF file
pdf_path = "C:\Aditya\Python\executing PDF chat\Large-PDF-Chat\docs\CGUHEmYUVq.pdf"

# Read the PDF file
tables = camelot.read_pdf(pdf_path, pages='all')

# Directory to save the CSV files
output_dir = "output_tables"
os.makedirs(output_dir, exist_ok=True)

# Save each table to a separate CSV file
for i, table in enumerate(tables):
    csv_path = os.path.join(output_dir, f"table_page_{i + 1}.csv")
    table.to_csv(csv_path)

print(f"Extracted {len(tables)} tables from the PDF and saved them in '{output_dir}' directory.")
