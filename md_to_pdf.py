from fpdf import FPDF
from fpdf.enums import XPos, YPos

class TechnicalPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Pakistani News RAG: Architecture and Logic', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_pdf(md_path, pdf_path):
    pdf = TechnicalPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    # Calculate usable width
    effective_page_width = pdf.w - 2 * pdf.l_margin
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace math/special symbols to avoid breaking basic font
    # Use simple replacements without backslashes to avoid syntax warnings
    clean_content = content.replace('$\\sqrt{\\sum (p_i - q_i)^2}$', 'Square root of (sum of squared differences)')
    clean_content = clean_content.replace('PK', 'PK') # Placeholder for emoji replacement logic
    
    lines = clean_content.split('\n')
        
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Basic character cleanup for PDF compatibility
        line = line.encode('latin-1', 'replace').decode('latin-1')
            
        if line.startswith('# '):
            pdf.set_font("helvetica", 'B', 16)
            pdf.multi_cell(effective_page_width, 10, line[2:].upper())
            pdf.set_font("helvetica", size=12)
        elif line.startswith('## '):
            pdf.set_font("helvetica", 'B', 14)
            pdf.multi_cell(effective_page_width, 10, line[3:])
            pdf.set_font("helvetica", size=12)
        elif line.startswith('### '):
            pdf.set_font("helvetica", 'B', 12)
            pdf.multi_cell(effective_page_width, 10, line[4:])
            pdf.set_font("helvetica", size=12)
        elif line.startswith('- '):
            pdf.multi_cell(effective_page_width, 8, f"* {line[2:]}")
        else:
            pdf.multi_cell(effective_page_width, 8, line)
            
    pdf.output(pdf_path)
    print(f"PDF generated successfully at: {pdf_path}")

if __name__ == "__main__":
    create_pdf('TECHNICAL_GUIDE.md', 'Project_Documentation.pdf')
