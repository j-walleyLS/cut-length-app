# Streamlit Cut Length Optimiser with OCR Support

import math
import copy
import streamlit as st
import re
from io import BytesIO
import requests
import base64

# OCR and PDF handling imports
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import requests
    import base64
    OCR_AVAILABLE = True
    # Check if using cloud OCR or local
    CLOUD_OCR = True  # Set to True to use OCR.space API
    OCR_API_KEY = "K87215639688957"  # Free tier API key
except ImportError:
    OCR_AVAILABLE = False
    st.warning("Libraries not installed. Run: pip install PyMuPDF pillow requests")

BLADE_WIDTH = 4
MAX_BRANCH_COUNT = 10000

# -----------------------------
# Utilities
# -----------------------------
def parse_dimensions(dim_str):
    parts = dim_str.lower().replace(" ", "").split("x")
    return int(parts[0]), int(parts[1])

def parse_boq_text(text):
    """Parse BOQ text and extract units with quantities and dimensions"""
    units = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that are clearly not BOQ data
        skip_patterns = [
            r'(?i)from:', r'(?i)tel:', r'(?i)ref:', r'(?i)needed',
            r'(?i)asap', r'(?i)cheer', r'(?i)london', r'(?i)stone',
            r'(?i)britannia', r'(?i)dear', r'(?i)please', r'(?i)thank'
        ]
        
        skip_line = False
        for pattern in skip_patterns:
            if re.search(pattern, line):
                skip_line = True
                break
        
        if skip_line:
            continue
            
        # Various regex patterns to match different formats
        patterns = [
            r'x(\d+)\s+(\d+)\s*[×x]\s*(\d+)',  # x1 1650×560 (with space)
            r'x(\d+)\s*(\d+)\s*[×x]\s*(\d+)',  # x1 1650×560
            r'(\d+)\s*x\s*(\d+)\s*[×x]\s*(\d+)',  # 1x 1650×560
            r'(\d+)\s*no\.?\s*(\d+)\s*[×x]\s*(\d+)',  # 1 no. 1650×560
            r'(\d+)\s*×\s*(\d+)\s*[×x]\s*(\d+)\s*\(qty[:\s]*(\d+)\)',  # 1650×560 (qty: 1)
            r'(\d+)\s*[×x]\s*(\d+)\s*[×x]\s*(\d+)',  # 1×1650×560
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:
                    if 'x' in line.lower() and line.lower().index('x') < 3:
                        # Format: x1 1650×560
                        qty, width, height = groups
                    else:
                        # Format: 1650×560×1 or similar
                        width, height, qty = groups
                elif len(groups) == 4:
                    # Format with qty in parentheses
                    width, height, _, qty = groups
                else:
                    continue
                
                try:
                    # Validate dimensions are reasonable (between 10mm and 10000mm)
                    w = int(width)
                    h = int(height)
                    q = int(qty)
                    
                    if 10 <= w <= 10000 and 10 <= h <= 10000 and q > 0:
                        units.append({
                            'width': w,
                            'height': h,
                            'quantity': q
                        })
                        break
                except ValueError:
                    continue
    
    return units

def extract_text_from_pdf_ocr(pdf_bytes, progress_callback=None):
    """Extract text from PDF using OCR"""
    if not OCR_AVAILABLE:
        return None
    
    try:
        if CLOUD_OCR:
            # Use OCR.space API (cloud-based, no local installation needed)
            return extract_text_with_cloud_ocr(pdf_bytes, progress_callback)
        else:
            # Use local Tesseract
            import pytesseract
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            all_text = []
            total_pages = len(pdf_document)
            
            for page_num in range(total_pages):
                if progress_callback:
                    progress_callback(page_num + 1, total_pages)
                
                # Get the page
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.pil_tobytes(format="PNG")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_data))
                
                # Perform OCR on the image
                text = pytesseract.image_to_string(image)
                all_text.append(text)
            
            pdf_document.close()
            return '\n'.join(all_text)
    
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

def extract_text_with_cloud_ocr(pdf_bytes, progress_callback=None):
    """Extract text using OCR.space cloud API"""
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = []
        total_pages = len(pdf_document)
        
        # OCR.space API endpoint
        api_url = "https://api.ocr.space/parse/image"
        
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages)
            
            # Get the page
            page = pdf_document[page_num]
            
            # Convert page to image - start with lower DPI to reduce file size
            mat = fitz.Matrix(200/72, 200/72)  # 200 DPI to stay under 1MB limit
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.pil_tobytes(format="PNG")
            
            # Convert to PIL Image for compression
            image = Image.open(BytesIO(img_data))
            
            # Compress image if needed
            output = BytesIO()
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            
            # Save as JPEG with compression
            image.save(output, format='JPEG', quality=85, optimize=True)
            compressed_data = output.getvalue()
            
            # Check size and compress more if needed
            if len(compressed_data) > 900000:  # 900KB to be safe
                output = BytesIO()
                image.save(output, format='JPEG', quality=70, optimize=True)
                compressed_data = output.getvalue()
            
            # Convert to base64
            img_base64 = base64.b64encode(compressed_data).decode()
            
            # Make API request with settings optimized for handwriting
            payload = {
                'apikey': OCR_API_KEY,
                'base64Image': f'data:image/jpeg;base64,{img_base64}',
                'language': 'eng',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2,  # Engine 2 is better for handwriting
                'isTable': True,  # Help with structured data
                'filetype': 'JPG'
            }
            
            response = requests.post(api_url, data=payload, timeout=30)
            result = response.json()
            
            if result.get('IsErroredOnProcessing') == False:
                text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
                if text.strip():  # Only add non-empty text
                    all_text.append(text)
                else:
                    st.warning(f"No text found on page {page_num + 1}")
            else:
                error_msg = result.get('ErrorMessage', [])
                if isinstance(error_msg, list):
                    error_msg = ', '.join(error_msg)
                st.warning(f"OCR failed for page {page_num + 1}: {error_msg}")
        
        pdf_document.close()
        
        combined_text = '\n'.join(all_text)
        
        # If no text was extracted, provide helpful message
        if not combined_text.strip():
            st.warning("OCR couldn't extract text from this PDF. This might be due to:")
            st.info("""
            • Handwritten text (OCR works better with printed text)
            • Image quality issues
            • API limitations
            
            **Please use the manual input below to enter your data.**
            """)
            return None
            
        return combined_text
        
    except requests.exceptions.Timeout:
        st.error("OCR request timed out. Please try manual input.")
        return None
    except Exception as e:
        st.error(f"Cloud OCR Error: {str(e)}")
        return None

def extract_text_from_image_ocr(image_bytes):
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return None
    
    try:
        if CLOUD_OCR:
            # Use cloud OCR
            img_base64 = base64.b64encode(image_bytes).decode()
            
            payload = {
                'apikey': OCR_API_KEY,
                'base64Image': f'data:image/png;base64,{img_base64}',
                'language': 'eng',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2
            }
            
            response = requests.post("https://api.ocr.space/parse/image", data=payload)
            result = response.json()
            
            if result.get('IsErroredOnProcessing') == False:
                return result.get('ParsedResults', [{}])[0].get('ParsedText', '')
            else:
                st.error("Cloud OCR processing failed")
                return None
        else:
            # Use local Tesseract
            import pytesseract
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
    
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

def parse_uploaded_file(uploaded_file):
    """Parse uploaded file and extract BOQ data"""
    try:
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
            return parse_boq_text(content)
        
        elif uploaded_file.type == "text/csv":
            content = str(uploaded_file.read(), "utf-8")
            return parse_boq_text(content)
        
        elif uploaded_file.type == "application/pdf":
            if OCR_AVAILABLE:
                # Use OCR to extract text from PDF
                pdf_bytes = uploaded_file.read()
                
                with st.spinner("📄 Processing PDF with OCR..."):
                    progress_bar = st.progress(0)
                    
                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                    
                    extracted_text = extract_text_from_pdf_ocr(pdf_bytes, update_progress)
                    progress_bar.empty()
                
                if extracted_text:
                    st.success("✅ Text extracted from PDF!")
                    
                    # Show extracted text in an expander for review
                    with st.expander("📝 Review extracted text", expanded=False):
                        st.text_area("Extracted content:", extracted_text, height=200)
                    
                    # Parse the extracted text
                    units = parse_boq_text(extracted_text)
                    
                    if units:
                        st.success(f"✅ Found {len(units)} unit types in the PDF - automatically added to list!")
                        
                        # Automatically add units to the session state
                        for unit in units:
                            st.session_state.units.append({
                                "width": unit["width"],
                                "height": unit["height"],
                                "quantity": unit["quantity"],
                                "forced_slabs": [],
                                "order": len(st.session_state.units)
                            })
                        
                        # Show what was added
                        st.info("**Added to unit list:**")
                        for unit in units:
                            st.write(f"• {unit['quantity']}x {unit['width']}×{unit['height']}")
                        
                        return units
                    else:
                        st.warning("No valid units found in the extracted text.")
                        st.info("**Based on your PDF, try entering your data like this:**")
                        st.code("""x1 1650×560
x1 1650×150
x1 2000×850
x5 2000×350
x6 2000×150""")
                        
                        # Allow manual editing
                        st.markdown("**Or edit the extracted text below:**")
                        edited_text = st.text_area(
                            "Edit or paste your BOQ data:",
                            value=extracted_text if extracted_text else "",
                            height=200,
                            placeholder="Enter data in format: x1 1650×560",
                            help="Use formats like: 'x1 1650×560' or '1x 1650×560' or '1 no. 1650×560'"
                        )
                        if st.button("Parse edited text"):
                            units = parse_boq_text(edited_text)
                            if units:
                                # Add parsed units to session state
                                for unit in units:
                                    st.session_state.units.append({
                                        "width": unit["width"],
                                        "height": unit["height"],
                                        "quantity": unit["quantity"],
                                        "forced_slabs": [],
                                        "order": len(st.session_state.units)
                                    })
                                st.success(f"✅ Added {len(units)} unit types to list!")
                                return units
                            else:
                                st.error("Still couldn't parse the data. Please check the format.")
                    
                    return units
                else:
                    st.error("Failed to extract text from PDF")
                    return []
            else:
                # Fallback to manual instructions if OCR not available
                st.info("📄 PDF detected! OCR not available.")
                st.markdown("""
                ### To enable automatic PDF text extraction:
                1. **Install required libraries:**
                   ```bash
                   pip install PyMuPDF pytesseract pillow
                   ```
                2. **Install Tesseract OCR:**
                   - **Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
                   - **Mac:** `brew install tesseract`
                   - **Linux:** `sudo apt-get install tesseract-ocr`
                
                ### Manual alternative:
                1. **Open your PDF** in a PDF viewer
                2. **Select all text** (Ctrl+A / Cmd+A) 
                3. **Copy** (Ctrl+C / Cmd+C)
                4. **Paste into the text box below** ⬇️
                """)
                return []
        
        elif uploaded_file.type.startswith("image/"):
            if OCR_AVAILABLE:
                # Use OCR to extract text from image
                image_bytes = uploaded_file.read()
                
                with st.spinner("🖼️ Processing image with OCR..."):
                    extracted_text = extract_text_from_image_ocr(image_bytes)
                
                if extracted_text:
                    st.success("✅ Text extracted from image!")
                    
                    # Show extracted text for review
                    with st.expander("📝 Review extracted text", expanded=False):
                        st.text_area("Extracted content:", extracted_text, height=200)
                    
                    # Parse the extracted text
                    units = parse_boq_text(extracted_text)
                    
                    if units:
                        st.info(f"Found {len(units)} unit types in the image")
                    else:
                        st.warning("No valid units found. You can edit the text below.")
                        edited_text = st.text_area(
                            "Edit extracted text:",
                            extracted_text,
                            height=300
                        )
                        if st.button("Parse edited text"):
                            units = parse_boq_text(edited_text)
                            if units:
                                return units
                    
                    return units
                else:
                    st.error("Failed to extract text from image")
                    return []
            else:
                # Fallback instructions
                st.info("🖼️ Image detected! OCR not available.")
                st.markdown("""
                To enable automatic text extraction, install OCR libraries (see PDF instructions above).
                
                **Manual alternative:** Use Google Lens on your phone to extract text.
                """)
                return []
        
        else:
            st.error("Unsupported file type. Please use .txt, .csv, .pdf, or image files.")
            return []
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return []

# -----------------------------
# DFS Packing Core Logic
# -----------------------------
class Placement:
    def __init__(self, unit_order, x, y, used_width, used_height):
        self.unit_order = unit_order
        self.x = x
        self.y = y
        self.used_width = used_width
        self.used_height = used_height

def calculate_layout_cost(layout, slab_width, slab_height):
    total_cut = 0
    for p in layout:
        right_edge = p.x + p.used_width
        top_edge = p.y + p.used_height
        right_gap = slab_width - right_edge
        top_gap = slab_height - top_edge
        if right_gap > 0:
            total_cut += p.used_height
        if top_gap > 0:
            total_cut += p.used_width
    if layout:
        min_x = min(p.x for p in layout)
        min_y = min(p.y for p in layout)
        if min_x > 0:
            total_cut += 10000
        if min_y > 0:
            total_cut += 10000
    return total_cut

def split_free_rectangle(free_rect, piece, placement):
    fx, fy, FW, FH = free_rect
    orig_top = fy + FH
    px, py = placement
    pw, ph = piece
    new_rects = []
    right_width = (fx + FW) - (px + pw + BLADE_WIDTH)
    if right_width > 0:
        new_rects.append((px + pw + BLADE_WIDTH, py, right_width, orig_top - py))
    top_height = orig_top - (py + ph + BLADE_WIDTH)
    if top_height > 0:
        new_rects.append((px, py + ph + BLADE_WIDTH, pw, top_height))
    return new_rects

def remove_overlapping_free_rectangles(free_rects):
    cleaned = []
    for rect in free_rects:
        x, y, w, h = rect
        contained = False
        for other in free_rects:
            if other == rect:
                continue
            ox, oy, ow, oh = other
            if x >= ox and y >= oy and (x + w) <= (ox + ow) and (y + h) <= (oy + oh):
                contained = True
                break
        if not contained:
            cleaned.append(rect)
    return cleaned

def repack_dfs(free_rects, pieces, current_layout, best_solution, slab_width, slab_height, branch_count):
    branch_count[0] += 1
    if branch_count[0] > MAX_BRANCH_COUNT:
        return
    placed_count = len(current_layout)
    current_cost = calculate_layout_cost(current_layout, slab_width, slab_height)
    if placed_count > best_solution["count"] or (placed_count == best_solution["count"] and current_cost < best_solution["cost"]):
        best_solution["count"] = placed_count
        best_solution["cost"] = current_cost
        best_solution["layout"] = copy.deepcopy(current_layout)
    if not pieces:
        return
    sorted_free_rects = sorted(free_rects, key=lambda rect: (rect[1], rect[0]))
    for i, free_rect in enumerate(sorted_free_rects):
        fx, fy, FW, FH = free_rect
        piece_data = pieces[0]
        w, h, unit_order = piece_data
        orients = [(w, h), (h, w)]
        for pw, ph in orients:
            if pw <= FW and ph <= FH:
                new_placement = Placement(unit_order, fx, fy, pw, ph)
                new_rects = split_free_rectangle(free_rect, (pw, ph), (fx, fy))
                new_free_rects = sorted_free_rects[:i] + sorted_free_rects[i+1:] + new_rects
                new_free_rects = remove_overlapping_free_rectangles(new_free_rects)
                current_layout.append(new_placement)
                repack_dfs(new_free_rects, pieces[1:], current_layout, best_solution, slab_width, slab_height, branch_count)
                current_layout.pop()

def pack_one_slab(slab_width, slab_height, units):
    """Pack units into a single slab, returning the best layout found"""
    pieces = []
    for u in units:
        for _ in range(u["quantity"]):
            pieces.append((u["width"], u["height"], u["order"]))
    
    best_solution = {"layout": None, "cost": float("inf"), "count": 0}
    branch_count = [0]
    repack_dfs([(0, 0, slab_width, slab_height)], pieces, [], best_solution, slab_width, slab_height, branch_count)
    return best_solution

# -----------------------------
# Global Optimization Functions
# -----------------------------
def can_unit_fit_in_slab(unit, slab_width, slab_height):
    """Check if a unit can fit in a slab (considering rotation)"""
    w, h = unit["width"], unit["height"]
    return (w <= slab_width and h <= slab_height) or (h <= slab_width and w <= slab_height)

def get_viable_slabs_for_unit(unit, slab_sizes):
    """Get all slab sizes that can accommodate this unit"""
    viable_slabs = []
    for slab in slab_sizes:
        if can_unit_fit_in_slab(unit, slab[0], slab[1]):
            viable_slabs.append(slab)
    return viable_slabs

def calculate_packing_efficiency(unit_types, slab, max_iterations=5):
    """Calculate how efficiently unit types pack into a given slab size"""
    slab_width, slab_height = slab
    slab_area = slab_width * slab_height
    
    # Create test units for efficiency calculation
    test_units = []
    total_unit_area = 0
    
    for unit_type in unit_types:
        # Use actual quantity or max_iterations, whichever is smaller
        test_quantity = min(unit_type["quantity"], max_iterations)
        if test_quantity > 0:
            test_units.append({
                "width": unit_type["width"],
                "height": unit_type["height"], 
                "quantity": test_quantity,
                "order": unit_type["order"]
            })
            total_unit_area += unit_type["width"] * unit_type["height"] * test_quantity
    
    if not test_units or total_unit_area == 0:
        return 0.0
    
    # Try packing these units
    result = pack_one_slab(slab_width, slab_height, test_units)
    
    if not result["layout"] or result["count"] == 0:
        return 0.0
    
    # Calculate actual area used by packed pieces
    used_area = 0
    for placement in result["layout"]:
        used_area += placement.used_width * placement.used_height
    
    # Efficiency is the ratio of unit area to slab area
    # This gives us the true material utilization
    efficiency = total_unit_area / slab_area
    
    # Bonus for fitting more pieces (packing density)
    packing_success_rate = result["count"] / sum(u["quantity"] for u in test_units)
    
    return efficiency * packing_success_rate

def optimize_unit_allocation(units, slab_sizes):
    """
    Globally optimize allocation of units across all slab sizes.
    Each unit type is allocated to exactly ONE optimal slab size.
    Returns a dictionary mapping slab sizes to lists of units to cut from that slab.
    """
    # Initialize allocation dictionary - only include available slab sizes
    allocation = {}
    
    # Process each unit type individually to find its optimal slab
    for unit in units:
        unit_allocated = False
        
        # Handle forced constraints first - these override optimization
        if unit.get("forced_slabs") and len(unit["forced_slabs"]) > 0:
            forced_slab = tuple(unit["forced_slabs"][0])
            if forced_slab in slab_sizes:  # Make sure forced slab is available
                if forced_slab not in allocation:
                    allocation[forced_slab] = []
                allocation[forced_slab].append(copy.deepcopy(unit))
                unit_allocated = True
            continue
        
        # For free units, find the single best slab
        viable_slabs = []
        for slab in slab_sizes:
            if can_unit_fit_in_slab(unit, slab[0], slab[1]):
                viable_slabs.append(slab)
        
        if not viable_slabs:
            continue  # Skip units that don't fit anywhere
        
        # Test each viable slab and find the most efficient one
        best_slab = None
        best_total_slabs_needed = float('inf')
        best_efficiency = -1
        
        for slab in viable_slabs:
            # Calculate packing efficiency for this unit in this slab
            efficiency = calculate_packing_efficiency([unit], slab)
            
            if efficiency <= 0:
                continue  # Skip slabs where nothing fits
            
            # Calculate how many slabs would actually be needed
            slab_area = slab[0] * slab[1]
            unit_area = unit["width"] * unit["height"] * unit["quantity"]
            
            # Estimate slabs needed based on efficiency
            # Higher efficiency = fewer slabs needed
            estimated_slabs_needed = math.ceil(unit_area / (slab_area * efficiency))
            
            # Prefer solution that needs fewer total slabs
            # Break ties with higher efficiency
            is_better = (estimated_slabs_needed < best_total_slabs_needed) or (
                estimated_slabs_needed == best_total_slabs_needed and efficiency > best_efficiency
            )
            
            if is_better:
                best_total_slabs_needed = estimated_slabs_needed
                best_efficiency = efficiency
                best_slab = slab
        
        # Allocate this unit type to its single best slab
        if best_slab:
            if best_slab not in allocation:
                allocation[best_slab] = []
            allocation[best_slab].append(copy.deepcopy(unit))
            unit_allocated = True
    
    return allocation

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(
    page_title="Cut Length Optimiser",
    page_icon="🪚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS with proper spacing control
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
    }
    .stSidebar > div:first-child {
        padding-top: 0rem;
    }
    .stSidebar .element-container:first-child {
        margin-top: -2rem;
    }
    
    /* Headers */
    .stSidebar .stMarkdown h3 {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        margin-bottom: 0rem !important;
    }
    
    /* SLAB BUTTONS - Give them small spacing */
    button[data-testid*="slab_"] {
        margin-bottom: 0.1rem !important;
    }
    
    /* Remove gaps between slab sections */
    .stSidebar .stMarkdown:has(strong) {
        margin-bottom: -1rem !important;
    }
    
    /* Tighten spacing around slab section headings */
    .stSidebar .stMarkdown strong {
        margin-bottom: 0rem !important;
    }
    
    /* Remove default spacing from other elements */
    .stSidebar .stNumberInput,
    .stSidebar .stSelectbox,
    .stSidebar .stTextInput {
        margin-bottom: 0rem !important;
    }
    
    /* Remove spacing around horizontal dividers */
    .stSidebar hr {
        margin: 0rem !important;
        padding: 0rem !important;
    }
    
    /* EXPANDER HEADERS - Try most common selectors for larger font */
    [data-testid="stExpander"] summary {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    .streamlit-expanderHeader {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Button colors */
    .stButton > button[kind="primary"] {
        background-color: #28a745;
        border-color: #28a745;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "units" not in st.session_state:
    st.session_state.units = []
if "custom_slabs" not in st.session_state:
    st.session_state.custom_slabs = []
if "selected_slabs" not in st.session_state:
    st.session_state.selected_slabs = []
if "unit_input_rows" not in st.session_state:
    st.session_state.unit_input_rows = [{"width": "", "height": "", "quantity": 1, "forced": "Any"}]
if "scant_slabs" not in st.session_state:
    st.session_state.scant_slabs = set()  # Track which slabs are scants

# -----------------------------
# Main UI
# -----------------------------
# Create header with logo
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🪚 Cut Length Optimiser")
    st.markdown("*Optimize material cutting from various slab sizes*")
    if OCR_AVAILABLE:
        st.markdown("🔍 **OCR Enabled** - Can extract text from PDFs and images")
with col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 1rem;">
        <img src="https://media.licdn.com/dms/image/v2/D4E0BAQHP523W42qnWw/company-logo_200_200/company-logo_200_200/0/1684478471407/london_stone_logo?e=2147483647&v=beta&t=dW-GcSLavPP7SD-qAqCyQgaH-LODRJqcoHyblQRHR9Q" 
             style="height: 3rem; background: transparent;" 
             alt="London Stone Logo"/>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar - Slab Selection
# -----------------------------
st.sidebar.markdown("### 📐 Available Slab Sizes")

# Define slab sizes with categories - (display_size, actual_size, label)
paving_slabs = [
    ((600, 600), (600, 600), "600×600"),
    ((900, 600), (900, 600), "900×600"), 
    ((1800, 700), (1800, 700), "1800×700"),
    ((1800, 900), (1800, 900), "1800×900")
]

treads_slabs = [
    ((900, 500), (900, 500), "900×500"),
    ((1500, 500), (1500, 500), "1500×500"),
    ((2000, 500), (2000, 500), "2000×500")
]

italian_porcelain_slabs = [
    ((600, 600), (596, 596), "600×600\n(596×596)"),
    ((900, 450), (897, 446), "900×450\n(897×446)"),
    ((900, 600), (897, 596), "900×600\n(897×596)"),
    ((800, 800), (794, 794), "800×800\n(794×794)"),
    ((1200, 600), (1194, 596), "1200×600\n(1194×596)"),
    ((1200, 1200), (1194, 1194), "1200×1200\n(1194×1194)")
]

scants_slabs = [
    ((1800, 700), (1800, 700), "Britannia\n(1800×700)"),
    ((1800, 700), (1800, 700), "Portland\n(1800×700)"),
    ((2000, 1000), (2000, 1000), "Jura\n(2000×1000)")
]

# Create slab buttons with categories and two columns
def create_slab_buttons(slab_list, category_name):
    st.sidebar.markdown(f"**{category_name}**")
    st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)  # Space below header
    
    # Create buttons in pairs for 2-column layout
    for i in range(0, len(slab_list), 2):
        col1, col2 = st.sidebar.columns(2)
        
        # First button in pair
        if i < len(slab_list):
            display_size, actual_size, label = slab_list[i]
            is_selected = actual_size in st.session_state.selected_slabs
            
            with col1:
                if st.button(
                    f"{'✓ ' if is_selected else ''}{label}",
                    key=f"slab_{actual_size}_{category_name}_{i}",  # Make key unique per category
                    help=f"Click to {'remove' if is_selected else 'add'} {label}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    if actual_size in st.session_state.selected_slabs:
                        st.session_state.selected_slabs.remove(actual_size)
                        # Remove from scants if it's a scant
                        if category_name == "Scants":
                            st.session_state.scant_slabs.discard(actual_size)
                    else:
                        st.session_state.selected_slabs.append(actual_size)
                        # Add to scants if it's a scant
                        if category_name == "Scants":
                            st.session_state.scant_slabs.add(actual_size)
                    st.rerun()
        
        # Second button in pair (if exists)
        if i + 1 < len(slab_list):
            display_size, actual_size, label = slab_list[i + 1]
            is_selected = actual_size in st.session_state.selected_slabs
            
            with col2:
                if st.button(
                    f"{'✓ ' if is_selected else ''}{label}",
                    key=f"slab_{actual_size}_{category_name}_{i+1}",  # Make key unique per category
                    help=f"Click to {'remove' if is_selected else 'add'} {label}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    if actual_size in st.session_state.selected_slabs:
                        st.session_state.selected_slabs.remove(actual_size)
                        # Remove from scants if it's a scant
                        if category_name == "Scants":
                            st.session_state.scant_slabs.discard(actual_size)
                    else:
                        st.session_state.selected_slabs.append(actual_size)
                        # Add to scants if it's a scant
                        if category_name == "Scants":
                            st.session_state.scant_slabs.add(actual_size)
                    st.rerun()
        
        # Add spacing between button rows, but not after the last row
        if i + 2 < len(slab_list):  # Only add spacing if not the last row
            st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Create all slab button sections
create_slab_buttons(paving_slabs, "Paving/Slabs")
create_slab_buttons(treads_slabs, "Treads")
create_slab_buttons(italian_porcelain_slabs, "Italian Porcelain")
create_slab_buttons(scants_slabs, "Scants")

st.sidebar.markdown("---")

# Custom slab input
st.sidebar.subheader("Custom Slab Sizes")

if "custom_input" not in st.session_state:
    st.session_state.custom_input = ""

custom_input = st.sidebar.text_input(
    "Enter one size at a time",
    value=st.session_state.custom_input,
    placeholder="e.g. 800x400",
    help="Enter width x height and press Enter",
    key="custom_slab_input"
)

# Process custom slab input
if custom_input != st.session_state.custom_input:
    st.session_state.custom_input = custom_input
    
    if custom_input.strip() and 'x' in custom_input.lower():
        try:
            new_slab = parse_dimensions(custom_input)
            if new_slab not in st.session_state.custom_slabs:
                st.session_state.custom_slabs.append(new_slab)
                st.session_state.custom_input = ""
                st.rerun()
        except Exception as e:
            st.sidebar.error("❌ Invalid format. Use: 800x400")

# Display custom slabs
if st.session_state.custom_slabs:
    st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Custom Slabs:**")
    st.sidebar.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
    
    custom_slabs_to_remove = []
    for i, slab in enumerate(st.session_state.custom_slabs):
        col1, col2 = st.sidebar.columns([0.3, 2.7])
        if col1.button("×", key=f"remove_custom_{i}_{slab[0]}_{slab[1]}", help="Remove this custom slab"):
            custom_slabs_to_remove.append(i)
        col2.markdown(f"<div style='text-align: left; padding-top: 8px;'>{slab[0]}×{slab[1]}mm</div>", unsafe_allow_html=True)
        # Add small gap after each custom slab
        st.sidebar.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
    
    if custom_slabs_to_remove:
        for i in reversed(custom_slabs_to_remove):
            if i < len(st.session_state.custom_slabs):
                del st.session_state.custom_slabs[i]
        st.rerun()

# Combine slabs
slab_sizes = st.session_state.selected_slabs + st.session_state.custom_slabs

st.sidebar.markdown("---")

def unit_too_large(unit, slab_w, slab_h):
    w, h = unit["width"], unit["height"]
    return not ((w <= slab_w and h <= slab_h) or (h <= slab_w and w <= slab_h))

# -----------------------------
# Sidebar - Unit Input
# -----------------------------
st.sidebar.header("📦 Finished Units")
st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# Bulk Import Section
st.sidebar.subheader("📁 Bulk Import")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload BOQ File",
    type=['txt', 'csv', 'pdf', 'png', 'jpg', 'jpeg'],
    help="Drag and drop your bill of quantities file (.txt, .csv, .pdf, or image)"
)

if uploaded_file is not None:
    if st.sidebar.button("Import from File", type="primary", use_container_width=True):
        imported_units = parse_uploaded_file(uploaded_file)
        if imported_units:
            # First, consolidate any existing manual input rows
            consolidated_units = {}
            
            # Add existing units from the list
            for unit in st.session_state.units:
                forced_key = tuple(unit["forced_slabs"][0]) if unit.get("forced_slabs") else None
                unit_key = (unit["width"], unit["height"], forced_key)
                
                if unit_key in consolidated_units:
                    consolidated_units[unit_key]["quantity"] += unit["quantity"]
                else:
                    consolidated_units[unit_key] = {
                        "width": unit["width"],
                        "height": unit["height"],
                        "quantity": unit["quantity"],
                        "forced_slabs": unit.get("forced_slabs", [])
                    }
            
            # Add any units from manual input rows that haven't been added yet
            for row_data in st.session_state.unit_input_rows:
                if row_data["width"] and row_data["height"] and row_data["quantity"]:
                    forced_slabs = []
                    if row_data["forced"] and row_data["forced"] != "Any":
                        for slab in slab_sizes:
                            if f"{slab[0]}×{slab[1]}" == row_data["forced"]:
                                forced_slabs = [slab]
                                break
                    
                    forced_key = tuple(forced_slabs[0]) if forced_slabs else None
                    unit_key = (row_data["width"], row_data["height"], forced_key)
                    
                    if unit_key in consolidated_units:
                        consolidated_units[unit_key]["quantity"] += row_data["quantity"]
                    else:
                        consolidated_units[unit_key] = {
                            "width": row_data["width"],
                            "height": row_data["height"],
                            "quantity": row_data["quantity"],
                            "forced_slabs": forced_slabs
                        }
            
            # Add imported units to consolidated list
            for unit in imported_units:
                unit_key = (unit["width"], unit["height"], None)  # Imported units have no forced slabs
                
                if unit_key in consolidated_units:
                    consolidated_units[unit_key]["quantity"] += unit["quantity"]
                else:
                    consolidated_units[unit_key] = {
                        "width": unit["width"],
                        "height": unit["height"],
                        "quantity": unit["quantity"],
                        "forced_slabs": []
                    }
            
            # Update the session state with all consolidated units
            st.session_state.units = []
            for i, (unit_key, unit_data) in enumerate(consolidated_units.items()):
                st.session_state.units.append({
                    "width": unit_data["width"],
                    "height": unit_data["height"],
                    "quantity": unit_data["quantity"],
                    "forced_slabs": unit_data["forced_slabs"],
                    "order": i
                })
            
            # Clear manual input rows after successful import
            st.session_state.unit_input_rows = [{"width": "", "height": "", "quantity": 1, "forced": "Any"}]
            
            st.sidebar.success(f"✅ Imported {len(imported_units)} unit types and updated list!")
            st.rerun()

# Text Area for Copy/Paste
bulk_text = st.sidebar.text_area(
    "Or Paste BOQ Text",
    placeholder="x1 1650×560",
    height=120,
    help="Paste your BOQ text. Supports formats like: x1 1650×560, 1x 1650×560, 1 no. 1650×560"
)

if bulk_text.strip():
    if st.sidebar.button("Import from Text", type="primary", use_container_width=True):
        imported_units = parse_boq_text(bulk_text)
        if imported_units:
            # First, consolidate any existing manual input rows and existing units
            consolidated_units = {}
            
            # Add existing units from the list
            for unit in st.session_state.units:
                forced_key = tuple(unit["forced_slabs"][0]) if unit.get("forced_slabs") else None
                unit_key = (unit["width"], unit["height"], forced_key)
                
                if unit_key in consolidated_units:
                    consolidated_units[unit_key]["quantity"] += unit["quantity"]
                else:
                    consolidated_units[unit_key] = {
                        "width": unit["width"],
                        "height": unit["height"],
                        "quantity": unit["quantity"],
                        "forced_slabs": unit.get("forced_slabs", [])
                    }
            
            # Add any units from manual input rows
            for row_data in st.session_state.unit_input_rows:
                if row_data["width"] and row_data["height"] and row_data["quantity"]:
                    forced_slabs = []
                    if row_data["forced"] and row_data["forced"] != "Any":
                        for slab in slab_sizes:
                            if f"{slab[0]}×{slab[1]}" == row_data["forced"]:
                                forced_slabs = [slab]
                                break
                    
                    forced_key = tuple(forced_slabs[0]) if forced_slabs else None
                    unit_key = (row_data["width"], row_data["height"], forced_key)
                    
                    if unit_key in consolidated_units:
                        consolidated_units[unit_key]["quantity"] += row_data["quantity"]
                    else:
                        consolidated_units[unit_key] = {
                            "width": row_data["width"],
                            "height": row_data["height"],
                            "quantity": row_data["quantity"],
                            "forced_slabs": forced_slabs
                        }
            
            # Add imported units to consolidated list
            for unit in imported_units:
                unit_key = (unit["width"], unit["height"], None)  # Imported units have no forced slabs
                
                if unit_key in consolidated_units:
                    consolidated_units[unit_key]["quantity"] += unit["quantity"]
                else:
                    consolidated_units[unit_key] = {
                        "width": unit["width"],
                        "height": unit["height"],
                        "quantity": unit["quantity"],
                        "forced_slabs": []
                    }
            
            # Update the session state with all consolidated units
            st.session_state.units = []
            for i, (unit_key, unit_data) in enumerate(consolidated_units.items()):
                st.session_state.units.append({
                    "width": unit_data["width"],
                    "height": unit_data["height"],
                    "quantity": unit_data["quantity"],
                    "forced_slabs": unit_data["forced_slabs"],
                    "order": i
                })
            
            # Clear manual input rows after successful import
            st.session_state.unit_input_rows = [{"width": "", "height": "", "quantity": 1, "forced": "Any"}]
            
            st.sidebar.success(f"✅ Imported {len(imported_units)} unit types and updated list!")
            st.rerun()
        else:
            st.sidebar.error("❌ No valid units found. Check your format.")

st.sidebar.markdown("---")

# Manual Input Section
# Initialize state
if "manual_input_enabled" not in st.session_state:
    st.session_state.manual_input_enabled = False

# Add some spacing after the separator
st.sidebar.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Create columns for header and toggle
col1, col2 = st.sidebar.columns([4, 1])

with col1:
    # Add padding to push the header down
    st.markdown("<div style='padding-top: 0.5rem;'><h3 style='margin: 0;'>✏️ Manual Input</h3></div>", unsafe_allow_html=True)

with col2:
    # No additional spacing needed - the toggle should align naturally
    manual_enabled = st.toggle(
        "",
        value=st.session_state.manual_input_enabled,
        key="manual_toggle_switch",
        help="Toggle manual input on/off",
        label_visibility="collapsed"
    )
    if manual_enabled != st.session_state.manual_input_enabled:
        st.session_state.manual_input_enabled = manual_enabled
        st.rerun()

# Add custom CSS to style the toggle
st.markdown("""
<style>
    /* Style the toggle switch */
    .stToggle > label {
        width: 50px !important;
    }
    
    .stToggle > label > div {
        background-color: #e0e0e0 !important;
    }
    
    .stToggle > label > div[data-checked="true"] {
        background-color: #2196F3 !important;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.manual_input_enabled:
    # Add spacing before input rows to prevent overlap
    st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # Show input rows
    rows_to_remove = []
    for row_idx, row_data in enumerate(st.session_state.unit_input_rows):
        col1, col2, col3, col4, col5 = st.sidebar.columns([1, 1, 0.7, 1.5, 0.4])
        
        with col1:
            label_vis = "visible" if row_idx == 0 else "collapsed"
            width = st.number_input("Width", min_value=1, value=row_data["width"] if row_data["width"] != "" else 1, step=1, key=f"width_input_{row_idx}", label_visibility=label_vis)
        
        with col2:
            label_vis = "visible" if row_idx == 0 else "collapsed"
            height = st.number_input("Height", min_value=1, value=row_data["height"] if row_data["height"] != "" else 1, step=1, key=f"height_input_{row_idx}", label_visibility=label_vis)
        
        with col3:
            label_vis = "visible" if row_idx == 0 else "collapsed"
            quantity = st.number_input("Qty", min_value=1, value=row_data["quantity"], step=1, key=f"quantity_input_{row_idx}", label_visibility=label_vis)
        
        with col4:
            slab_options = ["Any"] + [f"{s[0]}×{s[1]}" for s in slab_sizes] if slab_sizes else ["Any"]
            current_index = 0
            if row_data["forced"] in slab_options:
                current_index = slab_options.index(row_data["forced"])
            
            label_vis = "visible" if row_idx == 0 else "collapsed"
            forced = st.selectbox(
                "Force Cutting From",
                slab_options,
                index=current_index,
                key=f"forced_input_{row_idx}",
                label_visibility=label_vis
            )
        
        with col5:
            if row_idx == 0:
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            
            if len(st.session_state.unit_input_rows) > 1 or row_idx > 0:
                if st.button("×", key=f"remove_row_{row_idx}", help="Remove this row"):
                    rows_to_remove.append(row_idx)
        
        # Add same spacing as slab buttons between rows
        st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        st.session_state.unit_input_rows[row_idx] = {
            "width": width,
            "height": height,
            "quantity": quantity,
            "forced": forced
        }
    
    # Remove rows
    if rows_to_remove:
        for row_idx in reversed(rows_to_remove):
            if row_idx < len(st.session_state.unit_input_rows):
                del st.session_state.unit_input_rows[row_idx]
        st.rerun()
    
    # Buttons with same spacing as input rows
    st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("➕ Add Unit", type="primary", use_container_width=True):
        st.session_state.unit_input_rows.append({"width": "", "height": "", "quantity": 1, "forced": "Any"})
        st.rerun()
    
    if col2.button("📝 Update List", type="secondary", use_container_width=True):
        # Consolidate units from input rows
        consolidated_units = {}
        
        for row_data in st.session_state.unit_input_rows:
            if row_data["width"] and row_data["height"] and row_data["quantity"]:
                forced_slabs = []
                if row_data["forced"] and row_data["forced"] != "Any":
                    for slab in slab_sizes:
                        if f"{slab[0]}×{slab[1]}" == row_data["forced"]:
                            forced_slabs = [slab]
                            break
                
                forced_key = tuple(forced_slabs[0]) if forced_slabs else None
                unit_key = (row_data["width"], row_data["height"], forced_key)
                
                if unit_key in consolidated_units:
                    consolidated_units[unit_key]["quantity"] += row_data["quantity"]
                else:
                    consolidated_units[unit_key] = {
                        "width": row_data["width"],
                        "height": row_data["height"],
                        "quantity": row_data["quantity"],
                        "forced_slabs": forced_slabs
                    }
        
        # Update the units list with consolidated data
        st.session_state.units = []
        for i, (unit_key, unit_data) in enumerate(consolidated_units.items()):
            st.session_state.units.append({
                "width": unit_data["width"],
                "height": unit_data["height"],
                "quantity": unit_data["quantity"],
                "forced_slabs": unit_data["forced_slabs"],
                "order": i
            })
        
        st.rerun()

# Show current units
if st.session_state.units:
    st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Current Units:**")
    for unit in st.session_state.units:
        forced_text = ""
        if unit['forced_slabs']:
            forced_text = f" → {unit['forced_slabs'][0][0]}×{unit['forced_slabs'][0][1]}"
        st.sidebar.markdown(f"• {unit['quantity']}no. {unit['width']}×{unit['height']}mm{forced_text}")

# Clear all button
if st.session_state.units:
    st.sidebar.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Clear All", type="secondary", use_container_width=True):
        st.session_state.units = []
        st.session_state.unit_input_rows = [{"width": "", "height": "", "quantity": 1, "forced": "Any"}]
        st.rerun()

# -----------------------------
# Main Results Section - COMPLETELY NEW ALGORITHM
# -----------------------------
if slab_sizes and st.session_state.units:
    st.header("📊 Optimization Results")
    
    with st.spinner("Optimizing cut layouts..."):
        # STEP 1: Find optimal allocation (each unit goes to ONE slab type only)
        allocation = optimize_unit_allocation(st.session_state.units, slab_sizes)
        
        # STEP 2: Generate results ONLY for allocated slabs
        slab_outputs = []
        global_boqlines = {}
        
        for slab, allocated_units in allocation.items():
            sw, sh = slab
            
            # Check if this is a scant slab
            is_scant = slab in st.session_state.scant_slabs
            
            if is_scant:
                # For scants, just calculate area without cut optimization
                total_area = 0
                produced = {}
                
                for unit in allocated_units:
                    qty = unit["quantity"]
                    produced[unit["order"]] = qty
                    total_area += unit["width"] * unit["height"] * qty
                
                # Calculate number of scants needed
                scant_area = sw * sh
                scants_needed = math.ceil(total_area / scant_area)
                
                slab_outputs.append({
                    "slab": f"{sw}×{sh}",
                    "slab_size": (sw, sh),
                    "units": produced,
                    "cut_length": 0,  # No cuts for scants
                    "area": total_area,
                    "waste_area": (scants_needed * scant_area) - total_area,
                    "efficiency": (total_area / (scants_needed * scant_area)) * 100 if scants_needed > 0 else 0,
                    "slab_count": scants_needed,
                    "is_scant": True
                })
                
                # Build global BOQ
                for order, qty in produced.items():
                    original_unit = next(u for u in st.session_state.units if u["order"] == order)
                    key = (original_unit['width'], original_unit['height'])
                    global_boqlines[key] = global_boqlines.get(key, 0) + qty
                    
            else:
                # Regular slab processing with cut optimization
                remaining = copy.deepcopy(allocated_units)
                produced = {}
                cut_length = 0
                slab_count = 0
                
                # Pack slabs until all allocated units are produced
                while any(u["quantity"] > 0 for u in remaining):
                    best = pack_one_slab(sw, sh, remaining)
                    if not best["layout"]:
                        break
                    
                    slab_count += 1
                    counts = {}
                    for p in best["layout"]:
                        counts[p.unit_order] = counts.get(p.unit_order, 0) + 1
                    
                    for u in remaining:
                        if u["order"] in counts:
                            used = counts[u["order"]]
                            u["quantity"] -= used
                            produced[u["order"]] = produced.get(u["order"], 0) + used
                    
                    cut_length += best["cost"]
                
                if produced:  # Only add if something was actually produced
                    # Calculate areas using original units for reference
                    area = 0
                    for order, qty in produced.items():
                        original_unit = next(u for u in st.session_state.units if u["order"] == order)
                        area += original_unit["width"] * original_unit["height"] * qty
                    
                    waste_area = (sw * sh * slab_count) - area
                    efficiency = (area / (sw * sh * slab_count)) * 100 if slab_count > 0 else 0
                    
                    slab_outputs.append({
                        "slab": f"{sw}×{sh}",
                        "slab_size": (sw, sh),
                        "units": produced,
                        "cut_length": cut_length,
                        "area": area,
                        "waste_area": waste_area,
                        "efficiency": efficiency,
                        "slab_count": slab_count,
                        "is_scant": False
                    })
                    
                    # Build global BOQ
                    for order, qty in produced.items():
                        original_unit = next(u for u in st.session_state.units if u["order"] == order)
                        key = (original_unit['width'], original_unit['height'])
                        global_boqlines[key] = global_boqlines.get(key, 0) + qty
    
    if slab_outputs:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_slabs = sum(result["slab_count"] for result in slab_outputs)
        total_cut_length = sum(result["cut_length"] for result in slab_outputs)
        total_produced_area = sum(result["area"] for result in slab_outputs)
        avg_efficiency = sum(result["efficiency"] for result in slab_outputs) / len(slab_outputs)
        
        col1.metric("Total Slabs", total_slabs)
        col2.metric("Total Cut Length", f"{total_cut_length / 1000:.2f} m")
        col3.metric("Produced Area", f"{total_produced_area / 1e6:.2f} m²")
        col4.metric("Avg Efficiency", f"{avg_efficiency:.1f}%")
        
        st.markdown("---")
        
        # Detailed results
        for result in slab_outputs:
            slab_name = result['slab']
            # Check if it's a scant and add the name
            if result.get('is_scant'):
                # Find which scant type this is
                for _, actual_size, label in scants_slabs:
                    if actual_size == result['slab_size']:
                        slab_name = label.replace('\n', ' ')
                        break
            
            with st.expander(f"{slab_name}mm {'Scant' if result.get('is_scant') else 'Slab'}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("No. of {'Scants' if result.get('is_scant') else 'Slabs'} Used", result['slab_count'])
                
                with col2:
                    if result.get('is_scant'):
                        st.metric("Total Area Required", f"{result['area'] / 1e6:.2f} m²")
                    else:
                        st.metric("Cut Length", f"{result['cut_length'] / 1000:.2f} m")
                
                with col3:
                    st.metric("{'Material' if result.get('is_scant') else 'Total'} Area", f"{result['area'] / 1e6:.2f} m²")
                
                with col4:
                    st.markdown("**Units Produced:**")
                    for order in sorted(result["units"]):
                        u = next(u for u in st.session_state.units if u["order"] == order)
                        qty = result['units'][order]
                        st.markdown(f"<div style='margin-bottom: 0px; line-height: 1.2;'>• {qty}no. {u['width']}×{u['height']}mm</div>", unsafe_allow_html=True)
        
        # Global summary
        if global_boqlines:
            st.markdown("---")
            st.subheader("📋 Overall Bill of Quantities")
            st.markdown("**Total units to be produced:**")
            
            for (w, h), qty in sorted(global_boqlines.items()):
                st.markdown(f"• **{qty}no.** {w}×{h}mm")
        
        # Check for unproduced units
        produced_units = {}
        for result in slab_outputs:
            for order, qty in result["units"].items():
                produced_units[order] = produced_units.get(order, 0) + qty
        
        unproduced = []
        for u in st.session_state.units:
            produced_qty = produced_units.get(u["order"], 0)
            if produced_qty < u["quantity"]:
                unproduced.append({
                    "unit": u,
                    "missing": u["quantity"] - produced_qty
                })
        
        if unproduced:
            st.markdown("---")
            st.error("⚠️ **Some units could not be produced:**")
            for item in unproduced:
                u = item["unit"]
                missing = item["missing"]
                st.markdown(f"• **{missing}no.** {u['width']}×{u['height']}mm (too large for selected slabs)")
    
    else:
        st.warning("⚠️ No units could be produced with the selected slab sizes. Please check your unit dimensions and slab selections.")

elif not slab_sizes:
    st.info("👆 Please select at least one slab size from the sidebar to begin optimization.")

elif not st.session_state.units:
    st.info("👆 Please add some finished units using the sidebar form to begin optimization.")

else:
    st.info("👆 Please select slab sizes and add finished units to begin optimization.")
