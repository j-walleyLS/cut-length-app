# Streamlit Cut Length Optimiser

import math
import copy
import streamlit as st

BLADE_WIDTH = 4
MAX_BRANCH_COUNT = 10000

# -----------------------------
# Utilities
# -----------------------------
def parse_dimensions(dim_str):
    parts = dim_str.lower().replace(" ", "").split("x")
    return int(parts[0]), int(parts[1])

def unit_too_large(unit, slab_w, slab_h):
    w, h = unit["width"], unit["height"]
    return not ((w <= slab_w and h <= slab_h) or (h <= slab_w and w <= slab_h))

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

# -----------------------------
# Main UI
# -----------------------------
st.title("🪚 Cut Length Optimiser")
st.markdown("*Optimize material cutting from various slab sizes*")

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
                    key=f"slab_{actual_size}",
                    help=f"Click to {'remove' if is_selected else 'add'} {display_size[0]}×{display_size[1]}mm slab",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    if actual_size in st.session_state.selected_slabs:
                        st.session_state.selected_slabs.remove(actual_size)
                    else:
                        st.session_state.selected_slabs.append(actual_size)
                    st.rerun()
        
        # Second button in pair (if exists)
        if i + 1 < len(slab_list):
            display_size, actual_size, label = slab_list[i + 1]
            is_selected = actual_size in st.session_state.selected_slabs
            
            with col2:
                if st.button(
                    f"{'✓ ' if is_selected else ''}{label}",
                    key=f"slab_{actual_size}",
                    help=f"Click to {'remove' if is_selected else 'add'} {display_size[0]}×{display_size[1]}mm slab",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    if actual_size in st.session_state.selected_slabs:
                        st.session_state.selected_slabs.remove(actual_size)
                    else:
                        st.session_state.selected_slabs.append(actual_size)
                    st.rerun()
        
        # Add spacing between button rows, but not after the last row
        if i + 2 < len(slab_list):  # Only add spacing if not the last row
            st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Create all slab button sections
create_slab_buttons(paving_slabs, "Paving")
create_slab_buttons(treads_slabs, "Treads")
create_slab_buttons(italian_porcelain_slabs, "Italian Porcelain")

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

# -----------------------------
# Sidebar - Unit Input
# -----------------------------
st.sidebar.header("📦 Finished Units")
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
                    "slab_count": slab_count
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
            with st.expander(f"{result['slab']}mm Slab", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("No. of Slabs Used", result['slab_count'])
                
                with col2:
                    st.metric("Cut Length", f"{result['cut_length'] / 1000:.2f} m")
                
                with col3:
                    st.metric("Total Area", f"{result['area'] / 1e6:.2f} m²")
                
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
