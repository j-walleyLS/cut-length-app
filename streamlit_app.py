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

def filter_units_for_slab(units, slab):
    sw, sh = slab
    return [u for u in units if ((not u["forced_slabs"]) or (slab in u["forced_slabs"])) and not unit_too_large(u, sw, sh)]

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
    pieces = []
    for u in units:
        for _ in range(u["quantity"]):
            pieces.append((u["width"], u["height"], u["order"]))
    best_solution = {"layout": None, "cost": float("inf"), "count": 0}
    branch_count = [0]
    repack_dfs([(0, 0, slab_width, slab_height)], pieces, [], best_solution, slab_width, slab_height, branch_count)
    return best_solution

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(
    page_title="Cut Length Optimiser",
    page_icon="🪚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to remove sidebar spacing and change button colors
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
    
    /* Consistent spacing for all sidebar elements */
    .stSidebar .stMarkdown {
        margin-bottom: 0.75rem;
    }
    .stSidebar .stTextInput {
        margin-bottom: 0.5rem;
    }
    
    /* Consistent spacing for input fields */
    .stSidebar .stNumberInput {
        margin-bottom: 14rem;
    }
    .stSidebar .stSelectbox {
        margin-bottom: 0.4rem;
    }
    
    /* Tighter spacing for slab buttons */
    .stSidebar .stButton {
        margin-bottom: 0.3rem;
    }
    
    /* Larger gaps for main headers */
    .stSidebar .stMarkdown h3 {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Larger gaps for section headers */
    .stSidebar .stMarkdown h2 {
        margin-bottom: 1.2rem !important;
    }
    
    /* Change button colors - primary buttons (green) */
    .stButton > button[kind="primary"] {
        background-color: #28a745;
        border-color: #28a745;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    .stButton > button[kind="primary"]:focus {
        background-color: #218838;
        border-color: #1e7e34;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5);
    }
    
    /* Make selected slab buttons green too */
    .stButton > button[kind="primary"]:active {
        background-color: #1e7e34;
        border-color: #1c7430;
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

# -----------------------------
# Main UI
# -----------------------------
st.title("🪚 Cut Length Optimiser")
st.markdown("*Optimize material cutting from various slab sizes*")

# -----------------------------
# Sidebar - Slab Selection
# -----------------------------
st.sidebar.markdown("### 📐 Available Slab Sizes")

# Define all available slab sizes in specific column order
col1_slabs = [(600, 600), (900, 700), (900, 500), (2000, 500)]
col2_slabs = [(900, 600), (1800, 700), (1500, 500)]

# Create a responsive grid layout for slab selection
col1, col2 = st.sidebar.columns(2)

# Column 1 slabs
for slab in col1_slabs:
    slab_key = f"{slab[0]}×{slab[1]}"
    is_selected = slab in st.session_state.selected_slabs
    
    if col1.button(
        f"{'✓ ' if is_selected else ''}{slab_key}",
        key=f"slab_{slab}",
        help=f"Click to {'remove' if is_selected else 'add'} {slab_key}mm slab",
        type="primary" if is_selected else "secondary",
        use_container_width=True
    ):
        if slab in st.session_state.selected_slabs:
            st.session_state.selected_slabs.remove(slab)
        else:
            st.session_state.selected_slabs.append(slab)
        st.rerun()

# Column 2 slabs
for slab in col2_slabs:
    slab_key = f"{slab[0]}×{slab[1]}"
    is_selected = slab in st.session_state.selected_slabs
    
    if col2.button(
        f"{'✓ ' if is_selected else ''}{slab_key}",
        key=f"slab_{slab}",
        help=f"Click to {'remove' if is_selected else 'add'} {slab_key}mm slab",
        type="primary" if is_selected else "secondary",
        use_container_width=True
    ):
        if slab in st.session_state.selected_slabs:
            st.session_state.selected_slabs.remove(slab)
        else:
            st.session_state.selected_slabs.append(slab)
        st.rerun()

st.sidebar.markdown("---")

# Custom slab input with improved UI
st.sidebar.subheader("Custom Slab Sizes")

# Initialize custom input in session state
if "custom_input" not in st.session_state:
    st.session_state.custom_input = ""

custom_input = st.sidebar.text_input(
    "Enter one size at a time",
    value=st.session_state.custom_input,
    placeholder="e.g. 800x400",
    help="Enter width x height and press Enter",
    key="custom_slab_input"
)

# Process single custom slab input
if custom_input != st.session_state.custom_input:
    st.session_state.custom_input = custom_input
    
    if custom_input.strip() and 'x' in custom_input.lower():
        try:
            new_slab = parse_dimensions(custom_input)
            if new_slab not in st.session_state.custom_slabs:
                st.session_state.custom_slabs.append(new_slab)
                # Clear the input by resetting the key
                st.session_state.custom_input = ""
                st.rerun()
        except Exception as e:
            st.sidebar.error("❌ Invalid format. Use: 800x400")

# Display custom slabs list
if st.session_state.custom_slabs:
    st.sidebar.markdown("**Custom Slabs:**")
    st.sidebar.markdown("")  # Add space line before first entry
    custom_slabs_to_remove = []
    for i, slab in enumerate(st.session_state.custom_slabs):
        col1, col2 = st.sidebar.columns([0.3, 2.7])
        if col1.button("×", key=f"remove_custom_{i}_{slab[0]}_{slab[1]}", help="Remove this custom slab"):
            custom_slabs_to_remove.append(i)
        col2.markdown(f"<div style='text-align: left; padding-top: 8px;'>{slab[0]}×{slab[1]}mm</div>", unsafe_allow_html=True)
    
    # Remove custom slabs (in reverse order) - avoid rerun in loop
    if custom_slabs_to_remove:
        for i in reversed(custom_slabs_to_remove):
            if i < len(st.session_state.custom_slabs):
                del st.session_state.custom_slabs[i]
        st.rerun()

# Combine selected default slabs with custom slabs
slab_sizes = st.session_state.selected_slabs + st.session_state.custom_slabs

# -----------------------------
# Sidebar - Unit Input
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("📦 Finished Units")
st.sidebar.markdown("")  # Add larger gap after header

# Initialize edit mode and unit input fields in session state
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "unit_input_rows" not in st.session_state:
    st.session_state.unit_input_rows = [{"width": 300, "height": 200, "quantity": 1, "forced": "Any"}]
if "unit_width" not in st.session_state:
    st.session_state.unit_width = 300
if "unit_height" not in st.session_state:
    st.session_state.unit_height = 200
if "unit_quantity" not in st.session_state:
    st.session_state.unit_quantity = 1
if "unit_forced" not in st.session_state:
    st.session_state.unit_forced = []

# Show input rows for adding/editing units
rows_to_remove = []
for row_idx, row_data in enumerate(st.session_state.unit_input_rows):
    col1, col2, col3, col4, col5 = st.sidebar.columns([1, 1, 0.7, 1.5, 0.4])
    
    with col1:
        # Only show label on first row
        label_vis = "visible" if row_idx == 0 else "collapsed"
        width = st.number_input("Width", min_value=1, value=row_data["width"], step=1, key=f"width_input_{row_idx}", label_visibility=label_vis)
    
    with col2:
        label_vis = "visible" if row_idx == 0 else "collapsed"
        height = st.number_input("Height", min_value=1, value=row_data["height"], step=1, key=f"height_input_{row_idx}", label_visibility=label_vis)
    
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
        # Add padding to align with input fields only on first row
        if row_idx == 0:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Space for label
        
        if len(st.session_state.unit_input_rows) > 1 or row_idx > 0:  # Always show remove except for single first row
            if st.button("×", key=f"remove_row_{row_idx}", help="Remove this row"):
                rows_to_remove.append(row_idx)
    
    # Update row data
    st.session_state.unit_input_rows[row_idx] = {
        "width": width,
        "height": height,
        "quantity": quantity,
        "forced": forced
    }

# Remove rows (in reverse order) - avoid rerun in loop
if rows_to_remove:
    for row_idx in reversed(rows_to_remove):
        if row_idx < len(st.session_state.unit_input_rows):
            del st.session_state.unit_input_rows[row_idx]
    st.rerun()

# Buttons row
col1, col2 = st.sidebar.columns(2)

# Add unit button - adds the current rows to units and creates a new input row
if col1.button("➕ Add Unit", type="primary", use_container_width=True):
    # Add all filled rows to units
    for row_data in st.session_state.unit_input_rows:
        if row_data["width"] and row_data["height"] and row_data["quantity"]:
            # Process forced slab selection
            forced_slabs = []
            if row_data["forced"] and row_data["forced"] != "Any":
                for slab in slab_sizes:
                    if f"{slab[0]}×{slab[1]}" == row_data["forced"]:
                        forced_slabs = [slab]
                        break
            
            # Add the unit to a temporary list (don't update main units list yet)
            # This will be processed when Update List is clicked
    
    # Add a new empty row for next input
    st.session_state.unit_input_rows.append({"width": 300, "height": 200, "quantity": 1, "forced": "Any"})
    st.rerun()

# Update list button - consolidates units and keeps input fields
if col2.button("📝 Update List", type="secondary", use_container_width=True):
    # Create consolidated units from all input rows
    consolidated_units = {}
    
    for row_data in st.session_state.unit_input_rows:
        if row_data["width"] and row_data["height"] and row_data["quantity"]:
            # Process forced slab selection
            forced_slabs = []
            if row_data["forced"] and row_data["forced"] != "Any":
                for slab in slab_sizes:
                    if f"{slab[0]}×{slab[1]}" == row_data["forced"]:
                        forced_slabs = [slab]
                        break
            
            # Create key for consolidation (width, height, forced_slab)
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
    
    # Replace units list with consolidated units
    st.session_state.units = []
    for i, (unit_key, unit_data) in enumerate(consolidated_units.items()):
        st.session_state.units.append({
            "width": unit_data["width"],
            "height": unit_data["height"],
            "quantity": unit_data["quantity"],
            "forced_slabs": unit_data["forced_slabs"],
            "order": i
        })
    
    # Keep the input rows as they are (don't reset them)
    st.rerun()

# Show the current units list
if st.session_state.units:
    st.sidebar.markdown("**Current Units:**")
    for unit in st.session_state.units:
        forced_text = ""
        if unit['forced_slabs']:
            forced_text = f" → {unit['forced_slabs'][0][0]}×{unit['forced_slabs'][0][1]}"
        st.sidebar.markdown(f"• {unit['quantity']}no. {unit['width']}×{unit['height']}mm{forced_text}")

# Clear all button
if st.session_state.units:
    if st.sidebar.button("Clear All", type="secondary", use_container_width=True):
        st.session_state.units = []
        st.session_state.unit_input_rows = [{"width": 300, "height": 200, "quantity": 1, "forced": "Any"}]
        st.session_state.edit_mode = False
        st.rerun()

# -----------------------------
# Main Results Section
# -----------------------------
if slab_sizes and st.session_state.units:
    st.header("📊 Optimization Results")
    
    # Add a progress indicator
    with st.spinner("Optimizing cut layouts..."):
        global_boqlines = {}
        slab_outputs = []
        
        for slab in slab_sizes:
            sw, sh = slab
            slab_units = filter_units_for_slab(st.session_state.units, slab)
            if not slab_units:
                continue
            
            remaining = copy.deepcopy(slab_units)
            produced = {}
            cut_length = 0
            slab_count = 0
            
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
                area = sum(u["width"] * u["height"] * qty for u in st.session_state.units for order, qty in produced.items() if u["order"] == order)
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
    
    # Display results in a more organized way
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
        
        # Detailed results for each slab size
        for result in slab_outputs:
            with st.expander(f"## {result['slab_count']}no. **{result['slab']}mm Slab**", expanded=True):
                
                # Metrics for this slab size - only show cut length and total area
                col1, col2 = st.columns(2)
                col1.metric("Cut Length", f"{result['cut_length'] / 1000:.2f} m")
                col2.metric("Total Area", f"{result['area'] / 1e6:.2f} m²")
                
                # Bill of quantities for this slab
                st.markdown("**Units Produced:**")
                boq_data = []
                for order in sorted(result["units"]):
                    u = next(u for u in st.session_state.units if u["order"] == order)
                    qty = result['units'][order]
                    boq_data.append(f"• {qty}no. {u['width']}×{u['height']}mm")
                    global_boqlines[(u['width'], u['height'])] = global_boqlines.get((u['width'], u['height']), 0) + qty
                
                st.markdown("\n".join(boq_data))
        
        # Global summary
        if global_boqlines:
            st.markdown("---")
            st.subheader("📋 Overall Bill of Quantities")
            st.markdown("**Total units to be produced:**")
            
            for (w, h), qty in sorted(global_boqlines.items()):
                st.markdown(f"• **{qty}no.** {w}×{h}mm")
        
        # Check for units that couldn't be produced
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
