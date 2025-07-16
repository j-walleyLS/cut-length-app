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
st.sidebar.header("📐 Available Slab Sizes")

# Define all available slab sizes
def_slabs = [(900, 600), (600, 600), (1800, 700), (900, 500), (1500, 500), (2000, 500), (900, 700)]

# Create a responsive grid layout for slab selection
col1, col2 = st.sidebar.columns(2)

for i, slab in enumerate(def_slabs):
    col = col1 if i % 2 == 0 else col2
    slab_key = f"{slab[0]}×{slab[1]}"
    
    # Check if this slab is currently selected
    is_selected = slab in st.session_state.selected_slabs
    
    # Create button with different styling based on selection
    if col.button(
        f"{'✓ ' if is_selected else ''}{slab_key}",
        key=f"slab_{slab}",
        help=f"Click to {'remove' if is_selected else 'add'} {slab_key}mm slab",
        type="primary" if is_selected else "secondary",
        use_container_width=True
    ):
        # Toggle selection
        if slab in st.session_state.selected_slabs:
            st.session_state.selected_slabs.remove(slab)
        else:
            st.session_state.selected_slabs.append(slab)
        st.rerun()

# Show currently selected slabs with better formatting
if st.session_state.selected_slabs:
    st.sidebar.success(f"**{len(st.session_state.selected_slabs)} slab size(s) selected**")
    with st.sidebar.expander("View Selected Slabs", expanded=False):
        for slab in sorted(st.session_state.selected_slabs, key=lambda x: (x[0], x[1])):
            st.markdown(f"• {slab[0]}×{slab[1]}mm")
else:
    st.sidebar.info("⚠️ Please select at least one slab size")

# Add utility buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("Select All", help="Select all available slab sizes"):
    st.session_state.selected_slabs = def_slabs.copy()
    st.rerun()

if col2.button("Clear All", help="Clear all selected slabs"):
    st.session_state.selected_slabs = []
    st.rerun()

st.sidebar.markdown("---")

# Custom slab input with improved UI
st.sidebar.subheader("Custom Slab Sizes")
custom_input = st.sidebar.text_input(
    "Enter custom sizes",
    placeholder="e.g. 800x400, 1000x500",
    help="Separate multiple sizes with commas"
)

if custom_input:
    try:
        st.session_state.custom_slabs = [parse_dimensions(x) for x in custom_input.split(",") if x.strip()]
        if st.session_state.custom_slabs:
            st.sidebar.success(f"✓ {len(st.session_state.custom_slabs)} custom slab(s) added")
    except:
        st.sidebar.error("❌ Invalid format. Use: 800x400, 1000x500")

# Combine selected default slabs with custom slabs
slab_sizes = st.session_state.selected_slabs + st.session_state.custom_slabs

# -----------------------------
# Sidebar - Unit Input
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("📦 Finished Units")

with st.sidebar.form("unit_form"):
    st.markdown("**Add a new unit:**")
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width (mm)", min_value=1, value=300, step=1)
    with col2:
        height = st.number_input("Height (mm)", min_value=1, value=200, step=1)
    
    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
    
    # Only show forced slab selection if slabs are available
    forced = []
    if slab_sizes:
        forced = st.multiselect(
            "Force specific slab(s)? (optional)",
            slab_sizes,
            default=[],
            help="Leave empty to allow any suitable slab"
        )
    
    col1, col2 = st.columns(2)
    submitted = col1.form_submit_button("➕ Add Unit", type="primary")
    
    if submitted:
        st.session_state.units.append({
            "width": width,
            "height": height,
            "quantity": quantity,
            "forced_slabs": forced,
            "order": len(st.session_state.units)
        })
        st.success(f"✓ Added {quantity}× {width}×{height}mm unit(s)")

# Unit management
if st.session_state.units:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Current Units")
    
    total_units = sum(u["quantity"] for u in st.session_state.units)
    st.sidebar.info(f"**Total: {total_units} unit(s) in {len(st.session_state.units)} type(s)**")
    
    # Show units with delete functionality
    units_to_remove = []
    for i, unit in enumerate(st.session_state.units):
        with st.sidebar.expander(f"{unit['quantity']}× {unit['width']}×{unit['height']}mm"):
            st.write(f"**Dimensions:** {unit['width']}×{unit['height']}mm")
            st.write(f"**Quantity:** {unit['quantity']}")
            if unit['forced_slabs']:
                st.write(f"**Forced slabs:** {', '.join(f'{s[0]}×{s[1]}' for s in unit['forced_slabs'])}")
            
            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                units_to_remove.append(i)
    
    # Remove units (in reverse order to maintain indices)
    for i in reversed(units_to_remove):
        del st.session_state.units[i]
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear All Units", type="secondary"):
        st.session_state.units = []
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
            with st.expander(f"🔍 **{result['slab']}mm Slab Details** ({result['slab_count']} slab(s) needed)", expanded=True):
                
                # Metrics for this slab size
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Units Placed", sum(result['units'].values()))
                col2.metric("Cut Length", f"{result['cut_length'] / 1000:.2f} m")
                col3.metric("Efficiency", f"{result['efficiency']:.1f}%")
                col4.metric("Waste Area", f"{result['waste_area'] / 1e6:.3f} m²")
                
                # Bill of quantities for this slab
                st.markdown("**Units Produced:**")
                boq_data = []
                for order in sorted(result["units"]):
                    u = next(u for u in st.session_state.units if u["order"] == order)
                    qty = result['units'][order]
                    boq_data.append(f"• {qty}× {u['width']}×{u['height']}mm")
                    global_boqlines[(u['width'], u['height'])] = global_boqlines.get((u['width'], u['height']), 0) + qty
                
                st.markdown("\n".join(boq_data))
        
        # Global summary
        if global_boqlines:
            st.markdown("---")
            st.subheader("📋 Overall Bill of Quantities")
            st.markdown("**Total units to be produced:**")
            
            for (w, h), qty in sorted(global_boqlines.items()):
                st.markdown(f"• **{qty}×** {w}×{h}mm")
        
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
                st.markdown(f"• **{missing}×** {u['width']}×{u['height']}mm (too large for selected slabs)")
    
    else:
        st.warning("⚠️ No units could be produced with the selected slab sizes. Please check your unit dimensions and slab selections.")

elif not slab_sizes:
    st.info("👆 Please select at least one slab size from the sidebar to begin optimization.")

elif not st.session_state.units:
    st.info("👆 Please add some finished units using the sidebar form to begin optimization.")

else:
    st.info("👆 Please select slab sizes and add finished units to begin optimization.")
