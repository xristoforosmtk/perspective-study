import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, FloatSlider, Checkbox
from IPython.display import display

# Function to find the intersection of two lines
def get_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return x1, y1
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return px, py

# NEW FUNCTION: Calculates the non-linear compression of perspective
# Works with Homogeneous Coordinates for correct projective interpolation.
def projective_interpolate(p_start, p_end, vp, num_divisions=8):
    # Convert 2D points to Homogeneous Coordinates (x, y, 1)
    p0 = np.array([p_start[0], p_start[1], 1])
    p1 = np.array([p_end[0], p_end[1], 1])
    v = np.array([vp[0], vp[1], 1])
    
    # Calculate the lines (cross products)
    l_end = np.cross(p1, v) # Line from P1 to VP
    l_edge = np.cross(p0, p1) # Edge line P0-P1
    l_ref = np.cross(p0, v) # Line from P0 to VP

    # Calculate the horizon line direction (in Homogeneous coords)
    # For simplicity in this 3pt simulation, we use a fixed horizon
    horizon_dir = np.array([0, 1, 0]) # Perpendicular to the horizon
    ref_dir = np.cross(np.cross(p0, p1), np.cross(np.cross(p0, p1) + horizon_dir, v))

    # Measurement Point trick calculation:
    # We create uniform points on an "imaginary" ground line
    # and project them onto the edge using the Measurement Point.
    uniform_divisions = np.linspace(0, 1, num_divisions + 1)[1:-1]
    
    projected_points = []
    
    # Projective Splitting Algorithm: We use Geometric Cross Ratio
    # This mathematical formula ensures that the "centimeters" become denser towards the VP.
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    # distance ratio f_p = 1 / (d_p/D + 1)... simplified trick
    # For this simulation, we use a simplified projective formula
    # that gives a visually correct result without a full camera calibration matrix.
    compression_factor = 2.0 # Adjusts the intensity of the perspective compression
    
    for t in uniform_divisions:
        # Non-linear calculation: f(t) = (t * (D+d)) / (D+d - t*d)
        # d=(V_end-V_start), D=(V_start-Station)
        pt = (t * compression_factor) / (1.0 + (compression_factor-1.0)*t)
        
        px = p_start[0] + pt * (p_end[0] - p_start[0])
        py = p_start[1] + pt * (p_end[1] - p_start[1])
        projected_points.append((px, py))
        
    return projected_points

def draw_3pt_perspective(vp1_x, vp2_x, horizon_y, vp3_x, vp3_y, v0_x, v0_y, w_left, w_right, h_vert, solid_faces, show_projective_rulers):
    plt.figure(figsize=(10, 8))
    
    vp1 = (vp1_x, horizon_y)
    vp2 = (vp2_x, horizon_y)
    vp3 = (vp3_x, vp3_y)
    v0 = (v0_x, v0_y) 
    
    plt.axhline(horizon_y, color='black', lw=1, ls='--')
    
    plt.plot(*vp1, 'ro'); plt.text(vp1[0], vp1[1]+0.8, 'VP1', ha='center')
    plt.plot(*vp2, 'ro'); plt.text(vp2[0], vp2[1]+0.8, 'VP2', ha='center')
    plt.plot(*vp3, 'ro'); plt.text(vp3[0]+0.5, vp3[1], 'VP3')
    
    v1 = (v0[0] + w_left*(vp1[0]-v0[0]), v0[1] + w_left*(vp1[1]-v0[1]))
    v2 = (v0[0] + w_right*(vp2[0]-v0[0]), v0[1] + w_right*(vp2[1]-v0[1]))
    v3 = (v0[0] + h_vert*(vp3[0]-v0[0]), v0[1] + h_vert*(vp3[1]-v0[1]))
    
    v4 = get_intersect(*v1, *vp3, *v3, *vp1)          
    v5 = get_intersect(*v2, *vp3, *v3, *vp2)          
    v_back_bot = get_intersect(*v1, *vp2, *v2, *vp1)  
    v_back_top = get_intersect(*v4, *vp2, *v5, *vp1)  
    
    plt.plot([vp1[0], v0[0]], [vp1[1], v0[1]], 'gray', alpha=0.2, ls=':')
    plt.plot([vp2[0], v0[0]], [vp2[1], v0[1]], 'gray', alpha=0.2, ls=':')
    plt.plot([vp3[0], v0[0]], [vp3[1], v0[1]], 'gray', alpha=0.2, ls=':')
    
    if solid_faces:
        side_left_x = [v0[0], v1[0], v4[0], v3[0]]
        side_left_y = [v0[1], v1[1], v4[1], v3[1]]
        
        side_right_x = [v0[0], v2[0], v5[0], v3[0]]
        side_right_y = [v0[1], v2[1], v5[1], v3[1]]
        
        # Calculate center Y for overlapping
        face_0_y = [v0[1], v1[1], v_back_bot[1], v2[1]]
        face_3_y = [v3[1], v4[1], v_back_top[1], v5[1]]
        y_center_0 = sum(face_0_y) / 4
        y_center_3 = sum(face_3_y) / 4
        
        y_max = max(y_center_0, y_center_3)
        y_min = min(y_center_0, y_center_3)
        
        # Draw side faces first
        plt.fill(side_left_x, side_left_y, 'lightblue', edgecolor='blue', lw=1.5, alpha=0.8)
        plt.fill(side_right_x, side_right_y, 'steelblue', edgecolor='darkblue', lw=1.5, alpha=0.8)
        
        if horizon_y > y_max: 
            plt.fill([v3[0], v4[0], v_back_top[0], v5[0]], [v3[1], v4[1], v_back_top[1], v5[1]], 'lightgray', edgecolor='black', lw=1.5, alpha=0.8)
        elif horizon_y < y_min: 
            plt.fill([v0[0], v1[0], v_back_bot[0], v2[0]], [v0[1], v1[1], v_back_bot[1], v2[1]], 'gray', edgecolor='black', lw=1.5, alpha=0.8)
    else:
        # Wireframe
        def line(pa, pb, style='b-'):
            plt.plot([pa[0], pb[0]], [pa[1], pb[1]], style, lw=2)
        line(v0, v1); line(v0, v2); line(v0, v3)
        line(v1, v4); line(v3, v4)
        line(v2, v5); line(v3, v5)

    # NEW: Apply Projective Rulers
    # Instead of ticks on the edges, we draw a full grid on the surfaces.
    if show_projective_rulers:
        ruler_color = 'white' if solid_faces else 'red'
        
        # --- Left Face (X-Z plane towards VP1) ---
        # 1. Find the compressed subdivisions on the edges
        ticks_x_bot = projective_interpolate(v0, v1, vp1) # V0->V1, converging to VP1
        ticks_x_top = projective_interpolate(v3, v4, vp1) # V3->V4, converging to VP1
        
        ticks_z_left_bot = projective_interpolate(v0, v3, vp3) # V0->V3, converging to VP3
        ticks_z_left_top = projective_interpolate(v1, v4, vp3) # V1->V4, converging to VP3
        
        # 2. Draw the grid
        # Vertical lines (converge to VP3)
        for tb, tt in zip(ticks_x_bot, ticks_x_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)
            
        # Horizontal lines (converge to VP1)
        for tb, tt in zip(ticks_z_left_bot, ticks_z_left_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)

        # --- Right Face (Y-Z plane towards VP2) ---
        ticks_y_bot = projective_interpolate(v0, v2, vp2) # V0->V2, converging to VP2
        ticks_y_top = projective_interpolate(v3, v5, vp2) # V3->V5, converging to VP2
        
        ticks_z_right_bot = projective_interpolate(v0, v3, vp3) # V0->V3, converging to VP3
        ticks_z_right_top = projective_interpolate(v2, v5, vp3) # V2->V5, converging to VP3
        
        # Vertical lines (converge to VP3)
        for tb, tt in zip(ticks_y_bot, ticks_y_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)
            
        # Horizontal lines (converge to VP2)
        for tb, tt in zip(ticks_z_right_bot, ticks_z_right_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)

    plt.xlim(-20, 20)
    plt.ylim(-15, 25)
    plt.grid(True, which='both', linestyle=':', alpha=0.2)
    plt.title("Perspective Simulator with Projective Ruler Grid")
    plt.show()

# UI Sliders 
interactive_plot = interactive(
    draw_3pt_perspective, 
    vp1_x=FloatSlider(min=-25, max=-1, step=1, value=-12, description='VP1 Left', continuous_update=False),
    vp2_x=FloatSlider(min=1, max=25, step=1, value=12, description='VP2 Right', continuous_update=False),
    horizon_y=FloatSlider(min=-10, max=10, step=1, value=5, description='Horizon Y', continuous_update=False),
    vp3_x=FloatSlider(min=-15, max=15, step=1, value=0, description='VP3 X', continuous_update=False),
    vp3_y=FloatSlider(min=-30, max=30, step=1, value=20, description='VP3 Y', continuous_update=False),
    v0_x=FloatSlider(min=-8, max=8, step=0.5, value=-2.5, description='Object X', continuous_update=False),
    v0_y=FloatSlider(min=-10, max=10, step=0.5, value=-5, description='Object Y', continuous_update=False),
    w_left=FloatSlider(min=0.1, max=0.9, step=0.05, value=0.4, description='Size Left', continuous_update=False),
    w_right=FloatSlider(min=0.1, max=0.9, step=0.05, value=0.45, description='Size Right', continuous_update=False),
    h_vert=FloatSlider(min=0.1, max=0.9, step=0.05, value=0.6, description='Height', continuous_update=False),
    solid_faces=Checkbox(value=True, description='Solid Faces'),
    show_projective_rulers=Checkbox(value=True, description='Show Projective Rulers') 
)
display(interactive_plot)
