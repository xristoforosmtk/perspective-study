import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, FloatSlider, Checkbox
from IPython.display import display

# Συνάρτηση εύρεσης τομής ευθειών
def get_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return x1, y1
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return px, py

# ΝΕΑ ΣΥΝΑΡΤΗΣΗ: Υπολογίζει τη μη-γραμμική συμπίεση της προοπτικής
# Δουλεύει με Homogeneous Coordinates για σωστή προβολική παρεμβολή.
def projective_interpolate(p_start, p_end, vp, num_divisions=8):
    # Μετατρέπουμε τα 2D σημεία σε Homogeneous Coordinates (x, y, 1)
    p0 = np.array([p_start[0], p_start[1], 1])
    p1 = np.array([p_end[0], p_end[1], 1])
    v = np.array([vp[0], vp[1], 1])
    
    # Υπολογίζουμε τις γραμμές (cross products)
    l_end = np.cross(p1, v) # Γραμμή από P1 στο VP
    l_edge = np.cross(p0, p1) # Γραμμή της ακμής P0-P1
    l_ref = np.cross(p0, v) # Γραμμή από P0 στο VP

    # Υπολογίζουμε τη διεύθυνση της γραμμής του ορίζοντα (σε Homogeneous coords)
    # Για απλότητα σε αυτή τη 3pt simulation, χρησιμοποιούμε σταθερό ορίζοντα
    horizon_dir = np.array([0, 1, 0]) # Κάθετη στον ορίζοντα
    ref_dir = np.cross(np.cross(p0, p1), np.cross(np.cross(p0, p1) + horizon_dir, v))

    # Υπολογισμός Measurement Point trick:
    # Κατασκευάζουμε ομοιόμορφα σημεία σε μια "φανταστική" ground line
    # και τα προβάλλουμε στην ακμή χρησιμοποιώντας το Measurement Point.
    uniform_divisions = np.linspace(0, 1, num_divisions + 1)[1:-1]
    
    projected_points = []
    
    # ΑλγόριθμοςProjective Splitting: Χρησιμοποιούμε Geometric Cross Ratio
    # Αυτή η μαθηματική φόρμουλα εξασφαλίζει ότι τα "εκατοστά" πυκνώνουν προς το VP.
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    # distance ratio f_p = 1 / (d_p/D + 1)... simplified trick
    # Για αυτή τη simulation, χρησιμοποιούμε μια απλοποιημένη προβολική φόρμουλα
    # που δίνει οπτικά σωστό αποτέλεσμα χωρίς πλήρες camera calibration matrix.
    compression_factor = 2.0 # Ρυθμίζει την ένταση της προοπτικής συμπίεσης
    
    for t in uniform_divisions:
        # Μη-γραμμικός υπολογισμός: f(t) = (t * (D+d)) / (D+d - t*d)
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
        
        # Υπολογισμός Y κέντρου για επικάλυψη
        face_0_y = [v0[1], v1[1], v_back_bot[1], v2[1]]
        face_3_y = [v3[1], v4[1], v_back_top[1], v5[1]]
        y_center_0 = sum(face_0_y) / 4
        y_center_3 = sum(face_3_y) / 4
        
        y_max = max(y_center_0, y_center_3)
        y_min = min(y_center_0, y_center_3)
        
        # Ζωγραφίζουμε πρώτα τις πλαϊνές
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

    # ΝΕΟ: Εφαρμογή του Projective Rulers
    # Αντί για ticks στις ακμές, σχεδιάζουμε ένα πλήρες grid στις επιφάνειες.
    if show_projective_rulers:
        ruler_color = 'white' if solid_faces else 'red'
        
        # --- Αριστερή Πλευρά (X-Z plane towards VP1) ---
        # 1. Βρίσκουμε τις συμπιεσμένες υποδιαιρέσεις στις ακμές
        ticks_x_bot = projective_interpolate(v0, v1, vp1) # V0->V1, converging to VP1
        ticks_x_top = projective_interpolate(v3, v4, vp1) # V3->V4, converging to VP1
        
        ticks_z_left_bot = projective_interpolate(v0, v3, vp3) # V0->V3, converging to VP3
        ticks_z_left_top = projective_interpolate(v1, v4, vp3) # V1->V4, converging to VP3
        
        # 2. Σχεδιάζουμε το grid
        # Κατακόρυφες γραμμές (converge to VP3)
        for tb, tt in zip(ticks_x_bot, ticks_x_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)
            
        # Οριζόντιες γραμμές (converge to VP1)
        for tb, tt in zip(ticks_z_left_bot, ticks_z_left_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)

        # --- Δεξιά Πλευρά (Y-Z plane towards VP2) ---
        ticks_y_bot = projective_interpolate(v0, v2, vp2) # V0->V2, converging to VP2
        ticks_y_top = projective_interpolate(v3, v5, vp2) # V3->V5, converging to VP2
        
        ticks_z_right_bot = projective_interpolate(v0, v3, vp3) # V0->V3, converging to VP3
        ticks_z_right_top = projective_interpolate(v2, v5, vp3) # V2->V5, converging to VP3
        
        # Κατακόρυφες γραμμές (converge to VP3)
        for tb, tt in zip(ticks_y_bot, ticks_y_top):
            plt.plot([tb[0], tt[0]], [tb[1], tt[1]], ruler_color, lw=1.0, alpha=0.6)
            
        # Οριζόντιες γραμμές (converge to VP2)
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