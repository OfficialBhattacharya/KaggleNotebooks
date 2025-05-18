import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Play, HBox, jslink, VBox
import matplotlib.patheffects as patheffects

# Set dark theme style
plt.style.use('dark_background')

def create_regular_polygon(n_sides, radius, center):
    """Create vertices of a regular polygon with n sides."""
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    vertices = np.array([
        [center[0] + radius * np.cos(angle),
         center[1] + radius * np.sin(angle)]
        for angle in angles
    ])
    return vertices

def rotate_point(point, center, angle):
    """Rotate a point around a center by given angle."""
    x, y = point
    cx, cy = center
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    new_x = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
    new_y = cy + (x - cx) * sin_theta + (y - cy) * cos_theta
    return np.array([new_x, new_y])

def plot_rotated_polygon(frame, n_sides=6):
    """Plot the rotating polygon with regression line."""
    # Set up the polygon
    center = np.array([2, 2])
    radius = 1.5
    vertices = create_regular_polygon(n_sides, radius, center)
    
    # Calculate rotation angle
    angle = frame * (2 * np.pi / 20)  # 20 steps for a full rotation
    
    # Rotate vertices
    rotated_vertices = np.array([rotate_point(v, center, angle) for v in vertices])
    
    # Create the plot with dark theme
    plt.figure(figsize=(10, 10), facecolor='#1a1a1a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')
    
    # Plot the polygon with glowing effect
    plt.plot(
        np.append(rotated_vertices[:, 0], rotated_vertices[0, 0]),
        np.append(rotated_vertices[:, 1], rotated_vertices[0, 1]),
        color='#00ff9f',  # Bright cyan color
        lw=3,
        alpha=0.8,
        path_effects=[patheffects.withStroke(linewidth=5, foreground='#00ff9f', alpha=0.2)]
    )
    
    # Plot vertices with glowing points
    plt.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], 
               color='#00ff9f', s=100, alpha=0.8,
               edgecolor='white', linewidth=1)
    
    # Plot the regression line with glowing effect
    x = rotated_vertices[:, 0]
    y = rotated_vertices[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([min(x), max(x)])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='#ff3366',  # Bright pink color
            lw=2, alpha=0.8,
            path_effects=[patheffects.withStroke(linewidth=4, foreground='#ff3366', alpha=0.2)])
    
    # Set plot properties with futuristic styling
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    ax.set_aspect('equal')
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3, color='#404040')
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1)
    
    # Add title with custom styling
    plt.title(f'Rotating {n_sides}-sided Polygon with Regression Line',
             color='#ffffff',
             fontsize=14,
             pad=20,
             fontweight='bold')
    
    # Add subtle background circle
    circle = plt.Circle(center, radius, fill=False, 
                       color='#404040', alpha=0.2, linestyle='--')
    ax.add_artist(circle)
    
    plt.show()

# Interactive widgets with dark theme
play = Play(value=0, min=0, max=19, step=1, interval=200, description="Press play", disabled=False)
slider = IntSlider(value=0, min=0, max=19, step=1, description='Frame')
jslink((play, 'value'), (slider, 'value'))
ui = HBox([play, slider])
out = interact(plot_rotated_polygon, frame=slider)
display(VBox([out, ui])) 