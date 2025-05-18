import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from ipywidgets import interact, IntSlider, Play, HBox, jslink, VBox

# Set dark theme style
plt.style.use('dark_background')

# Set up the triangle
center = np.array([2, 2])
side_length = 2
height = side_length * np.sqrt(3) / 2
vertices = np.array([
    [center[0] - side_length/2, center[1] - height/3],
    [center[0] + side_length/2, center[1] - height/3],
    [center[0], center[1] + 2*height/3]
])

def rotate_point(point, center, angle):
    x, y = point
    cx, cy = center
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    new_x = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
    new_y = cy + (x - cx) * sin_theta + (y - cy) * cos_theta
    return np.array([new_x, new_y])

def plot_rotated_triangle(frame):
    angle = frame * (2 * np.pi / 20)  # 20 steps for a full rotation
    rotated_vertices = np.array([rotate_point(v, center, angle) for v in vertices])
    
    # Create the plot with dark theme
    plt.figure(figsize=(10, 10), facecolor='#1a1a1a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')
    
    # Plot the triangle with glowing effect
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
    
    # Regression line with glowing effect
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
    plt.title('Rotating Triangle with Regression Line',
             color='#ffffff',
             fontsize=14,
             pad=20,
             fontweight='bold')
    
    # Add subtle background circle
    circle = plt.Circle(center, max(side_length, height), fill=False, 
                       color='#404040', alpha=0.2, linestyle='--')
    ax.add_artist(circle)
    
    plt.show()

# Interactive widgets with dark theme
play = Play(value=0, min=0, max=19, step=1, interval=200, description="Press play", disabled=False)
slider = IntSlider(value=0, min=0, max=19, step=1, description='Frame')
jslink((play, 'value'), (slider, 'value'))
ui = HBox([play, slider])
interact(plot_rotated_triangle, frame=slider)
display(ui) 