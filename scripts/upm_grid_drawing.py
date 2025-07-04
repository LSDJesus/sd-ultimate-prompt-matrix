import math
from PIL import Image, ImageDraw, ImageFont

def get_font(fontsize):
    try: return ImageFont.truetype("dejavu.ttf", fontsize)
    except IOError:
        try: return ImageFont.truetype("arial.ttf", fontsize)
        except IOError: return ImageFont.load_default()

def draw_preview_grid(x_labels, y_labels, page_labels):
    """Generates a simple black and white PIL image to visually represent the grid structure."""
    cell_size = 64
    margin = 8
    font = get_font(24)

    # Determine dimensions
    num_cols = len(x_labels) if x_labels else 1
    num_rows = len(y_labels) if y_labels else 1
    
    # Calculate text sizes to determine overall image dimensions
    x_label_max_w = max(font.getbbox(label)[2] for label in x_labels) if x_labels else 0
    y_label_max_w = max(font.getbbox(label)[2] for label in y_labels) if y_labels else 0
    
    # Define widths and heights for different sections
    header_h = font.getbbox("Tg")[3] + margin * 2  # For X-axis labels
    y_axis_w = y_label_max_w + margin * 2
    
    title_text = " | ".join(page_labels) if page_labels else "Single Grid"
    title_h = font.getbbox(title_text)[3] + margin * 2 if title_text else 0

    grid_w = num_cols * cell_size
    grid_h = num_rows * cell_size

    total_w = y_axis_w + grid_w
    total_h = title_h + header_h + grid_h

    # Create image and draw object
    img = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(img)

    # Draw Title (Page Axes)
    if title_text:
        draw.text((total_w / 2, margin), title_text, font=font, fill='black', anchor='mt')

    # Draw X-axis labels (headers)
    for i, label in enumerate(x_labels):
        x = y_axis_w + (i * cell_size) + (cell_size / 2)
        y = title_h + margin
        draw.text((x, y), str(label), font=font, fill='black', anchor='mt')
    
    # Draw Y-axis labels
    for i, label in enumerate(y_labels):
        x = margin
        y = title_h + header_h + (i * cell_size) + (cell_size / 2)
        draw.text((x, y), str(label), font=font, fill='black', anchor='lm')

    # Draw grid lines
    for i in range(num_cols + 1):
        x0 = y_axis_w + i * cell_size
        y0 = title_h + header_h
        x1 = x0
        y1 = total_h
        draw.line([(x0, y0), (x1, y1)], fill='black', width=1)
        
    for i in range(num_rows + 1):
        x0 = y_axis_w
        y0 = title_h + header_h + i * cell_size
        x1 = total_w
        y1 = y0
        draw.line([(x0, y0), (x1, y1)], fill='black', width=1)

    return img

def draw_grid_with_annotations(grid_images, x_labels, y_labels, margin_size, title="", show_annotations=True):
    if not grid_images or not any(grid_images) or not isinstance(grid_images[0], Image.Image): return None
    num_cols = len(x_labels) if x_labels else math.ceil(math.sqrt(len(grid_images)))