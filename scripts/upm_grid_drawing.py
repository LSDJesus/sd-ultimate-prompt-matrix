import math
from PIL import Image, ImageDraw, ImageFont

def get_font(fontsize):
    try: return ImageFont.truetype("dejavu.ttf", fontsize)
    except IOError:
        try: return ImageFont.truetype("arial.ttf", fontsize)
        except IOError: return ImageFont.load_default()

def draw_grid_with_annotations(grid_images, x_labels, y_labels, margin_size, title="", show_annotations=True):
    if not grid_images or not any(grid_images) or not isinstance(grid_images[0], Image.Image): return None
    num_cols = len(x_labels) if x_labels else math.ceil(math.sqrt(len(grid_images)))
    if num_cols == 0: return None
    num_rows = math.ceil(len(grid_images) / num_cols)
    if num_rows == 0: return None
    img_w, img_h = grid_images[0].size
    label_font = get_font(30); title_font = get_font(36)
    y_label_w = (max(label_font.getbbox(label)[2] for label in y_labels) + margin_size * 2) if y_labels and show_annotations else 0
    x_label_h = (label_font.getbbox("Tg")[3] + margin_size * 2) if x_labels and show_annotations else 0
    title_h = (title_font.getbbox("Tg")[3] + margin_size * 2) if title and show_annotations else 0
    grid_w = y_label_w + (num_cols * img_w) + (margin_size * (num_cols - 1))
    grid_h = title_h + x_label_h + (num_rows * img_h) + (margin_size * (num_rows - 1))
    grid_image = Image.new('RGB', (int(grid_w), int(grid_h)), color='white')
    draw = ImageDraw.Draw(grid_image)
    if title and show_annotations: draw.text((grid_w / 2, margin_size), title, font=title_font, fill='black', anchor="mt")
    if x_labels and show_annotations:
        for i, label in enumerate(x_labels):
            x_pos = y_label_w + (i * (img_w + margin_size)) + (img_w / 2); y_pos = title_h + margin_size
            draw.text((x_pos, y_pos), label, font=label_font, fill='black', anchor="mt")
    if y_labels and show_annotations:
        for i, label in enumerate(y_labels):
            x_pos = margin_size; y_pos = title_h + x_label_h + (i * (img_h + margin_size)) + (img_h / 2)
            draw.text((x_pos, y_pos), label, font=label_font, fill='black', anchor="lm")
    for i, img in enumerate(grid_images):
        col, row = i % num_cols, i // num_cols
        paste_x = y_label_w + col * (img_w + margin_size); paste_y = title_h + x_label_h + row * (img_h + margin_size)
        grid_image.paste(img, (int(paste_x), int(paste_y)))
    return grid_image

def create_mega_grid(all_grids, page_labels, margin_size, show_annotations=True):
    valid_grids = [g for g in all_grids if g is not None]
    if not valid_grids or len(valid_grids) <= 1: return None
    mega_cols = math.ceil(math.sqrt(len(valid_grids))); mega_rows = math.ceil(len(valid_grids) / mega_cols)
    grid_w, grid_h = valid_grids[0].size
    font = get_font(36); title_h = 50 if show_annotations else 0
    mega_w = mega_cols * grid_w + margin_size * (mega_cols + 1)
    mega_h = mega_rows * (grid_h + title_h) + margin_size * (mega_rows + 1)
    mega_image = Image.new('RGB', (int(mega_w), int(mega_h)), color='#DDDDDD')
    draw = ImageDraw.Draw(mega_image)
    for i, grid in enumerate(valid_grids):
        col, row = i % mega_cols, i // mega_cols
        cell_x = margin_size + col * (grid_w + margin_size); cell_y = margin_size + row * (grid_h + title_h + margin_size)
        if show_annotations: draw.text((cell_x + grid_w / 2, cell_y + title_h / 2), page_labels[i], font=font, fill='black', anchor="mm")
        mega_image.paste(grid, (int(cell_x), int(cell_y + title_h)))
    return mega_image