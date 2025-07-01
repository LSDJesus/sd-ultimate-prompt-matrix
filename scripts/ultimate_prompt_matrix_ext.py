"""
Ultimate Prompt Matrix Extension v5.1 (LoRA Hotfix) for AUTOMATIC1111 & Forge

This version fixes a critical timing issue where the LoRA dropdowns would not populate
on startup. A "Refresh LoRAs" button has been added to load the list on-demand,
ensuring compatibility and reliability across all Web UI versions.
"""

import math
import re
import copy
import os
import random
from PIL import Image, ImageDraw, ImageFont
from itertools import product

import modules.scripts as scripts
import gradio as gr
from modules import images, shared, processing, sd_samplers, sd_schedulers, sd_models
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts

try:
    from modules.shared import last_info
    LAST_INFO_AVAILABLE = True
except ImportError:
    last_info = ""
    LAST_INFO_AVAILABLE = False

# --- Constants and Regex ---
MAX_LORA_ROWS = 5
REX_MATRIX = re.compile(r'(<(?!lora:)([^>]+)>)')
REX_RANDOM = re.compile(r'<random\(([^)]+)\)>')

# --- Helper Functions (Unchanged, but get_lora_names is now called differently) ---
def get_lora_names():
    """Returns a list of all available LoRA model names."""
    try:
        # Re-scan the loras folder every time this is called to ensure it's up to date
        sd_models.refresh_loras()
        return [lora.name for lora in sd_models.loras]
    except Exception as e:
        print(f"[Ultimate Matrix] Error getting LoRA names: {e}")
        return ["None"]

def get_font(fontsize):
    try: return ImageFont.truetype("dejavu.ttf", fontsize)
    except IOError:
        try: return ImageFont.truetype("arial.ttf", fontsize)
        except IOError: return ImageFont.load_default()

# --- All other helper functions (draw_grid, create_mega_grid, etc.) are unchanged ---
def draw_grid_with_annotations(grid_images, x_labels, y_labels, margin_size, title="", show_annotations=True):
    if not grid_images or not any(grid_images): return None
    num_cols = len(x_labels) if x_labels else math.ceil(math.sqrt(len(grid_images)))
    num_rows = math.ceil(len(grid_images) / num_cols)
    if num_rows == 0 or num_cols == 0: return None
    img_w, img_h = grid_images[0].size
    label_font, title_font = get_font(30), get_font(36)
    y_label_width = (max(label_font.getbbox(label)[2] for label in y_labels) + margin_size * 2) if y_labels and show_annotations else 0
    x_label_height = (label_font.getbbox("Tg")[3] + margin_size * 2) if x_labels and show_annotations else 0
    title_height = (title_font.getbbox("Tg")[3] + margin_size * 2) if title and show_annotations else 0
    grid_w = y_label_width + (num_cols * img_w) + (margin_size * (num_cols - 1))
    grid_h = title_height + x_label_height + (num_rows * img_h) + (margin_size * (num_rows - 1))
    grid_image = Image.new('RGB', (int(grid_w), int(grid_h)), color='white')
    draw = ImageDraw.Draw(grid_image)
    if title and show_annotations: draw.text((grid_w / 2, margin_size), title, font=title_font, fill='black', anchor="mt")
    if x_labels and show_annotations:
        for i, label in enumerate(x_labels):
            x, y = y_label_width + (i * (img_w + margin_size)) + (img_w / 2), title_height + margin_size
            draw.text((x, y), label, font=label_font, fill='black', anchor="mt")
    if y_labels and show_annotations:
        for i, label in enumerate(y_labels):
            x, y = margin_size, title_height + x_label_height + (i * (img_h + margin_size)) + (img_h / 2)
            draw.text((x, y), label, font=label_font, fill='black', anchor="lm")
    for i, img in enumerate(grid_images):
        col, row = i % num_cols, i // num_cols
        x, y = y_label_width + col * (img_w + margin_size), title_height + x_label_height + row * (img_h + margin_size)
        grid_image.paste(img, (int(x), int(y)))
    return grid_image

def create_mega_grid(all_grids, page_labels, margin_size, show_annotations=True):
    if not all_grids or len(all_grids) <= 1: return None
    mega_cols = math.ceil(math.sqrt(len(all_grids)))
    mega_rows = math.ceil(len(all_grids) / mega_cols)
    grid_w, grid_h = all_grids[0].size
    font, title_height = get_font(36), 50 if show_annotations else 0
    mega_w = mega_cols * grid_w + margin_size * (mega_cols + 1)
    mega_h = mega_rows * (grid_h + title_height) + margin_size * (mega_rows + 1)
    mega_image = Image.new('RGB', (int(mega_w), int(mega_h)), color='#DDDDDD')
    draw = ImageDraw.Draw( mega_image)
    for i, grid in enumerate(all_grids):
        col, row = i % mega_cols, i // mega_cols
        cell_x, cell_y = margin_size + col * (grid_w + margin_size), margin_size + row * (grid_h + title_height + margin_size)
        if show_annotations:
            draw.text((cell_x + grid_w / 2, cell_y + title_height / 2), page_labels[i], font=font, fill='black', anchor="mm")
        mega_image.paste(grid, (int(cell_x), int(cell_y + title_height)))
    return mega_image

def sanitize_filename(text):
    if not text: return ""
    return re.sub(r'[\\/*?:"<>|]', '', text)[:100]

def paste_last_prompts():
    if hasattr(shared, 'last_info') and shared.last_info and shared.last_info != "":
        info_text = shared.last_info
        neg_prompt_match = re.search(r'Negative prompt: (.+?)(?=\nSteps:)', info_text, re.DOTALL)
        neg_prompt = neg_prompt_match.group(1).strip() if neg_prompt_match else ""
        pos_prompt = info_text.split('Negative prompt:')[0].strip()
        return pos_prompt, neg_prompt
    return "Could not find last generation info.", ""

# --- Main Logic Function ---
def run_matrix_processing(*args):
    # This now uses a generator `yield` to give live feedback
    all_args = list(args)
    # The LoRA components are at the end, we don't need them here, just the main settings
    (
        prompt, negative_prompt, sampler_name, scheduler, steps, cfg_scale, width, height,
        matrix_mode, prompt_type, different_seeds, margin_size, create_mega_grid_toggle,
        margin_toggle, dry_run, save_prompt_list, use_descriptive_filenames,
        show_annotations, enable_dynamic_prompts
    ) = all_args[:19]

    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model, outpath_samples=opts.outdir_txt2img_samples, outpath_grids=opts.outdir_txt2img_grids,
        prompt=prompt, negative_prompt=negative_prompt, seed=-1, sampler_name=sampler_name, scheduler=scheduler,
        steps=steps, cfg_scale=cfg_scale, width=width, height=height, batch_size=1, n_iter=1,
    )
    # The rest of this function is unchanged
    processing.fix_seed(p)
    is_permutation_mode = (matrix_mode == "Permutation")
    prompt_type_lower = prompt_type.lower()
    if prompt_type_lower == "positive" or prompt_type_lower == "both": master_prompt_text_unresolved = p.prompt
    else: master_prompt_text_unresolved = p.negative_prompt
    master_prompt_text = re.sub(REX_RANDOM, lambda m: random.choice(m.group(1).split('|')).strip(), master_prompt_text_unresolved)
    matches = list(REX_MATRIX.finditer(master_prompt_text))
    if not matches:
        if re.search(REX_RANDOM, master_prompt_text_unresolved): 
            p.prompt = master_prompt_text
            processed = process_images(p)
            yield { image_state: [processed.images], image_display: processed.images[0], html_info: processed.infotexts[0], html_log: "", image_slider: gr.Slider.update(visible=True, maximum=1, value=1) }
            return
        else: 
            yield { html_log: "No matrix syntax found in prompt." }
            return
    dynamic_prompts_active = False
    if enable_dynamic_prompts:
        try: from sd_dynamic_prompts.prompt_parser import parse; dynamic_prompts_active = True; print("Dynamic Prompts extension found and enabled.")
        except ImportError: print("WARNING: Dynamic Prompts extension not found. __wildcard__ syntax will be ignored.")
    all_prompts_data = []
    if is_permutation_mode:
        x_axis_match = matches.pop() if matches else None; y_axis_match = matches.pop() if matches else None; page_matches = matches
        x_options = [opt.strip() for opt in x_axis_match.group(2).split("|")] if x_axis_match else [""]
        y_options = [opt.strip() for opt in y_axis_match.group(2).split("|")] if y_axis_match else [""]
        page_options_list = [[opt.strip() for opt in m.group(2).split("|")] for m in page_matches]
        page_combinations = list(product(*page_options_list)) if page_options_list else [()]
        for page_values in page_combinations:
            for y_val in y_options:
                for x_val in x_options: all_prompts_data.append({'x': x_val, 'y': y_val, 'page': page_values})
    else:
        base_prompt = ', '.join(filter(None, [x.strip() for x in re.sub(REX_MATRIX, '', master_prompt_text).split(',')]))
        optional_tags = [m.group(2).strip() for m in matches]
        for i in range(2**len(optional_tags)):
            selected_tags = [optional_tags[j] for j in range(len(optional_tags)) if (i >> j) & 1]
            all_prompts_data.append(", ".join(filter(None, [base_prompt] + selected_tags)))
    final_prompts_list = []
    for i, item in enumerate(all_prompts_data):
        if is_permutation_mode:
            temp_prompt = master_prompt_text
            for j, match in enumerate(page_matches): temp_prompt = temp_prompt.replace(match.group(1), item['page'][j], 1)
            if y_axis_match: temp_prompt = temp_prompt.replace(y_axis_match.group(1), item['y'], 1)
            if x_axis_match: temp_prompt = temp_prompt.replace(x_axis_match.group(1), item['x'], 1)
        else: temp_prompt = item
        final_prompts_list.append(temp_prompt)
    if dry_run:
        print(f"--- DRY RUN: {len(final_prompts_list)} prompts generated. ---"); [print(f"{i+1:03d}: {prompt}") for i, prompt in enumerate(final_prompts_list)]
        yield { html_log: "Dry run complete. No images generated.", image_slider: gr.Slider.update(visible=False) }
        return
    shared.state.job_count = len(final_prompts_list)
    final_margin_size = margin_size if margin_toggle else 0
    all_generated_images, all_infotexts = [], []
    original_seed = p.seed
    for i, prompt_text in enumerate(final_prompts_list):
        if shared.state.interrupted: break
        shared.state.job = f"Image {i+1}/{shared.state.job_count}"; p_copy = copy.copy(p); p_copy.n_iter = 1; p_copy.batch_size = 1
        if prompt_type_lower == "positive": p_copy.prompt = prompt_text
        elif prompt_type_lower == "negative": p_copy.negative_prompt = prompt_text
        elif prompt_type_lower == "both":
            p_copy.prompt = prompt_text; temp_neg = p.negative_prompt
            if is_permutation_mode:
                data = all_prompts_data[i]
                for j, match in enumerate(page_matches): temp_neg = temp_neg.replace(match.group(1), data['page'][j], 1)
                if y_axis_match: temp_neg = temp_neg.replace(y_axis_match.group(1), data['y'], 1)
                if x_axis_match: temp_neg = temp_neg.replace(x_axis_match.group(1), data['x'], 1)
            else:
                base_prompt = ', '.join(filter(None, [x.strip() for x in re.sub(REX_MATRIX, '', master_prompt_text).split(',')]))
                added_tags = prompt_text.replace(base_prompt, "").strip(", "); temp_neg = ", ".join(filter(None, [p.negative_prompt, added_tags]))
            p_copy.negative_prompt = temp_neg
        if dynamic_prompts_active: p_copy.prompt = parse(p_copy.prompt)
        p_copy.seed = original_seed + i if different_seeds else original_seed
        processed_single = process_images(p_copy)
        all_generated_images.append(processed_single.images[0])
        all_infotexts.append(processed_single.infotexts[0])
        yield { image_state: all_generated_images, image_display: processed_single.images[0], html_info: processed_single.infotexts[0], html_log: f"Generated {len(all_generated_images)} of {len(final_prompts_list)} images.", image_slider: gr.Slider.update(visible=True, maximum=len(all_generated_images), value=len(all_generated_images)) }
    if shared.state.interrupted: print("Matrix generation interrupted by user.")
    if not all_generated_images: return
    all_grid_images, page_labels = [], []
    if is_permutation_mode:
        for page_idx, page_values in enumerate(page_combinations):
            start_index = page_idx * len(y_options) * len(x_options); end_index = start_index + len(y_options) * len(x_options)
            images_for_this_grid = all_generated_images[start_index:end_index]
            page_combination_text = ", ".join(page_values); full_page_title = ""
            if len(page_combinations) > 1: full_page_title = f"Page {page_idx + 1}/{len(page_combinations)}: {page_combination_text}"
            elif page_combination_text: full_page_title = f"Page: {page_combination_text}"
            page_labels.append(full_page_title)
            grid_image = draw_grid_with_annotations(images_for_this_grid, x_options, y_options, final_margin_size, title=full_page_title, show_annotations=show_annotations)
            if grid_image:
                all_grid_images.append(grid_image)
                if opts.grid_save:
                    filename_part = sanitize_filename(page_combination_text) if use_descriptive_filenames else ""
                    images.save_image(grid_image, p.outpath_grids, f"matrix_{filename_part}", prompt=master_prompt_text_unresolved, seed=p.seed, grid=True, p=p)
    else:
        grid_image = draw_grid_with_annotations(all_generated_images, [], [], final_margin_size, show_annotations=show_annotations)
        if grid_image: all_grid_images.append(grid_image)
    mega_grid_image = None
    if is_permutation_mode and create_mega_grid_toggle and len(all_grid_images) > 1:
        mega_grid_image = create_mega_grid(all_grid_images, page_labels, final_margin_size, show_annotations=show_annotations)
        if mega_grid_image and opts.grid_save:
            filename_part = sanitize_filename(master_prompt_text_unresolved) if use_descriptive_filenames else ""
            images.save_image(mega_grid_image, p.outpath_grids, f"mega-matrix_{filename_part}", prompt=master_prompt_text_unresolved, seed=p.seed, grid=True, p=p)
    if save_prompt_list:
        log_prompts = [];
        for i, item in enumerate(all_prompts_data):
            if is_permutation_mode:
                temp_prompt = master_prompt_text_unresolved
                for j, match in enumerate(page_matches): temp_prompt = temp_prompt.replace(match.group(1), item['page'][j], 1)
                if y_axis_match: temp_prompt = temp_prompt.replace(y_axis_match.group(1), item['y'], 1)
                if x_axis_match: temp_prompt = temp_prompt.replace(x_axis_match.group(1), item['x'], 1)
                log_prompts.append(temp_prompt)
            else: log_prompts.append(item)
        prompt_log = f"--- Ultimate Prompt Matrix Log ---\nMaster Prompt: {master_prompt_text_unresolved}\n\n--- Generated Prompts ---\n" + "\n".join(f"{i+1:03d}: {p}" for i, p in enumerate(log_prompts))
        filename = os.path.join(p.outpath_grids or p.outpath_samples, "prompt_log.txt")
        with open(filename, "w", encoding="utf-8") as f: f.write(prompt_log)
        print(f"Saved prompt list to {filename}")
    final_images = []
    if mega_grid_image: final_images.append(mega_grid_image)
    final_images.extend(all_grid_images); final_images.extend(all_generated_images)
    yield { image_state: final_images, image_slider: gr.Slider.update(maximum=len(final_images), value=1), image_display: final_images[0] }

# --- UI Functions ---
def update_image_display(slider_value, image_list):
    if image_list and 0 < slider_value <= len(image_list):
        return image_list[int(slider_value) - 1]
    return None

def add_lora_row(current_count):
    current_count += 1
    updates = {lora_row_count: current_count}
    for i in range(MAX_LORA_ROWS):
        updates[lora_rows[i]] = gr.Row.update(visible=(i < current_count))
    return updates

# --- THE FIX IS HERE ---
def update_lora_dropdowns():
    """Gets the fresh list of LoRAs and returns updates for all dropdowns."""
    lora_names = get_lora_names()
    updates = []
    for _ in range(MAX_LORA_ROWS):
        updates.append(gr.Dropdown.update(choices=lora_names))
    return updates

def on_ui_tabs():
    global lora_rows, lora_row_count # Make these globally accessible for the helper
    
    with gr.Blocks(analytics_enabled=False) as ui_component:
        gr.Markdown("# Ultimate Prompt Matrix")
        gr.Markdown("A standalone tool for generating complex image grids using permutation, combination, or random syntax.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("### Prompts")
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt with matrix syntax here...", elem_id="matrix-prompt")
                    with gr.Column(min_width=40, scale=0):
                        clear_prompt_btn = gr.Button("🗑️", elem_classes=["tool"])
                        paste_prompts_btn = gr.Button("↙️", elem_classes=["tool"], tooltip="Paste prompts from last generation job")
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=3, placeholder="Enter negative prompts here...", elem_id="matrix-neg-prompt")
                    with gr.Column(min_width=40, scale=0):
                        clear_neg_prompt_btn = gr.Button("🗑️", elem_classes=["tool"])
                
                with gr.Accordion("Generation Settings", open=True):
                    with gr.Row():
                        sampler_name = gr.Dropdown(label='Sampling method', choices=[s.name for s in sd_samplers.samplers], value='Euler a')
                        scheduler = gr.Dropdown(label='Schedule type', choices=[s.label for s in sd_schedulers.schedulers], value='Automatic')
                    with gr.Row():
                        steps = gr.Slider(label='Sampling steps', minimum=1, maximum=150, value=20, step=1)
                        cfg_scale = gr.Slider(label='CFG Scale', minimum=1.0, maximum=30.0, value=7.0, step=0.5)
                    with gr.Row():
                        width = gr.Slider(label='Width', minimum=64, maximum=2048, value=512, step=64)
                        height = gr.Slider(label='Height', minimum=64, maximum=2048, value=512, step=64)

                with gr.Accordion("LoRA Matrix Builder", open=False):
                    with gr.Row():
                        gr.Markdown("Click Refresh to load your LoRA models.")
                        refresh_loras_btn = gr.Button("🔃 Refresh LoRAs", elem_classes="tool")
                    lora_rows, lora_dropdowns, lora_weights = [], [], []
                    for i in range(MAX_LORA_ROWS):
                        with gr.Row(visible=(i==0), elem_classes="lora-row") as row:
                            # Start with an empty list of choices
                            dropdown = gr.Dropdown([], multiselect=True, label=f"LoRA Block {i+1}")
                            weight = gr.Slider(minimum=-2.0, maximum=2.0, value=1.0, step=0.05, label="Weight")
                            lora_rows.append(row); lora_dropdowns.append(dropdown); lora_weights.append(weight)
                    with gr.Row():
                        add_lora_btn = gr.Button("[+] Add LoRA Block")
                        insert_loras_btn = gr.Button("Insert LoRAs into Prompt", variant="primary")
                        lora_row_count = gr.State(1)

            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### Matrix Core Settings")
                matrix_mode = gr.Radio(["Permutation", "Combination"], label="Matrix Mode", value="Permutation")
                prompt_type = gr.Radio(["Positive", "Negative", "Both"], label="Matrix prompt target", value="Positive")
                different_seeds = gr.Checkbox(label='Use different seed for each image', value=False)
                gr.Markdown("### Output & Grid Control")
                show_annotations = gr.Checkbox(label="Show grid labels & titles", value=True)
                margin_toggle = gr.Checkbox(label="Enable Grid Margins", value=True)
                margin_size = gr.Slider(label="Grid Margins (px)", minimum=0, maximum=500, value=10, step=2, visible=True)
                create_mega_grid_toggle = gr.Checkbox(label="Create a 'Mega-Grid' (Permutation mode)", value=True, visible=True)
                use_descriptive_filenames = gr.Checkbox(label="Use descriptive filenames for grids", value=False)
                save_prompt_list = gr.Checkbox(label="Save prompt list to a text file", value=False)

        with gr.Accordion("Advanced Features", open=False):
            dry_run = gr.Checkbox(label="Dry Run (don't generate images, just print prompts to terminal)", value=False)
            with gr.Blocks():
                enable_dynamic_prompts = gr.Checkbox(label="Process Dynamic Prompts (__wildcards__)", value=False)
                gr.Markdown("[Click here for Dynamic Prompts installation instructions.](https://github.com/adieyal/sd-dynamic-prompts)")
        
        submit = gr.Button("Generate", variant="primary")

        with gr.Blocks():
            image_display = gr.Image(label="Output Image", show_label=False, type="pil", interactive=False)
            image_slider = gr.Slider(label="Image", minimum=1, step=1, interactive=True, visible=False)
            image_state = gr.State([])
        with gr.Row():
            html_info = gr.HTML()
            html_log = gr.HTML()

        insert_loras_js = """
        function(...args) {
            let current_prompt = args[0];
            const lora_row_count = args[1];
            let lora_blocks = [];
            for (let i = 0; i < lora_row_count; i++) {
                let selected_loras = args[i*2 + 2];
                let weight = args[i*2 + 3];
                if (selected_loras && selected_loras.length > 0) {
                    let block_parts = selected_loras.map(lora => `<lora:${lora}:${weight}>`);
                    lora_blocks.push(block_parts.join('|'));
                }
            }
            if (lora_blocks.length > 0) {
                let matrix_string = `<${lora_blocks.join('>,<')}>`;
                let new_prompt = current_prompt.trim().endsWith(',') ? current_prompt.trim() + ' ' : current_prompt.trim() + ', ';
                new_prompt += matrix_string;
                const prompt_textarea = document.querySelector("#matrix-prompt textarea");
                prompt_textarea.value = new_prompt;
                updateInput(prompt_textarea); 
                return new_prompt;
            }
            return current_prompt;
        }
        """
        
        # --- Event Handlers ---
        add_lora_btn.click(add_lora_row, inputs=[lora_row_count], outputs=[lora_row_count] + lora_rows)
        refresh_loras_btn.click(fn=update_lora_dropdowns, inputs=[], outputs=lora_dropdowns)
        
        lora_js_inputs = [prompt, lora_row_count]
        for i in range(MAX_LORA_ROWS):
            lora_js_inputs.extend([lora_dropdowns[i], lora_weights[i]])
        insert_loras_btn.click(None, lora_js_inputs, None, _js=insert_loras_js)

        ui_inputs = [
            prompt, negative_prompt, sampler_name, scheduler, steps, cfg_scale, width, height,
            matrix_mode, prompt_type, different_seeds, margin_size, create_mega_grid_toggle, 
            margin_toggle, dry_run, save_prompt_list, use_descriptive_filenames, 
            show_annotations, enable_dynamic_prompts
        ]
        
        clear_prompt_btn.click(fn=lambda: "", inputs=[], outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", inputs=[], outputs=[negative_prompt])
        paste_prompts_btn.click(fn=paste_last_prompts, inputs=[], outputs=[prompt, negative_prompt])
        margin_toggle.change(fn=lambda x: gr.Slider.update(visible=x), inputs=[margin_toggle], outputs=[margin_size])
        matrix_mode.change(fn=lambda mode: gr.Checkbox.update(visible=(mode == "Permutation")), inputs=[matrix_mode], outputs=[create_mega_grid_toggle])
        
        submit.click(
            fn=run_matrix_processing, 
            inputs=ui_inputs, 
            outputs=[image_state, image_display, html_info, html_log, image_slider],
            show_progress="full" 
        ).then(
            fn=None, inputs=None, outputs=None, _js="() => { document.querySelector('#ultimate_matrix button[aria-label=\"Interrupt\"]').scrollIntoView({ behavior: 'smooth', block: 'center' }) }"
        )

        image_slider.change(
            fn=update_image_display,
            inputs=[image_slider, image_state],
            outputs=[image_display]
        )
    
    return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)
