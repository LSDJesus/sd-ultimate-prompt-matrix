"""
Ultimate Prompt Matrix Extension v7.3.4 (Stability Refactor) for AUTOMATIC1111 & Forge

This version fixes a NameError crash on startup caused by missing variable
assignments for the LoRA Builder buttons.
"""

import math
import re
import copy
import os
import random
import time
import json
from PIL import Image, ImageDraw, ImageFont
from itertools import product

import modules.scripts as scripts
import gradio as gr
from modules import images, shared, processing, sd_samplers, sd_schedulers, sd_models
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts

# --- Constants and Regex ---
MAX_LORA_ROWS = 5
PRESETS_DIR = os.path.join(scripts.basedir(), "presets")
REX_MATRIX = re.compile(r'(<(?!lora:)([^>]+)>)')
REX_RANDOM = re.compile(r'<random\(([^)]+)\)>')
REX_PARAM = re.compile(r'<(?!random\b)([a-zA-Z_]+)\(([^)]+)\)>')

# --- Preset Management ---
if not os.path.exists(PRESETS_DIR):
    os.makedirs(PRESETS_DIR)

def list_presets():
    return [""] + [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]

# --- Helper Functions ---
def get_font(fontsize):
    try: return ImageFont.truetype("dejavu.ttf", fontsize)
    except IOError:
        try: return ImageFont.truetype("arial.ttf", fontsize)
        except IOError: return ImageFont.load_default()

def get_lora_names():
    try:
        if hasattr(sd_models, 'refresh_loras'): sd_models.refresh_loras()
        return ["None"] + [lora.name for lora in sd_models.loras]
    except Exception as e:
        print(f"[Ultimate Matrix] Error getting LoRA names: {e}")
        return ["None"]

def draw_grid_with_annotations(grid_images, x_labels, y_labels, margin_size, title="", show_annotations=True):
    if not grid_images or not any(grid_images) or not isinstance(grid_images[0], Image.Image): return None
    num_cols = len(x_labels) if x_labels else math.ceil(math.sqrt(len(grid_images)))
    if num_cols == 0: return None # Prevent division by zero
    num_rows = math.ceil(len(grid_images) / num_cols)
    if num_rows == 0: return None
    img_w, img_h = grid_images[0].size
    label_font, title_font = get_font(30), get_font(36)
    y_label_w = (max(label_font.getbbox(label)[2] for label in y_labels) + margin_size*2) if y_labels and show_annotations else 0
    x_label_h = (label_font.getbbox("Tg")[3] + margin_size*2) if x_labels and show_annotations else 0
    title_h = (title_font.getbbox("Tg")[3] + margin_size*2) if title and show_annotations else 0
    grid_w = y_label_w + (num_cols*img_w) + (margin_size*(num_cols-1)); grid_h = title_h + x_label_h + (num_rows*img_h) + (margin_size*(num_rows-1))
    grid_image = Image.new('RGB', (int(grid_w), int(grid_h)), color='white')
    draw = ImageDraw.Draw(grid_image)
    if title and show_annotations: draw.text((grid_w/2, margin_size), title, font=title_font, fill='black', anchor="mt")
    if x_labels and show_annotations:
        for i, label in enumerate(x_labels): draw.text((y_label_w + (i*(img_w+margin_size)) + (img_w/2), title_h+margin_size), label, font=label_font, fill='black', anchor="mt")
    if y_labels and show_annotations:
        for i, label in enumerate(y_labels): draw.text((margin_size, title_h+x_label_h+(i*(img_h+margin_size))+(img_h/2)), label, font=label_font, fill='black', anchor="lm")
    for i, img in enumerate(grid_images):
        col, row = i % num_cols, i // num_cols
        grid_image.paste(img, (int(y_label_w+col*(img_w+margin_size)), int(title_h+x_label_h+row*(img_h+margin_size))))
    return grid_image

def create_mega_grid(all_grids, page_labels, margin_size, show_annotations=True):
    valid_grids = [g for g in all_grids if g is not None]
    if not valid_grids or len(valid_grids) <= 1: return None
    mega_cols = math.ceil(math.sqrt(len(valid_grids)))
    mega_rows = math.ceil(len(valid_grids)/mega_cols)
    grid_w, grid_h = valid_grids[0].size
    font, title_h = get_font(36), 50 if show_annotations else 0
    mega_w = mega_cols*grid_w + margin_size*(mega_cols+1); mega_h = mega_rows*(grid_h+title_h) + margin_size*(mega_rows+1)
    mega_image = Image.new('RGB', (int(mega_w), int(mega_h)), color='#DDDDDD')
    draw = ImageDraw.Draw(mega_image)
    for i, grid in enumerate(valid_grids):
        col, row = i % mega_cols, i // mega_cols
        cell_x, cell_y = margin_size + col*(grid_w+margin_size), margin_size + row*(grid_h+title_h+margin_size)
        if show_annotations: draw.text((cell_x+grid_w/2, cell_y+title_h/2), page_labels[i], font=font, fill='black', anchor="mm")
        mega_image.paste(grid, (int(cell_x), int(cell_y + title_h)))
    return mega_image

def sanitize_filename(text):
    if not text: return ""
    return re.sub(r'[\\/*?:"<>|]', '', text)[:100]

def paste_last_prompts():
    if hasattr(shared, 'last_info') and shared.last_info:
        info = shared.last_info; neg_match = re.search(r'Negative prompt: (.+?)(?=\nSteps:)', info, re.DOTALL)
        return info.split('Negative prompt:')[0].strip(), neg_match.group(1).strip() if neg_match else ""
    return "Could not find last generation info.", ""

# --- Core Logic Functions ---
def parse_axis_options(prompt):
    param_matches = list(REX_PARAM.finditer(prompt)); random_matches = list(REX_RANDOM.finditer(prompt))
    specific_matches = param_matches + random_matches
    temp_prompt = prompt
    for match in specific_matches:
        start, end = match.span(); temp_prompt = temp_prompt[:start] + (' ' * (end - start)) + temp_prompt[end:]
    generic_matches = list(REX_MATRIX.finditer(temp_prompt))
    all_matches = sorted(specific_matches + generic_matches, key=lambda m: m.start())
    options = []
    for match in all_matches:
        is_param = (match.re == REX_PARAM); is_random = (match.re == REX_RANDOM)
        param_name = match.group(1) if is_param else ('random' if is_random else None)
        options.append({"token":match.group(0), "options":[opt.strip() for opt in match.group(2 if not is_random else 1).split("|")],
                        "is_param":is_param or is_random, "param_name":param_name, "start": match.start(), "end": match.end()})
    return options

def build_job_list(p, seed_behavior, axis_overrides=None):
    pos_axes = parse_axis_options(p.prompt); neg_axes = parse_axis_options(p.negative_prompt)
    for axis in pos_axes: axis['origin'] = 'positive'
    for axis in neg_axes: axis['origin'] = 'negative'
    all_axes = axis_overrides if axis_overrides is not None else sorted(pos_axes + neg_axes, key=lambda x: x.get('start', 0))
    if not all_axes: return [], None
    random_axes = [axis for axis in all_axes if axis.get('param_name') == 'random']
    grid_axes = [axis for axis in all_axes if axis.get('param_name') != 'random']
    if not grid_axes: return [], None

    x_axis = grid_axes[-1] if len(grid_axes) > 0 else None; y_axis = grid_axes[-2] if len(grid_axes) > 1 else None
    page_axes = grid_axes[:-2] if len(grid_axes) > 2 else []
    x_options = x_axis['options'] if x_axis else ['']; y_options = y_axis['options'] if y_axis else ['']
    page_options_list = [axis['options'] for axis in page_axes]
    page_combinations = list(product(*page_options_list)) if page_options_list else [()]

    job_list = []
    for page_idx, page_vals in enumerate(page_combinations):
        for row_idx, y_val in enumerate(y_options):
            for col_idx, x_val in enumerate(x_options):
                job = { "pos_prompt": p.prompt, "neg_prompt": p.negative_prompt, "params": {} }
                for r_axis in random_axes:
                    chosen_val = random.choice(r_axis['options']); key = 'pos_prompt' if r_axis['origin'] == 'positive' else 'neg_prompt'
                    job[key] = job[key].replace(r_axis['token'], chosen_val, 1)
                assignments = []
                if page_axes: assignments.extend(zip(page_axes, page_vals))
                if y_axis: assignments.append((y_axis, y_val));
                if x_axis: assignments.append((x_axis, x_val))
                for axis, value in assignments:
                    if axis['is_param']: job['params'][axis['param_name']] = value
                    else:
                        key = 'pos_prompt' if axis['origin'] == 'positive' else 'neg_prompt'
                        job[key] = job[key].replace(axis['token'], value, 1)
                
                # Use a dummy seed for job list creation, real seed is set in the job
                image_index = (page_idx*len(y_options)*len(x_options))+(row_idx*len(x_options))+col_idx
                if seed_behavior == "Fixed": job['seed'] = p.seed
                elif seed_behavior == "Iterate Per Image": job['seed'] = p.seed + image_index
                elif seed_behavior == "Iterate Per Row": job['seed'] = p.seed + (page_idx * len(y_options)) + row_idx
                elif seed_behavior == "Random": job['seed'] = -1
                job_list.append(job)

    for job in job_list:
        job['pos_prompt'] = re.sub(REX_PARAM, '', re.sub(REX_MATRIX, '', re.sub(REX_RANDOM, '', job['pos_prompt']))).strip(', ')
        job['neg_prompt'] = re.sub(REX_PARAM, '', re.sub(REX_MATRIX, '', re.sub(REX_RANDOM, '', job['neg_prompt']))).strip(', ')

    grid_info = {"x_labels": x_options, "y_labels": y_options, "page_combinations": page_combinations,
                 "page_labels": [", ".join(f"{ax.get('param_name', '')}:{v}" if ax.get('is_param') else v for ax, v in zip(page_axes, vals)) for vals in page_combinations]}
    return job_list, grid_info

def run_matrix_processing(p_text, n_text, seed_behavior, axis_overrides, base_params, margin, mega, desc_fn, save_list, show_anno, dyn_prompts):
    # Create a temporary 'p' object just for building the job list. It will not be "consumed".
    p_temp = StableDiffusionProcessingTxt2Img(prompt=p_text, negative_prompt=n_text, seed=-1)
    job_list, grid_info = build_job_list(p_temp, seed_behavior, axis_overrides)
    if not job_list:
        yield {"html_log": "No matrix syntax found."}
        return

    if save_list:
        log_path = os.path.join(opts.outdir_grids or opts.outdir_txt2img_grids, f"prompt_log_{int(time.time())}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"--- Ultimate Prompt Matrix Log ---\n\n")
            for i, job in enumerate(job_list): f.write(f"{i+1:03d}: Pos: '{job['pos_prompt']}' | Neg: '{job['neg_prompt']}' | Params: {job['params']} | Seed: {job['seed']}\n")
        print(f"Saved prompt list to {log_path}")

    shared.state.job_count = len(job_list)
    all_generated_images, all_infotexts, all_grid_images = [], [], []
    original_checkpoint = shared.sd_model.sd_checkpoint_info.title if hasattr(shared.sd_model, 'sd_checkpoint_info') and shared.sd_model.sd_checkpoint_info else None
    
    dynamic_prompts_active = False
    if dyn_prompts:
        try: from sd_dynamic_prompts.prompt_parser import parse; dynamic_prompts_active = True
        except ImportError: print("WARNING: Dynamic Prompts extension not found.")

    for i, job in enumerate(job_list):
        if shared.state.interrupted: break
        shared.state.job = f"Image {i+1}/{len(job_list)}"
        
        # Create a BRAND NEW processing object for every job
        p_job = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=job['pos_prompt'],
            negative_prompt=job['neg_prompt'],
            seed=job['seed'],
            **base_params # Apply the base settings from the UI
        )
        p_job.n_iter = 1
        p_job.batch_size = 1

        for param, value in job['params'].items():
            param_lower = param.lower()
            try:
                if param_lower == 'cfg': p_job.cfg_scale = float(value)
                elif param_lower == 'steps': p_job.steps = int(value)
                elif param_lower == 'sampler': p_job.sampler_name = value
                elif param_lower == 'scheduler': p_job.scheduler = value
                elif param_lower == 'checkpoint':
                    if value != shared.sd_model.sd_checkpoint_info.title:
                        info = sd_models.get_closet_checkpoint_match(value)
                        if info: sd_models.reload_model_weights(shared.sd_model, info)
            except Exception as e: print(f"Warning: Could not apply parameter '{param}' with value '{value}'. Error: {e}")
        
        if dynamic_prompts_active: p_job.prompt = parse(p_job.prompt)
        
        processed_single = process_images(p_job)
        all_generated_images.append(processed_single.images[0])
        all_infotexts.append(processed_single.infotexts[0])
        yield {"gallery": all_generated_images,
               "html_info": processed_single.infotexts[0],
               "html_log": f"Generated {len(all_generated_images)} of {len(job_list)} images."}

    if shared.state.interrupted: print("Matrix generation interrupted."); return
    
    if original_checkpoint and original_checkpoint != shared.sd_model.sd_checkpoint_info.title:
        print(f"[Ultimate Matrix] Restoring original checkpoint: {original_checkpoint}")
        sd_models.reload_model_weights()
    
    images_per_grid = len(grid_info['x_labels']) * len(grid_info['y_labels'])
    for i, page_label in enumerate(grid_info['page_labels']):
        images_for_this_grid = all_generated_images[i*images_per_grid:(i+1)*images_per_grid]
        title = f"Page {i+1}/{len(grid_info['page_labels'])}: {page_label}" if len(grid_info['page_labels']) > 1 else page_label
        grid_image = draw_grid_with_annotations(images_for_this_grid, grid_info['x_labels'], grid_info['y_labels'], margin, title, show_anno)
        if grid_image:
            all_grid_images.append(grid_image)
            # FINAL FIX: Use the un-consumed p_temp for grid saving metadata
            if opts.grid_save: images.save_image(grid_image, (opts.outdir_grids or opts.outdir_txt2img_grids), f"matrix_{sanitize_filename(page_label) if desc_fn else f'page_{i+1}'}", p=p_temp)
    
    mega_grid = create_mega_grid(all_grid_images, grid_info['page_labels'], margin, show_anno) if mega and len(all_grid_images) > 1 else None
    # FINAL FIX: Use the un-consumed p_temp for grid saving metadata
    if mega_grid and opts.grid_save: images.save_image(mega_grid, (opts.outdir_grids or opts.outdir_txt2img_grids), "mega-matrix", p=p_temp)
    
    yield {"all_grids_state": [mega_grid] + all_grid_images if mega_grid else all_grid_images,
           "summary_grid_display": mega_grid if mega_grid else (all_grid_images[0] if all_grid_images else None),
           "html_log": f"Finished generating {len(job_list)} images.",
           "tabs": gr.Tabs.update(selected=1)}

# --- UI Definition ---
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        # UI Components
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, variant="panel"):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter prompts with <matrix|syntax> here..."); clear_prompt_btn = gr.Button("🗑️", elem_classes="tool", size="sm"); paste_prompts_btn = gr.Button("↙️", elem_classes="tool", size="sm")
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Enter negative prompts here..."); clear_neg_prompt_btn = gr.Button("🗑️", elem_classes="tool", size="sm")
                with gr.Accordion("Generation Settings", open=True):
                    sampler_name = gr.Dropdown(label='Sampling method', choices=[s.name for s in sd_samplers.samplers], value=opts.data.get('sampler_name', 'Euler a'))
                    scheduler = gr.Dropdown(label='Schedule type', choices=[s.label for s in sd_schedulers.schedulers], value=opts.data.get('scheduler', 'Automatic'))
                    steps = gr.Slider(label='Steps', minimum=1, maximum=150, value=20, step=1); cfg_scale = gr.Slider(label='CFG', minimum=1.0, maximum=30.0, value=7.0, step=0.5)
                    width = gr.Slider(label='Width', minimum=64, maximum=2048, value=512, step=64); height = gr.Slider(label='Height', minimum=64, maximum=2048, value=512, step=64)
                    seed_behavior = gr.Radio(["Iterate Per Image", "Iterate Per Row", "Fixed", "Random"], label="Seed Behavior", value="Iterate Per Image")
                with gr.Accordion("LoRA Matrix Builder", open=False):
                    refresh_loras_btn = gr.Button("🔃 Refresh LoRAs", size="sm")
                    lora_rows, lora_dropdowns, lora_weights = [], [], []
                    for i in range(MAX_LORA_ROWS):
                        with gr.Row(visible=(i==0)) as row:
                            d = gr.Dropdown([], multiselect=True, label=f"Block {i+1}"); w = gr.Slider(minimum=-2.0, maximum=2.0, value=1.0, step=0.05, label="Weight")
                            lora_rows.append(row); lora_dropdowns.append(d); lora_weights.append(w)
                    lora_row_count = gr.State(1)
                    add_lora_btn = gr.Button("Add LoRA Block")
                    insert_loras_btn = gr.Button("Insert LoRA Matrix", variant="primary")
            with gr.Column(scale=1, variant="panel"):
                with gr.Accordion("Matrix Preview & Control", open=True):
                    axis_x_display = gr.Textbox(label="X-Axis", interactive=False); axis_y_display = gr.Textbox(label="Y-Axis", interactive=False)
                    axis_page_display = gr.Textbox(label="Page-Axes", interactive=False); random_axis_display = gr.Textbox(label="Random Axes (Not in Grid)", interactive=False)
                    detected_axes_state = gr.State([]); swap_xy_btn = gr.Button("Swap X/Y"); cycle_page_btn = gr.Button("Cycle Page/Y/X")
                with gr.Accordion("Grid & File Settings", open=True):
                    show_annotations = gr.Checkbox(label="Show grid labels & titles", value=True)
                    margin_toggle = gr.Checkbox(label="Enable Grid Margins", value=True)
                    margin_size = gr.Slider(label="Grid Margins (px)", minimum=0, maximum=200, value=10, step=2)
                    create_mega_grid_toggle = gr.Checkbox(label="Create \"Mega-Grid\": combine all page grids into one image", value=True)
                    use_descriptive_filenames = gr.Checkbox(label="Descriptive filenames: use page axis values in grid filenames", value=False)
                    save_prompt_list = gr.Checkbox(label="Save job list: creates a text file with all generated prompts", value=False)
                with gr.Accordion("Preset Management", open=False):
                    preset_dropdown = gr.Dropdown(label="Load Preset", choices=list_presets()); refresh_presets_btn = gr.Button("🔃", elem_classes="tool", size="sm")
                    save_preset_name = gr.Textbox(label="Save as Preset", placeholder="Enter name..."); save_preset_btn = gr.Button("Save", size="sm")
        
        with gr.Row():
            base_speed_input = gr.Number(label="Time for 512x512 (s)", value=5.0); calculate_btn = gr.Button("Pre-process & Calculate", variant="secondary")
        calculation_results_display = gr.Markdown("Click 'Pre-process' to calculate batch size and time.")
        with gr.Row():
            submit_button_main = gr.Button("Generate", variant="primary", interactive=False); generate_anyways_button = gr.Button("Generate Anyways", variant="stop", interactive=False); dry_run_btn = gr.Button("Dry Run to Console")
        
        tabs = gr.Tabs()
        with tabs:
            with gr.TabItem("All Images", id=0):
                gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="ultimate_matrix_gallery", columns=8, object_fit="contain", height="auto")
                all_images_state = gr.State([])
            with gr.TabItem("Summary Grids", id=1):
                summary_grid_display = gr.Image(type="pil", interactive=False, height=768)
                all_grids_state = gr.State([])
        
        html_info = gr.HTML(); html_log = gr.HTML()
        
        with gr.Accordion("Single Prompt Sandbox", open=False):
             gr.Markdown("Use this to test a single prompt before building a matrix, or to get an accurate speed reading for the calculator above.")
             sandbox_prompt = gr.Textbox(label="Sandbox Prompt", lines=2, value="photo of a cat"); sandbox_negative_prompt = gr.Textbox(label="Sandbox Negative Prompt", lines=2)
             generate_sandbox_btn = gr.Button("Generate Test Image", variant="secondary")
             sandbox_image_display = gr.Image(type="pil", interactive=False); sandbox_log_display = gr.HTML()
        with gr.Accordion("Advanced & Help", open=False):
            enable_dynamic_prompts = gr.Checkbox(label="Process Dynamic Prompts", value=True)
            ultimate_matrix_large_batch_threshold = gr.Number(label="Large Batch Warning Threshold", value=100, precision=0)
            gr.Markdown("### Syntax Guide\n- **Prompt Matrix:** `a photo of a <cat|dog>`\n- **Parameter Matrix:** `<cfg(5|7.5)>`\n- **Supported:** `cfg`, `steps`, `sampler`, `scheduler`, `checkpoint`\n- **Random:** `<random(cat|dog)>`")
        
        # --- Event Handlers ---
        def on_calculate_click(prompt_text, neg_prompt_text, w, h, base_speed, threshold):
            p = StableDiffusionProcessingTxt2Img(prompt=prompt_text, negative_prompt=neg_prompt_text)
            job_list, _ = build_job_list(p, "Iterate Per Image", None)
            if not job_list: return "No matrix syntax found.", gr.update(), gr.update(), "N/A", "N/A", "N/A", "N/A", []
            total_images = len(job_list)
            est_time = (base_speed * (w * h / 512**2)) * total_images; time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(est_time))
            result_text = f"**Total Images:** {total_images}\n- **Estimated Time:** {time_str}"
            gen_interactive = total_images <= threshold
            
            pos_axes = parse_axis_options(prompt_text)
            neg_axes = parse_axis_options(neg_prompt_text)
            for axis in pos_axes: axis['origin'] = 'positive'
            for axis in neg_axes: axis['origin'] = 'negative'
            axes = sorted(pos_axes + neg_axes, key=lambda x: x.get('start', 0))

            grid_axes = [ax for ax in axes if ax.get('param_name') != 'random']
            random_axes_list = [ax for ax in axes if ax.get('param_name') == 'random']
            x_opts = grid_axes[-1]['token'] if len(grid_axes) > 0 else "N/A"
            y_opts = grid_axes[-2]['token'] if len(grid_axes) > 1 else "N/A"
            p_opts = ", ".join([ax['token'] for ax in grid_axes[:-2]]) if len(grid_axes) > 2 else "N/A"
            r_opts = ", ".join([ax['token'] for ax in random_axes_list]) if random_axes_list else "N/A"
            return result_text, gr.update(interactive=gen_interactive), gr.update(interactive=not gen_interactive, value=f"Generate Anyways ({total_images})"), x_opts, y_opts, p_opts, r_opts, axes
        
        outputs_list = [calculation_results_display, submit_button_main, generate_anyways_button, axis_x_display, axis_y_display, axis_page_display, random_axis_display, detected_axes_state]
        calculate_btn.click(on_calculate_click, inputs=[prompt, negative_prompt, width, height, base_speed_input, ultimate_matrix_large_batch_threshold], outputs=outputs_list)
        
        def on_axis_control_click(axes, mode):
            if not axes: return axes, "N/A", "N/A", "N/A"
            grid_axes = [ax for ax in axes if ax.get('param_name') != 'random']
            if mode == 'swap' and len(grid_axes) >= 2: grid_axes[-1], grid_axes[-2] = grid_axes[-2], grid_axes[-1]
            if mode == 'cycle' and len(grid_axes) >= 3: grid_axes = [grid_axes[-1]] + grid_axes[:-1]
            new_axes = [ax for ax in axes if ax.get('param_name') == 'random'] + grid_axes
            x_opts = grid_axes[-1]['token'] if len(grid_axes) > 0 else "N/A"
            y_opts = grid_axes[-2]['token'] if len(grid_axes) > 1 else "N/A"
            p_opts = ", ".join([ax['token'] for ax in grid_axes[:-2]]) if len(grid_axes) > 2 else "N/A"
            return new_axes, x_opts, y_opts, p_opts
        
        swap_xy_btn.click(lambda axes: on_axis_control_click(axes, 'swap'), inputs=[detected_axes_state], outputs=[detected_axes_state, axis_x_display, axis_y_display, axis_page_display])
        cycle_page_btn.click(lambda axes: on_axis_control_click(axes, 'cycle'), inputs=[detected_axes_state], outputs=[detected_axes_state, axis_x_display, axis_y_display, axis_page_display])

        run_outputs_dict = {
            "gallery": gallery,
            "all_images_state": all_images_state,
            "all_grids_state": all_grids_state,
            "summary_grid_display": summary_grid_display,
            "html_info": html_info,
            "html_log": html_log,
            "tabs": tabs,
            "submit_button_main": submit_button_main,
            "generate_anyways_button": generate_anyways_button,
            "dry_run_btn": dry_run_btn
        }
        
        def on_generate_click(dry_run, p_text, n_text, detected_axes, *args):
            yield {b: gr.update(interactive=False) for b in [submit_button_main, generate_anyways_button, dry_run_btn]}
            
            base_params = {k: v for k, v in zip(['sampler_name', 'scheduler', 'steps', 'cfg_scale', 'width', 'height'], args[:6])}
            run_args_list = [p_text, n_text, args[6], detected_axes, base_params] + list(args[7:])

            if dry_run:
                # Dry run is not a generator, so we need to handle its return differently
                for update in run_matrix_processing(*run_args_list):
                     yield {run_outputs_dict[k]: v for k, v in update.items() if k in run_outputs_dict}
            else:
                for update in run_matrix_processing(*run_args_list):
                    yield {run_outputs_dict[k]: v for k, v in update.items()}
            
            yield {submit_button_main: gr.update(interactive=True), generate_anyways_button: gr.update(interactive=False), dry_run_btn: gr.update(interactive=True)}
            
        run_args = [prompt, negative_prompt, detected_axes_state, sampler_name, scheduler, steps, cfg_scale, width, height, seed_behavior, margin_size, create_mega_grid_toggle, use_descriptive_filenames, save_prompt_list, show_annotations, enable_dynamic_prompts]
        
        submit_button_main.click(on_generate_click, inputs=[gr.State(False)] + run_args, outputs=list(run_outputs_dict.values()), show_progress="full")
        generate_anyways_button.click(on_generate_click, inputs=[gr.State(False)] + run_args, outputs=list(run_outputs_dict.values()), show_progress="full")
        dry_run_btn.click(on_generate_click, inputs=[gr.State(True)] + run_args, outputs=list(run_outputs_dict.values()))
        
        ui_inputs_for_preset = [prompt, negative_prompt, sampler_name, scheduler, steps, cfg_scale, width, height, seed_behavior, show_annotations, margin_toggle, margin_size, create_mega_grid_toggle, use_descriptive_filenames, save_prompt_list, enable_dynamic_prompts]
        def save_preset(name, *args):
            if not name: return
            filepath = os.path.join(PRESETS_DIR, f"{sanitize_filename(name)}.json")
            with open(filepath, 'w') as f: json.dump({comp.label: val for comp, val in zip(ui_inputs_for_preset, args)}, f, indent=4)
            return gr.Dropdown.update(choices=list_presets())
        def load_preset(name):
            if not name: return [gr.update() for _ in ui_inputs_for_preset]
            with open(os.path.join(PRESETS_DIR, name), 'r') as f: data = json.load(f)
            return [data.get(comp.label) for comp in ui_inputs_for_preset]
        save_preset_btn.click(save_preset, inputs=[save_preset_name] + ui_inputs_for_preset, outputs=[preset_dropdown])
        preset_dropdown.change(load_preset, inputs=[preset_dropdown], outputs=ui_inputs_for_preset)
        refresh_presets_btn.click(lambda: gr.Dropdown.update(choices=list_presets()), outputs=[preset_dropdown])
        
        add_lora_btn.click(lambda c: (min(c + 1, MAX_LORA_ROWS), *[gr.update(visible=i < min(c + 1, MAX_LORA_ROWS)) for i in range(MAX_LORA_ROWS)]), inputs=[lora_row_count], outputs=[lora_row_count] + lora_rows)
        refresh_loras_btn.click(lambda: [gr.update(choices=get_lora_names()) for _ in range(MAX_LORA_ROWS)], outputs=lora_dropdowns)
        def insert_loras(p, c, *a):
            matrix = f"<{'>,<'.join(['|'.join([f'<lora:{n}:{a[i*2+1]}>' for n in a[i*2]]) for i in range(c) if a[i*2]])}>"
            return (p.strip() + (", " if p.strip() else "") + matrix) if matrix != "<>" else p
        insert_loras_btn.click(insert_loras, inputs=[prompt, lora_row_count] + lora_dropdowns + lora_weights, outputs=[prompt])

        def run_sandbox(p, n, s, samp, sched, st, cfg, w, h):
            p_obj = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, prompt=p, negative_prompt=n, sampler_name=samp, scheduler=sched, steps=st, cfg_scale=cfg, width=w, height=h, seed=s)
            proc = process_images(p_obj)
            return proc.images[0], f"Generated in {proc.execution_time:.2f}s"
        generate_sandbox_btn.click(run_sandbox, inputs=[sandbox_prompt, sandbox_negative_prompt, gr.State(-1), sampler_name, scheduler, steps, cfg_scale, width, height], outputs=[sandbox_image_display, sandbox_log_display])

    return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)