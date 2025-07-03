"""
Ultimate Prompt Matrix Extension v13.2.1 (Clarity & Dummy-Proofing) for AUTOMATIC1111 & Forge
- NEW: UI and logic completely redesigned to match the v13.2.0 flowchart.
- NEW: "Matrix Builder" UI for defining matrix axes with labels.
- NEW: Prompt syntax now uses simple reference tags like |matrix:animal| or |Δ:cfg|.
- NEW: Centralized "Pre-process & Stage Job" button with toggleable modes.
- RETAINED: Robust two-phase execution, manual grid infotext fix, and Python-based paste button.
- FIX: Improved robustness and error messages for clarity.
"""
import os
import re
import json
import gradio as gr
import modules.scripts as scripts
from modules import images, shared, processing, sd_samplers, sd_schedulers, sd_models
from modules.processing import process_images, StableDiffusionProcessingTxt2Img

# IMPORTANT: We now import our own helper files.
# Make sure these files exist in the same /scripts/ folder!
from scripts.upm_logic import build_job_list
from scripts.upm_grid_drawing import draw_grid_with_annotations, create_mega_grid
from scripts.upm_utils import sanitize_filename, paste_last_prompts

# --- CONSTANTS ---
MAX_MATRIX_BLOCKS = 5
PRESETS_DIR = os.path.join(scripts.basedir(), "presets")
if not os.path.exists(PRESETS_DIR): os.makedirs(PRESETS_DIR)

# --- HELPER FUNCTIONS ---
def list_presets(): return [""] + [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]

def format_label(axis, value):
    # This function creates the text labels for the grid axes (e.g., "cfg: 7.0")
    if not axis or axis.get('param_name', 'prompt') in ['prompt', 'random']:
        return value
    return f"{axis['param_name']}: {value}"

# --- AXIS CONTROL HELPER FUNCTIONS ---
def swap_tags_in_text(text):
    """Finds the last two |tags| in a string and swaps their positions."""
    tags = re.findall(r'\|(?:matrix|random|Δ):[^|]+\|', text)
    if len(tags) < 2:
        return text # Not enough tags to swap

    last_tag = tags[-1]
    second_last_tag = tags[-2]

    # To handle cases where tags might be identical, we replace by position from the end.
    last_pos = text.rfind(last_tag)
    second_last_pos = text.rfind(second_last_tag, 0, last_pos)

    # If we found both correctly
    if last_pos != -1 and second_last_pos != -1:
        # Build the new string piece by piece to avoid errors
        part1 = text[:second_last_pos]
        part2 = text[second_last_pos + len(second_last_tag) : last_pos]
        part3 = text[last_pos + len(last_tag):]
        return f"{part1}{last_tag}{part2}{second_last_tag}{part3}"

    return text # Return original if something went wrong

def cycle_tags_in_text(text):
    """Finds the last three |tags| in a string and cycles them (Page->Y, Y->X, X->Page)."""
    tags = re.findall(r'\|(?:matrix|random|Δ):[^|]+\|', text)
    if len(tags) < 2: # Allow cycling even with 2, which is just a swap
        return swap_tags_in_text(text)
    
    # Identify the last, second-last, and third-last tags
    tags_to_cycle = tags[-3:] if len(tags) >= 3 else tags[-2:]
    
    # The new order is the last element moved to the front
    new_order = [tags_to_cycle[-1]] + tags_to_cycle[:-1]
    
    # Replace the old tags with the new order, starting from the first tag's position
    temp_text = text
    for i, old_tag in enumerate(tags_to_cycle):
        # We replace one by one to maintain order
        temp_text = temp_text.replace(old_tag, new_order[i], 1)
        
    return temp_text

# --- CORE BACKEND PHASES (These are the reliable workhorse functions) ---

# PHASE 2: IMAGE GENERATION
def run_image_generation(job_list, base_params, dyn_prompts):
    shared.state.job_count = len(job_list)
    all_generated_images, all_infotexts = [], []
    original_checkpoint = shared.sd_model.sd_checkpoint_info.title if hasattr(shared.sd_model, 'sd_checkpoint_info') else "Unknown"

    print(f"[UPM] Starting Phase 2: Silent Image Generation of {len(job_list)} images.")
    for i, job in enumerate(job_list):
        if shared.state.interrupted:
            print("[UPM] Generation interrupted by user.")
            break
        shared.state.job = f"Image {i+1}/{len(job_list)}"
        print(f"[UPM] Generating image {i+1}/{len(job_list)} with prompt: {job['pos_prompt']}")

        p_job = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
            prompt=job['pos_prompt'],
            negative_prompt=job['neg_prompt'],
            seed=job['seed'],
            **base_params
        )
        p_job.n_iter, p_job.batch_size = 1, 1

        for param, value in job['params'].items():
            param_lower = param.lower()
            try:
                if param_lower == 'cfg': p_job.cfg_scale = float(value)
                elif param_lower == 'steps': p_job.steps = int(value)
                elif param_lower == 'sampler': p_job.sampler_name = value
                elif param_lower == 'scheduler': p_job.scheduler = value
                elif param_lower == 'checkpoint':
                    current_checkpoint_title = shared.sd_model.sd_checkpoint_info.title if hasattr(shared.sd_model, 'sd_checkpoint_info') else "Unknown"
                    if value != current_checkpoint_title:
                        info = sd_models.get_closet_checkpoint_match(value)
                        if info:
                            print(f"[UPM] Changing checkpoint to: {info.title}")
                            sd_models.reload_model_weights(shared.sd_model, info)
                        else:
                            print(f"[UPM] Warning: Checkpoint '{value}' not found. Using current one.")
            except Exception as e:
                print(f"[UPM] Warning: Could not apply parameter '{param}' with value '{value}'. Error: {e}")

        if dyn_prompts:
            try:
                from sd_dynamic_prompts.prompt_parser import parse
                p_job.prompt = parse(p_job.prompt)
            except ImportError:
                print("[UPM] Warning: 'Process Dynamic Prompts' is checked, but the extension is not available.")
                pass

        processed_single = process_images(p_job)
        all_generated_images.append(processed_single.images[0])
        all_infotexts.append(processed_single.infotexts[0])

    current_checkpoint_title = shared.sd_model.sd_checkpoint_info.title if hasattr(shared.sd_model, 'sd_checkpoint_info') else "Unknown"
    if original_checkpoint != "Unknown" and original_checkpoint != current_checkpoint_title:
        print(f"[UPM] Restoring original checkpoint: {original_checkpoint}")
        sd_models.reload_model_weights()

    print("[UPM] Phase 2 complete.")
    return all_generated_images, all_infotexts

# PHASE 3: GRID GENERATION (This function is very stable)
def run_grid_generation(images, infotexts, grid_info, p_base, margin, mega, desc_fn, show_anno):
    print("[UPM] Starting Phase 3: Grid Generation.")
    if not grid_info:
        print("[UPM] No grid information available. Skipping grid generation.")
        return [], None

    x_labels = [format_label(grid_info.get('x_axis'), v) for v in (grid_info.get('x_axis', {}).get('options', ['']))]
    y_labels = [format_label(grid_info.get('y_axis'), v) for v in (grid_info.get('y_axis', {}).get('options', ['']))]
    page_labels = [", ".join(format_label(ax, v) for ax, v in zip(grid_info.get('page_axes', []), vals)) for vals in grid_info.get('page_combinations', [[]])]

    all_grid_images = []
    images_per_grid = len(x_labels) * len(y_labels)
    if images_per_grid == 0:
        print("[UPM] Cannot generate grid with 0 images per grid. There may only be random axes defined.")
        return [], None

    # CRITICAL FIX: Manually build a safe infotext for grids to prevent crashes.
    safe_infotext_lines = [
        f"Prompt: {p_base.prompt}",
        f"Negative prompt: {p_base.negative_prompt}",
        f"Steps: {p_base.steps}, Sampler: {p_base.sampler_name}, CFG scale: {p_base.cfg_scale}",
        f"Seed: {p_base.seed}, Size: {p_base.width}x{p_base.height}"
    ]
    if hasattr(shared.sd_model, 'sd_checkpoint_info') and shared.sd_model.sd_checkpoint_info:
        safe_infotext_lines.append(f"Model: {shared.sd_model.sd_checkpoint_info.title}, Model hash: {shared.sd_model.sd_checkpoint_info.hash}")
    safe_infotext = "\n".join(safe_infotext_lines)

    for i, page_label in enumerate(page_labels):
        start_index = i * images_per_grid
        end_index = (i + 1) * images_per_grid
        images_for_this_grid = images[start_index:end_index]

        if not images_for_this_grid:
            print(f"[UPM] Warning: No images found for page {i+1}. It may have been interrupted.")
            continue

        title = f"Page {i+1}/{len(page_labels)}: {page_label}" if len(page_labels) > 1 else (page_labels[0] or "Grid")
        grid_image = draw_grid_with_annotations(images_for_this_grid, x_labels, y_labels, margin, title, show_anno)

        if grid_image:
            all_grid_images.append(grid_image)
            if shared.opts.grid_save:
                filename = f"matrix_{sanitize_filename(page_label) if desc_fn else f'page_{i+1}'}"
                images.save_image(grid_image, (shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids), filename, info=safe_infotext, p=None)

    mega_grid = create_mega_grid(all_grid_images, page_labels, margin, show_anno) if mega and len(all_grid_images) > 1 else None
    if mega_grid and shared.opts.grid_save:
        images.save_image(mega_grid, (shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids), "mega-matrix", info=safe_infotext, p=None)

    print("[UPM] Phase 3 complete.")
    return all_grid_images, mega_grid

# --- THE UI (This is where all the buttons and sliders are defined) ---
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        # --- HIDDEN "MEMORY" FOR THE SCRIPT ---
        staged_job_list = gr.State([])
        staged_grid_info = gr.State({})
        staged_p_base = gr.State(None)
        staged_base_params = gr.State({})
        staged_images = gr.State([])
        staged_infotexts = gr.State([])
        staged_matrix_definitions = gr.State([])

        with gr.Row(equal_height=False):
            # --- LEFT COLUMN (INPUTS) ---
            with gr.Column(scale=2, variant="panel"):
                with gr.Accordion("Matrix Builder", open=True):
                    gr.Markdown("Define your matrix axes here. Use the 'Label' in your prompt, e.g., `a photo of a |matrix:animal|`.")
                    matrix_builder_inputs = []
                    for i in range(MAX_MATRIX_BLOCKS):
                        with gr.Row(visible=(i < 2), elem_classes=f"upm-matrix-row-{i}"): # Show first two rows by default
                            enabled = gr.Checkbox(label="On", value=(i == 0), elem_classes="upm-matrix-enable")
                            m_type = gr.Dropdown(["matrix", "random", "Δ"], value="matrix", label="Type", elem_classes="upm-matrix-type")
                            m_label = gr.Textbox(placeholder="e.g., animal, cfg, style", label="Label", elem_classes="upm-matrix-label")
                            m_vars = gr.Textbox(placeholder="Comma-separated values", label="Variables", lines=1, elem_classes="upm-matrix-vars")
                            matrix_builder_inputs.extend([enabled, m_type, m_label, m_vars])

                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter prompts with |matrix:label| syntax here...", elem_id="upm_prompt")
                    clear_prompt_btn = gr.Button("🗑️", elem_classes="tool", size="sm")
                    paste_prompts_btn = gr.Button("↙️", elem_classes="tool", size="sm", tooltip="Paste Last Generation's Prompts")
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Enter negative prompts here...", elem_id="upm_negative_prompt")
                    clear_neg_prompt_btn = gr.Button("🗑️", elem_classes="tool", size="sm")

                with gr.Accordion("Generation Settings", open=True):
                    sampler_name = gr.Dropdown(label='Sampling method', choices=[s.name for s in sd_samplers.samplers], value=shared.opts.data.get('sampler_name', 'Euler a'))
                    scheduler = gr.Dropdown(label='Schedule type', choices=[s.label for s in sd_schedulers.schedulers], value=shared.opts.data.get('scheduler', 'Automatic'))
                    steps = gr.Slider(label='Steps', minimum=1, maximum=150, value=20, step=1)
                    cfg_scale = gr.Slider(label='CFG', minimum=1.0, maximum=30.0, value=7.0, step=0.5)
                    width = gr.Slider(label='Width', minimum=64, maximum=2048, value=512, step=64)
                    height = gr.Slider(label='Height', minimum=64, maximum=2048, value=512, step=64)
                    seed_behavior = gr.Radio(["Iterate Per Image", "Iterate Per Row", "Fixed", "Random"], label="Seed Behavior", value="Iterate Per Image")

            # --- RIGHT COLUMN (CONTROLS & SETTINGS) ---
            with gr.Column(scale=1, variant="panel"):
                with gr.Accordion("Matrix Preview & Control", open=True):
                    axis_x_display = gr.Textbox(label="X-Axis", interactive=False)
                    axis_y_display = gr.Textbox(label="Y-Axis", interactive=False)
                    axis_page_display = gr.Textbox(label="Page-Axes", interactive=False)
                    random_axis_display = gr.Textbox(label="Random Axes (Not in Grid)", interactive=False)
                    swap_xy_btn = gr.Button("Swap X/Y")
                    cycle_page_btn = gr.Button("Cycle Page/Y/X")
                with gr.Accordion("Grid & File Settings", open=True):
                    show_annotations = gr.Checkbox(label="Show grid labels & titles", value=True)
                    margin_size = gr.Slider(label="Grid Margins (px)", minimum=0, maximum=200, value=10, step=2)
                    create_mega_grid_toggle = gr.Checkbox(label="Create \"Mega-Grid\"", value=True)
                    use_descriptive_filenames = gr.Checkbox(label="Descriptive filenames for grids", value=True)
                    auto_generate_grids = gr.Checkbox(label="Auto-generate grids after images are done", value=True)
                with gr.Accordion("Advanced & Help", open=False):
                    enable_dynamic_prompts = gr.Checkbox(label="Process with Dynamic Prompts extension", value=True)
                    gr.Markdown("### Syntax Guide\n- **Prompt Matrix:** `a photo of a |matrix:label|`\n- **Parameter Matrix:** `|Δ:cfg|`\n- **Random Matrix:** `wearing a |random:hat|`")

        # --- ACTION BUTTONS ---
        with gr.Row():
            preprocess_btn = gr.Button("1. Pre-process & Stage Job", variant="secondary", elem_id="upm_preprocess_btn")
            dry_run_btn = gr.Button("Dry Run to Console")
        calculation_results_display = gr.Markdown("Please Pre-process your job before generating.")
        with gr.Row():
            generate_btn = gr.Button("2. Generate Images", variant="primary", interactive=False)
            generate_grids_btn = gr.Button("3. Generate Grids", variant="secondary", interactive=False)

        # --- OUTPUTS ---
        with gr.Tabs():
            with gr.TabItem("All Images", id=0):
                gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="ultimate_matrix_gallery", columns=8, object_fit="contain", height="auto")
            with gr.TabItem("Summary Grids", id=1):
                summary_grid_display = gr.Image(type="pil", interactive=False, height=768)
                all_grids_state = gr.State([])
        html_info = gr.HTML()
        html_log = gr.HTML()

        # --- EVENT HANDLERS (How buttons and inputs talk to the code) ---
        all_ui_inputs = [prompt, negative_prompt, sampler_name, scheduler, steps, cfg_scale, width, height, seed_behavior, show_annotations, margin_size, create_mega_grid_toggle, use_descriptive_filenames, auto_generate_grids, enable_dynamic_prompts] + matrix_builder_inputs

        def on_input_change(*args):
            return {
                generate_btn: gr.update(interactive=False),
                generate_grids_btn: gr.update(interactive=False),
                calculation_results_display: "Settings changed. Please Pre-process again."
            }
        for component in all_ui_inputs:
            component.change(fn=on_input_change, outputs=[generate_btn, generate_grids_btn, calculation_results_display])

        # PHASE 1: Pre-process button click
        def on_preprocess_click(p_text, n_text, sb, s_name, s_ched, s_teps, c_fg, w, h, *matrix_builder_values):
            # 1. Collect Matrix Definitions from the UI
            matrix_defs = []
            for i in range(0, len(matrix_builder_values), 4):
                enabled, m_type, m_label, m_vars = matrix_builder_values[i:i+4]
                if enabled and m_label and m_vars:
                    matrix_defs.append({
                        "label": m_label.strip(),
                        "type": m_type,
                        "options": [opt.strip() for opt in m_vars.split(",")]
                    })

            # 2. Setup base objects
            base_params = {'sampler_name': s_name, 'scheduler': s_ched, 'steps': s_teps, 'cfg_scale': c_fg, 'width': w, 'height': h}
            p_base_obj = StableDiffusionProcessingTxt2Img(prompt=p_text, negative_prompt=n_text, seed=-1, **base_params)

            # 3. Call the new logic
            job_list, grid_info = build_job_list(p_base_obj, sb, matrix_defs)

            if not job_list:
                return "No valid matrix definitions or references found.", [], {}, None, {}, [], gr.update(interactive=False), gr.update(interactive=False), "N/A", "N/A", "N/A", "N/A"

            # 4. Populate Preview fields
            x_ax = grid_info.get('x_axis')
            y_ax = grid_info.get('y_axis')
            page_axes = grid_info.get('page_axes', [])
            all_grid_axes_labels = {ax['label'] for ax in (page_axes + ([y_ax] if y_ax else []) + ([x_ax] if x_ax else []))}
            random_axes = [d for d in matrix_defs if d['type'] == 'random' and d['label'] not in all_grid_axes_labels]

            x_display = x_ax['token'] if x_ax else "N/A"
            y_display = y_ax['token'] if y_ax else "N/A"
            page_display = ", ".join([ax['token'] for ax in page_axes]) if page_axes else "N/A"
            random_display = ", ".join([f"|random:{ax['label']}|" for ax in random_axes]) if random_axes else "N/A"

            result_text = f"**Staging Complete.** Total images to generate: **{len(job_list)}**."
            return result_text, job_list, grid_info, p_base_obj, base_params, matrix_defs, gr.update(interactive=True), gr.update(interactive=False), x_display, y_display, page_display, random_display

        preprocess_btn.click(
            fn=on_preprocess_click,
            inputs=[prompt, negative_prompt, seed_behavior, sampler_name, scheduler, steps, cfg_scale, width, height] + matrix_builder_inputs,
            outputs=[calculation_results_display, staged_job_list, staged_grid_info, staged_p_base, staged_base_params, staged_matrix_definitions, generate_btn, generate_grids_btn, axis_x_display, axis_y_display, axis_page_display, random_axis_display]
        )

        # Dry Run button click
        def on_dry_run_click(job_list):
            if not job_list:
                return "Pre-process first to see a dry run."
            log_message = "--- UPM Dry Run ---\n"
            for i, job in enumerate(job_list):
                log_message += f"Job {i+1}:\n  Prompt: {job['pos_prompt']}\n  Neg Prompt: {job['neg_prompt']}\n  Params: {job['params']}\n  Seed: {job['seed']}\n"
            print(log_message)
            return f"Dry run for {len(job_list)} jobs printed to console."
        
        dry_run_btn.click(fn=on_dry_run_click, inputs=[staged_job_list], outputs=[html_log])

        # PHASE 2: Generate Images button click
        def on_generate_click_wrapper(job_list, p_base_obj, base_params, dyn_prompts, auto_grids, margin, mega, desc_fn, show_anno, grid_info):
            yield {generate_btn: gr.update(value="Generating...", interactive=False), dry_run_btn: gr.update(interactive=False), generate_grids_btn: gr.update(interactive=False), html_log: "Starting image generation..."}
            images, infotexts = run_image_generation(job_list, base_params, dyn_prompts)
            if not images:
                yield {html_log: "Generation failed or was interrupted.", generate_btn: gr.update(value="Generate Images", interactive=True), dry_run_btn: gr.update(interactive=True)}
                return

            yield {html_log: "Image generation complete.", gallery: images, staged_images: images, staged_infotexts: infotexts}
            
            if auto_grids:
                yield {html_log: "Auto-generating grids..."}
                for update in on_generate_grids_click_wrapper(images, infotexts, grid_info, p_base_obj, margin, mega, desc_fn, show_anno):
                    yield update
            else:
                yield {generate_grids_btn: gr.update(interactive=True)}

        generate_btn.click(
            fn=on_generate_click_wrapper,
            inputs=[staged_job_list, staged_p_base, staged_base_params, enable_dynamic_prompts, auto_generate_grids, margin_size, create_mega_grid_toggle, use_descriptive_filenames, show_annotations, staged_grid_info],
            outputs=[generate_btn, dry_run_btn, generate_grids_btn, html_log, gallery, staged_images, staged_infotexts, summary_grid_display, all_grids_state]
        )

        # PHASE 3: Generate Grids button click
        def on_generate_grids_click_wrapper(images, infotexts, grid_info, p_base_obj, margin, mega, desc_fn, show_anno):
            if not images:
                yield {html_log: "No images were generated, so no grid can be created."}
                return
            yield {generate_grids_btn: gr.update(value="Creating Grids...", interactive=False)}
            all_grids, mega_grid = run_grid_generation(images, infotexts, grid_info, p_base_obj, margin, mega, desc_fn, show_anno)
            yield {
                summary_grid_display: gr.update(value=mega_grid if mega_grid else (all_grids[0] if all_grids else None), _js="(v) => { try { document.querySelector('#ultimate_matrix_gallery').parentNode.querySelector('.tabs > .tab-nav > button:nth-child(2)').click() } catch(e){} return v }"),
                all_grids_state: [mega_grid] + all_grids if mega_grid else all_grids,
                generate_grids_btn: gr.update(value="Generate Grids", interactive=True),
                html_log: "Grid generation complete."
            }

        generate_grids_btn.click(
            fn=on_generate_grids_click_wrapper,
            inputs=[staged_images, staged_infotexts, staged_grid_info, staged_p_base, margin_size, create_mega_grid_toggle, use_descriptive_filenames, show_annotations],
            outputs=[generate_grids_btn, summary_grid_display, all_grids_state, html_log]
        )

        # Simple UI handlers
        def on_gallery_select(evt: gr.SelectData, all_infotexts):
            """When a user clicks a thumbnail, this finds and returns the correct infotext."""
            if not all_infotexts or evt.index is None or evt.index >= len(all_infotexts):
                return ""
            return all_infotexts[evt.index]

        gallery.select(fn=on_gallery_select, inputs=[staged_infotexts], outputs=[html_info])
        clear_prompt_btn.click(fn=lambda: "", outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", outputs=[negative_prompt])
        paste_prompts_btn.click(fn=paste_last_prompts, outputs=[prompt, negative_prompt])

    return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

    # --- AXIS CONTROL HANDLERS ---
    def on_swap_click(p, n):
        return swap_tags_in_text(p), swap_tags_in_text(n)
        
    def on_cycle_click(p, n):
        return cycle_tags_in_text(p), cycle_tags_in_text(n)

    # When a button is clicked, it runs its function, updates the prompt boxes,
    # and then the .then() part automatically clicks the "Pre-process" button for you.
    swap_xy_btn.click(
        fn=on_swap_click,
        inputs=[prompt, negative_prompt],
        outputs=[prompt, negative_prompt]
    ).then(fn=None, _js="() => { document.getElementById('upm_preprocess_btn').click() }")

    cycle_page_btn.click(
        fn=on_cycle_click,
        inputs=[prompt, negative_prompt],
        outputs=[prompt, negative_prompt]
    ).then(fn=None, _js="() => { document.getElementById('upm_preprocess_btn').click() }")

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)