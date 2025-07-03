"""
Ultimate Prompt Matrix v15.0.0 (UI Scaffolding)
This version focuses exclusively on building the UI layout to match the flowchart.
Functionality is intentionally disconnected.
"""
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers, sd_schedulers

# --- CONSTANTS ---
MAX_MATRIX_BLOCKS = 5

# --- THE UI (Layout Only - Single Column Workflow) ---
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        
        # --- I. Global Settings, Presets, and Advanced (Top of Workflow) ---
        with gr.Accordion("Settings & Presets", open=False):
            # Check if Dynamic Prompts is installed
            try:
                from sd_dynamic_prompts.prompt_parser import parse
                dp_installed = True
                dp_tooltip = "Enable compatibility with the Dynamic Prompts extension."
            except ImportError:
                dp_installed = False
                dp_tooltip = "Dynamic Prompts extension not detected. Install it to use this feature. (https://github.com/adieyal/sd-dynamic-prompts)"

            # Row 1: Checkboxes
            with gr.Row():
                show_annotations = gr.Checkbox(label="Show Grid Labels", value=True)
                create_mega_grid_toggle = gr.Checkbox(label="Create Mega-Grid", value=True)
                use_descriptive_filenames = gr.Checkbox(label="Descriptive Filenames", value=True)
            
            # Row 2: Sliders and Toggles
            with gr.Row():
                enable_dynamic_prompts = gr.Checkbox(label="Enable Dynamic Prompts", value=False, interactive=dp_installed, info=dp_tooltip)
                enable_config_matrices = gr.Checkbox(label="Enable Parameter Matrices", value=True, info="Allows |Δ:...| axes. Note: These can only be used as Page axes (Z-axis) in the current Hybrid mode.")
            with gr.Row():
                 margin_size = gr.Slider(label="Grid Margins (px)", minimum=0, maximum=200, value=10, step=2)

            # Row 3: Preset Management
            with gr.Row():
                preset_dropdown = gr.Dropdown(label="Load Preset", choices=[""], scale=3)
                refresh_presets_btn = gr.Button("🔃", scale=1, elem_classes="tool_sm")
            with gr.Row():
                save_preset_name = gr.Textbox(label="Save as Preset", placeholder="Enter name...", scale=3)
                save_preset_btn = gr.Button("💾 Save", scale=1)

        # --- II. Core Input & Matrix Definition ---
        
        # 1. Generation Settings (Condensed)
        gr.Markdown("### Generation Settings")
        with gr.Row():
            sampler_name = gr.Dropdown(label='Sampling method', choices=[s.name for s in sd_samplers.samplers], value="Euler a", scale=2)
            scheduler = gr.Dropdown(label='Schedule type', choices=[s.label for s in sd_schedulers.schedulers], value="Automatic", scale=2)
            steps = gr.Slider(label='Steps', minimum=1, maximum=150, value=20, step=1, scale=3)
            cfg_scale = gr.Slider(label='CFG Scale', minimum=1.0, maximum=30.0, value=7.0, step=0.5, scale=3)
        with gr.Row():
            width = gr.Slider(label='Width', minimum=64, maximum=2048, value=512, step=64, scale=1)
            height = gr.Slider(label='Height', minimum=64, maximum=2048, value=512, step=64, scale=1)
            # For now, we will use the simple Radio buttons for seed behavior.
            # The "smart textbox" is a great idea, but is a complex feature to add later.
            seed_behavior = gr.Radio(["Iterate Per Image", "Iterate Per Row", "Fixed", "Random"], label="Seed Behavior", value="Iterate Per Image", scale=2)
            # We will also need a place for the user to input the starting seed.
            seed = gr.Number(label="Start Seed", value=-1, precision=0, interactive=True)
        
        # 2. Static LoRA/Embedding Section
        with gr.Accordion("Static LoRA/Embedding", open=False):
            gr.Markdown("These are applied globally to every image generated, prepended to the main prompt.")
            static_loras_textbox = gr.Textbox(label="Static LoRAs", placeholder="<lora:name:1.0>, <lora:style:0.7>...")
            static_embeddings_textbox = gr.Textbox(label="Static Embeddings", placeholder="embedding_A, embedding_B...")

        # 3. Prompt & Negative Prompt Boxes
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter prompts with |matrix:label| syntax here...", elem_id="upm_prompt")
        with gr.Row():
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Enter negative prompts here...", elem_id="upm_negative_prompt")

        # 4. Matrix Builder
        with gr.Accordion("Matrix Builder", open=True):
            gr.Markdown("Define your matrix axes here. Remember: **|Δ:param|** axes must come before **|matrix:prompt|** axes for live previews to work.")
            matrix_builder_rows = []
            for i in range(MAX_MATRIX_BLOCKS):
                with gr.Row(visible=(i < 2)) as row:
                    enabled = gr.Checkbox(label="On", value=(i == 0))
                    m_type = gr.Dropdown(["matrix", "Δ", "random"], value="matrix", label="Type")
                    m_label = gr.Textbox(placeholder="e.g., animal, cfg, style", label="Label")
                    m_vars = gr.Textbox(placeholder="Comma-separated values", label="Variables", lines=1)
                    matrix_builder_rows.append(row)
            with gr.Row():
                matrix_row_slider = gr.Slider(label="Visible Matrix Rows", minimum=1, maximum=MAX_MATRIX_BLOCKS, value=2, step=1)
        
        # --- III. Pre-process Controls (Primary Interaction Point) ---
        with gr.Row():
            preprocess_btn = gr.Button("1. Pre-process & Stage Job", variant="secondary", elem_id="upm_preprocess_btn")
            dry_run_btn = gr.Button("Dry Run to Console")
        calculation_results_display = gr.Markdown("Status: Ready")

        # --- IV. Preview & Auxiliary Tools ---
        with gr.Accordion("Matrix Preview & Control", open=True):
            axis_x_display = gr.Textbox(label="X-Axis", interactive=False)
            axis_y_display = gr.Textbox(label="Y-Axis", interactive=False)
            axis_page_display = gr.Textbox(label="Page-Axes", interactive=False)
            random_axis_display = gr.Textbox(label="Random Axes (Not in Grid)", interactive=False)
            swap_xy_btn = gr.Button("Swap X/Y")
            cycle_page_btn = gr.Button("Cycle Page/Y/X")

        # --- V. Action Buttons & Output ---
        gr.Markdown("### Generation")
        with gr.Row():
            generate_btn = gr.Button("2. Generate Images", variant="primary", interactive=False)
            generate_grids_btn = gr.Button("3. Generate Grids", variant="secondary", interactive=False)
        
        auto_generate_grids = gr.Checkbox(label="Auto-generate grids after images are done", value=True)

        with gr.Tabs():
            with gr.TabItem("All Images", id=0): 
                # Set interactive=False to prevent the "Drop Image" prompt
                gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="ultimate_matrix_gallery", columns=4, object_fit="contain", height="auto", interactive=False)
            with gr.TabItem("Summary Grids", id=1):
                summary_grid_display = gr.Image(type="pil", interactive=False, height=768)
        
        html_info = gr.HTML()
        html_log = gr.HTML()

        # --- EVENT HANDLERS (Still disconnected, but placed correctly) ---
        def on_matrix_row_slider_change(num_rows):
            updates = {}
            for i, row in enumerate(matrix_builder_rows):
                updates[row] = gr.update(visible=(i < num_rows))
            return updates
        matrix_row_slider.change(fn=on_matrix_row_slider_change, inputs=matrix_row_slider, outputs=matrix_builder_rows)
        
        return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

# This is the function that A1111 calls to add the tab to the UI.
scripts.script_callbacks.on_ui_tabs(on_ui_tabs)