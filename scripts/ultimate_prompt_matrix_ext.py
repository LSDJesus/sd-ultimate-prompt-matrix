"""
Ultimate Prompt Matrix v15.1.7
Author: LSDJesus
Changes:
- Overhauled "Matrix Preview & Control" section:
    - Replaced text-based axis displays with a single graphical preview image.
    - Removed the "Cycle Page/Y/X" button to simplify UI and prevent axis-type errors.
    - The "Swap X/Y" button now triggers a redraw of the graphical preview.
- Integrated `draw_preview_grid` function from `upm_grid_drawing` to generate the preview.
- Updated `on_preprocess_click` event handler to call the new drawing function and update the preview image.
"""
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers, sd_schedulers, shared, sd_models
import modules.sd_models # For LoRA/Embedding dropdowns

# --- CRITICAL FIX: Ensure script's directory is in sys.path for local imports ---
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
# --- END CRITICAL FIX ---

import upm_wildcard_handler # For wildcard dropdowns
import upm_utils # For paste_last_prompts
import upm_grid_drawing # Added for the graphical preview

# --- CONSTANTS ---
MAX_MATRIX_BLOCKS = 5 # Used for both top-level and LoRA sub-blocks for now

# --- Helper for Generation Settings Dropdown ---
# This list will be used for the "Setting Type" dropdown
GENERATION_SETTINGS_CHOICES = [
    "cfg_scale", "steps", "size", "sampler_name", "scheduler",
    "seed", "denoising_strength", "clip_skip", "eta", "seed_resize_from_w",
    "seed_resize_from_h", "tiling", "batch_size", "restore_faces" # Add more as needed
]

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
            seed_behavior = gr.Radio(["Iterate Per Image", "Iterate Per Row", "Fixed", "Random"], label="Seed Behavior", value="Iterate Per Image", scale=2)
            seed = gr.Number(label="Start Seed", value=-1, precision=0, interactive=True)
        
        # 2. Static LoRA/Embedding Section (OVERHAULED)
        with gr.Accordion("Static LoRA/Embedding", open=False):
            gr.Markdown("These are applied globally to every image generated, prepended to the main prompt.")
            
            with gr.Row():
                static_embeddings_dropdown = gr.Dropdown(
                    label="Embeddings", 
                    choices=[], # Initialized empty, populated by refresh
                    multiselect=True, 
                    scale=3
                )
                add_selected_embeddings_btn = gr.Button("Add Selected Embeddings", scale=1)
                
                static_loras_dropdown = gr.Dropdown(
                    label="LoRAs", 
                    choices=[], # Initialized empty, populated by refresh
                    multiselect=True, 
                    scale=3
                )
                add_selected_loras_btn = gr.Button("Add Selected LoRAs", scale=1)

            with gr.Row():
                current_embeddings_textbox = gr.Textbox(label="Embeddings:", placeholder="embedding_A, embedding_B...", interactive=True)
                clear_embeddings_btn = gr.Button("🗑️", elem_classes="tool_sm", min_width=50) # Added min_width for consistency
                
                current_loras_textbox = gr.Textbox(label="LoRAs:", placeholder="<lora:name:1.0>, <lora:style:0.7>...", interactive=True)
                clear_loras_btn = gr.Button("🗑️", elem_classes="tool_sm", min_width=50)
            
            refresh_static_extras_btn = gr.Button("🔃 Refresh LoRAs/Embeddings")

        # 3. Prompt & Negative Prompt Boxes (OVERHAULED with buttons)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter prompts with |matrix:label| syntax here...", elem_id="upm_prompt")
            with gr.Column(scale=0, min_width=50): # Column to stack buttons
                clear_prompt_btn = gr.Button("🗑️", elem_classes="tool_sm")
                paste_prompts_btn = gr.Button("↙️", elem_classes="tool_sm", tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.")
        with gr.Row():
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Enter negative prompts here...", elem_id="upm_negative_prompt")
            with gr.Column(scale=0, min_width=50):
                clear_neg_prompt_btn = gr.Button("🗑️", elem_classes="tool_sm")

        # 4. Matrix Builder (OVERHAULED)
        with gr.Accordion("Matrix Builder", open=True):
            gr.Markdown("Define your matrix axes here. *Number of generated images increases multiplicatively with number of variables per matrix and exponentially with dimensions.*")
            
            # State to manage the number of active matrix blocks (for advanced mode)
            num_active_matrix_blocks = gr.State(value=1) 
            
            # New Radio Buttons for Builder Mode (1D, 2D, 3D, 3D+Advanced) - Changed to Radio
            builder_mode = gr.Radio( # Changed from gr.Dropdown to gr.Radio
                ["1D Axis", "2D Grid", "3D Grid", "3D+ (Advanced)"],
                label="Builder Mode",
                value="1D Axis" # Default selection
            )

            # New Markdown for layout description (updates dynamically)
            builder_layout_description = gr.Markdown("Current Layout: **1D Axis** (A single list of images, defined by the axis below).")

            # Lists to hold dynamically created components for event binding
            matrix_block_uis = [] 
            matrix_type_dropdowns = [] 
            matrix_label_textboxes = [] 
            matrix_variable_groups = [] 
            matrix_setting_type_dropdowns = [] 
            matrix_wildcard_name_dropdowns = []
            all_lora_sub_managers = []
            all_embedding_sub_managers = []
            std_insert_btns = [] # New list for standard insert buttons
            rand_insert_btns = [] # New list for random insert buttons
            wc_insert_btns = [] # New list for wildcard insert buttons
            setting_insert_btns = [] # New list for setting insert buttons 
            
            # Dynamically create MAX_MATRIX_BLOCKS (5) matrix definition rows
            for i in range(MAX_MATRIX_BLOCKS):
                with gr.Group(visible=(i == 0)) as matrix_block: 
                    with gr.Row():
                        # The 'On' checkbox has been removed
                        m_type_dropdown = gr.Dropdown(
                            ["Standard", "Random", "Wildcard", "LoRA", "Embedding", "Generation Setting", "Model"], 
                            value="Standard", 
                            label=f"Axis {i+1} Type", # Added index for clarity
                            scale=1, 
                            elem_id=f"matrix_type_dd_{i}" 
                        )
                        m_label_textbox = gr.Textbox(
                            placeholder="e.g., animal, cfg, style", 
                            label="Reference Label", 
                            scale=2, 
                            elem_id=f"matrix_label_tb_{i}"
                        )
                    
                    # Conditional input areas for variables based on type (nested directly inside the matrix_block)
                    with gr.Column(visible=True) as standard_group: 
                        with gr.Row():
                            standard_vars_textbox = gr.Textbox(placeholder="e.g., cat, dog, bird", label="Variables", scale=3)
                            std_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|Name/title|>", scale=2)
                            std_insert_btn = gr.Button("Insert Shortcut", scale=1) # Renamed
                            std_insert_btns.append(std_insert_btn)

                    with gr.Column(visible=False) as random_group:
                        with gr.Row():
                            random_vars_textbox = gr.Textbox(placeholder="e.g., hat, scarf, boots", label="Variables", scale=3)
                            rand_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|~Name/Title~|>", scale=2)
                            rand_insert_btn = gr.Button("Insert Shortcut", scale=1) # Renamed
                            rand_insert_btns.append(rand_insert_btn)
                    
                    with gr.Column(visible=False) as setting_group:
                        with gr.Row():
                            setting_type_dd = gr.Dropdown( # Changed from setting_type_dropdown to setting_type_dd for consistent naming
                                label="Setting Type",
                                choices=GENERATION_SETTINGS_CHOICES,
                                value="cfg_scale",
                                scale=1, 
                                elem_id=f"setting_type_dd_{i}"
                            )
                            setting_vars_textbox = gr.Textbox(
                                placeholder="e.g., 7.0, 9.0 (for CFG) or 512x768, 768x768 (for Size)", 
                                label="Variables (comma-separated)", 
                                lines=1, 
                                scale=2, 
                                elem_id=f"setting_vars_tb_{i}"
                            )
                            setting_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |Δ:CFG_Values| or |Δ:cfg_scale|", scale=2)
                            setting_insert_btn = gr.Button("Insert Shortcut to Prompt", scale=1)
                    
                    with gr.Column(visible=False) as wildcard_group:
                        with gr.Row():
                            wildcard_name_dropdown = gr.Dropdown(
                                label="Wildcard Name",
                                choices=upm_wildcard_handler.get_wildcard_files(), # Dynamically load wildcard files
                                value=None,
                                elem_id=f"wildcard_dropdown_{i}",
                                scale=2
                            )
                            wc_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|__wildcard__|>", scale=2)
                            wc_insert_btn = gr.Button("Insert Shortcut to Prompt", scale=1)
                    
                    with gr.Column(visible=False) as lora_group:
                        with gr.Row():
                            lora_group_label_name = gr.Textbox(label="Name/Title (Optional)", placeholder="e.g., LoRA_Styles", scale=3)
                            add_lora_matrix_block_btn = gr.Button("Add LoRA/Weight Block", elem_id=f"add_lora_block_btn_{i}", scale=1) # Renamed to differentiate from top-level add button
                        gr.Markdown("Define LoRA combinations for this matrix label:")
                        num_active_lora_sub_blocks = gr.State(value=0) # State for managing sub-blocks within this specific LoRA matrix block
                        
                        lora_sub_block_containers = [] 
                        lora_sub_block_dropdowns = []
                        lora_sub_block_weights = []

                        for j in range(MAX_MATRIX_BLOCKS): # Max 5 sub-blocks per LoRA matrix (using MAX_MATRIX_BLOCKS as a convenient limit)
                            with gr.Row(visible=(j==0)) as lora_sub_block_row: # First sub-block visible by default
                                lora_matrix_dropdown = gr.Dropdown(
                                    label=f"LoRA {j+1}", 
                                    choices=[], # Initialized empty, populated by refresh
                                    multiselect=False, 
                                    scale=2,
                                    elem_id=f"lora_matrix_dd_{i}_{j}"
                                ) 
                                lora_matrix_weight_slider = gr.Slider(
                                    label="Weight", 
                                    minimum=-2.0, maximum=2.0, value=1.0, step=0.05, 
                                    scale=1,
                                    elem_id=f"lora_matrix_weight_sl_{i}_{j}"
                                )
                            lora_sub_block_containers.append(lora_sub_block_row)
                            lora_sub_block_dropdowns.append(lora_matrix_dropdown)
                            lora_sub_block_weights.append(lora_matrix_weight_slider)

                        gr.Markdown("Note: LoRAs will be appended to the end of the prompt for each image.")
                        all_lora_sub_managers.append({
                            "num_active_state": num_active_lora_sub_blocks,
                            "sub_block_containers": lora_sub_block_containers,
                            "sub_block_dropdowns": lora_sub_block_dropdowns, 
                            "add_button": add_lora_matrix_block_btn,
                            "group_label_name": lora_group_label_name 
                        })
                    
                    with gr.Column(visible=False) as embedding_group:
                        with gr.Row():
                            embedding_group_label_name = gr.Textbox(label="Name/Title (Optional)", placeholder="e.g., Embedding_Details", scale=3)
                            add_embedding_matrix_block_btn = gr.Button("Add Embedding/Weight Block", elem_id=f"add_embedding_block_btn_{i}", scale=1)
                        gr.Markdown("Define Embedding combinations for this matrix label:")
                        num_active_embedding_sub_blocks = gr.State(value=0)
                        
                        embedding_sub_block_containers = []
                        embedding_sub_block_dropdowns = []
                        embedding_sub_block_weights = []

                        for j in range(MAX_MATRIX_BLOCKS):
                            with gr.Row(visible=(j==0)) as embedding_sub_block_row:
                                embedding_matrix_dropdown = gr.Dropdown(
                                    label=f"Embedding {j+1}",
                                    choices=[], # Initialized empty, populated by refresh
                                    multiselect=False,
                                    scale=2
                                )
                                embedding_matrix_weight_slider = gr.Slider(
                                    label="Weight", minimum=-2.0, maximum=2.0, value=1.0, step=0.05, scale=1
                                )
                            embedding_sub_block_containers.append(embedding_sub_block_row)
                            embedding_sub_block_dropdowns.append(embedding_matrix_dropdown)
                            embedding_sub_block_weights.append(embedding_matrix_weight_slider)

                        gr.Markdown("Note: Embeddings will be appended to the beginning of the prompt for each image.")
                        # This should be a separate list/manager from lora, but reusing for structure
                        all_lora_sub_managers.append({
                            "num_active_state": num_active_embedding_sub_blocks,
                            "sub_block_containers": embedding_sub_block_containers,
                            "sub_block_dropdowns": embedding_sub_block_dropdowns, 
                            "add_button": add_embedding_matrix_block_btn,
                            "group_label_name": embedding_group_label_name 
                        })

                    with gr.Column(visible=False) as model_group:
                        with gr.Row():
                            model_name_dropdown = gr.Dropdown(
                                label="Model", 
                                choices=sd_models.checkpoints_list, # CORRECTED LINE 
                                multiselect=True,
                                scale=4
                            )
                            # Custom rendering for multi-select dropdown to mimic VAE selector
                            model_name_dropdown.elem_classes = ["multiselect-vae"]
                            model_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|model:ModelName|>", scale=2)
                            model_insert_btn = gr.Button("Insert Shortcut to Prompt", scale=1)
                        gr.Markdown("Note: Model changes are only available in 3D or 3D+ modes and will act as a Page (Z) axis.")

                # Collect components for event binding outside the loop
                matrix_block_uis.append(matrix_block)
                matrix_type_dropdowns.append(m_type_dropdown)
                matrix_label_textboxes.append(m_label_textbox)
                matrix_variable_groups.append({
                    "standard": standard_group,
                    "random": random_group,
                    "setting": setting_group,
                    "wildcard": wildcard_group,
                    "lora": lora_group,
                    "embedding": embedding_group,
                    "model": model_group
                })
                matrix_setting_type_dropdowns.append(setting_type_dd)
                matrix_wildcard_name_dropdowns.append(wildcard_name_dropdown)

            # Button to add new top-level matrix blocks (only visible in 3D+ Advanced mode)
            add_new_matrix_block_btn = gr.Button("Add New Matrix", elem_id="add_new_matrix_block_btn", visible=False)


        # --- III. Pre-process Controls (Primary Interaction Point) ---
        with gr.Row():
            # Placeholder for dry run/matrix preview/sandbox toggles as per flowchart
            # For now, just the pre-process and dry-run buttons
            preprocess_btn = gr.Button("1. Pre-process & Stage Job", variant="secondary", elem_id="upm_preprocess_btn")
            dry_run_btn = gr.Button("Dry Run to Console")
        calculation_results_display = gr.Markdown("Status: Ready")

        # --- IV. Preview & Auxiliary Tools ---
        with gr.Accordion("Matrix Preview & Control", open=True):
            # The textboxes have been replaced by a single gr.Image component
            matrix_preview_display = gr.Image(label="Grid Preview", interactive=False, show_label=False)
            
            # Keep the Swap X/Y button, but remove the Cycle button
            with gr.Row():
                swap_xy_btn = gr.Button("Swap X/Y")
            
            # Hidden state components to hold the staged job data
            staged_job_list = gr.State([])
            staged_grid_info = gr.State({})

        # --- V. Action Buttons & Output (Bottom Section) ---
        gr.Markdown("### Generation")
        with gr.Row():
            generate_btn = gr.Button("2. Generate Images", variant="primary", interactive=False)
            generate_grids_btn = gr.Button("3. Generate Grids", variant="secondary", interactive=False)
        
        auto_generate_grids = gr.Checkbox(label="Auto-generate grids after images are done", value=True)

        with gr.Tabs():
            with gr.TabItem("All Images", id=0): 
                gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="ultimate_matrix_gallery", columns=4, object_fit="contain", height="auto", interactive=False)
            with gr.TabItem("Summary Grids", id=1):
                summary_grid_display = gr.Image(type="pil", interactive=False, height=768)
        
        html_info = gr.HTML()
        html_log = gr.HTML()

        # --- EVENT HANDLERS ---

        # 1. Static LoRA/Embedding Section Handlers
        def on_add_embeddings(selected_embeddings_list, current_text_box_value):
            new_embeddings = [name.replace(".pt", "") for name in selected_embeddings_list]
            existing_embeddings = [e.strip() for e in current_text_box_value.split(',') if e.strip()]
            combined = sorted(list(set(existing_embeddings + new_embeddings)))
            return ", ".join(combined)

        def on_add_loras(selected_loras_list, current_text_box_value):
            new_loras = [f"<lora:{lora.replace('.safetensors', '').replace('.pt', '')}:1.0>" for lora in selected_loras_list]
            existing_loras = [l.strip() for l in current_text_box_value.split(',') if l.strip()]
            combined = sorted(list(set(existing_loras + new_loras)))
            return ", ".join(combined)

        add_selected_embeddings_btn.click(
            fn=on_add_embeddings,
            inputs=[static_embeddings_dropdown, current_embeddings_textbox],
            outputs=[current_embeddings_textbox]
        )
        clear_embeddings_btn.click(fn=lambda: "", outputs=[current_embeddings_textbox])

        add_selected_loras_btn.click(
            fn=on_add_loras,
            inputs=[static_loras_dropdown, current_loras_textbox],
            outputs=[current_loras_textbox]
        )
        clear_loras_btn.click(fn=lambda: "", outputs=[current_loras_textbox])

        def refresh_static_extras_choices():
            embeddings_choices = list(shared.sd_embeddings.embeddings.keys()) if hasattr(shared, 'sd_embeddings') and shared.sd_embeddings else []
            lora_choices = [lora.name for lora in sd_models.loras] if hasattr(sd_models, 'loras') else []
            model_choices = [x.name for x in sd_models.checkpoints_list]
            
            updates = []
            updates.append(gr.update(choices=embeddings_choices)) # static_embeddings_dropdown
            updates.append(gr.update(choices=lora_choices)) # static_loras_dropdown

            for manager in all_lora_sub_managers:
                for item_dd in manager["sub_block_dropdowns"]:
                     updates.append(gr.update(choices=lora_choices))

            # Update Model dropdowns in builder
            for dd in matrix_type_dropdowns:
                # Assuming 'model_group' contains a dropdown at a known position
                # This part needs to be more robust.
                # Let's find the model dropdown by its specific object if possible.
                # For now, we're iterating and assuming structure.
                # model_dd = matrix_variable_groups[i]["model"].children[0].children[0]
                # updates.append(gr.update(choices=model_choices, component=model_dd))
                pass

            return tuple(updates)

        refresh_static_extras_btn.click(
            fn=refresh_static_extras_choices,
            outputs=[
                static_embeddings_dropdown, static_loras_dropdown,
                *[dd for manager in all_lora_sub_managers for dd in manager["sub_block_dropdowns"]]
                # Add model dropdowns here when they're uniquely identifiable
            ]
        )

        # 2. Prompt & Negative Prompt Button Handlers
        clear_prompt_btn.click(fn=lambda: "", outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", outputs=[negative_prompt])
        paste_prompts_btn.click(
            fn=upm_utils.paste_last_prompts,
            outputs=[
                prompt,
                negative_prompt,
                steps,
                sampler_name,
                scheduler,
                cfg_scale,
                seed,
                width,
                height
            ]
        )

        # 3. Matrix Builder Handlers
        
        # --- Builder Mode and Layout Description Handler ---
        def on_builder_mode_change(mode):
            num_visible = 1
            layout_desc = "Current Layout: **1D Axis** (A single list of images, defined by the axis below)."
            add_btn_visible = False
            add_btn_interactive = False
            num_active_state = 1

            if mode == "2D Grid":
                num_visible = 2
                layout_desc = "Current Layout: **2D Grid**. Axis 1 is X, Axis 2 is Y."
            elif mode == "3D Grid":
                num_visible = 3
                layout_desc = "Current Layout: **3D Grid**. Axis 1 is X, Axis 2 is Y, Axis 3 is Page."
            elif mode == "3D+ (Advanced)":
                num_visible = 4  # Default to 4 visible rows for advanced
                num_active_state = 4
                description = f"Current Layout: **3D+ Advanced Grid**. {num_visible} axes defined. The first is X, the second is Y, and subsequent axes are Page axes."
                layout_desc = description
                add_btn_visible = True
                add_btn_interactive = num_visible < MAX_MATRIX_BLOCKS
            
            updates = {
                builder_layout_description: gr.update(value=layout_desc),
                add_new_matrix_block_btn: gr.update(visible=add_btn_visible, interactive=add_btn_interactive),
                num_active_matrix_blocks: num_active_state
            }

            for i, block in enumerate(matrix_block_uis):
                updates[block] = gr.update(visible=(i < num_visible))

            # Dynamically show/hide "Model" option in Axis Type dropdowns
            choices = ["Standard", "Random", "Wildcard", "LoRA", "Embedding", "Generation Setting"]
            if mode in ["3D Grid", "3D+ (Advanced)"]:
                choices.append("Model")
            
            for dd in matrix_type_dropdowns:
                updates[dd] = gr.update(choices=choices)
            
            return updates
        
        builder_mode.change(
            fn=on_builder_mode_change,
            inputs=[builder_mode],
            outputs=[builder_layout_description, add_new_matrix_block_btn, num_active_matrix_blocks] + matrix_block_uis + matrix_type_dropdowns
        )

        def on_add_new_matrix_block(current_num_active):
            new_num = min(current_num_active + 1, MAX_MATRIX_BLOCKS)
            updates = {
                num_active_matrix_blocks: new_num,
                add_new_matrix_block_btn: gr.update(interactive=(new_num < MAX_MATRIX_BLOCKS))
            }
            for i, block in enumerate(matrix_block_uis):
                updates[block] = gr.update(visible=(i < new_num))

            return updates

        add_new_matrix_block_btn.click(
            fn=on_add_new_matrix_block,
            inputs=[num_active_matrix_blocks],
            outputs=[num_active_matrix_blocks, add_new_matrix_block_btn] + matrix_block_uis
        )

        # Handler for each Matrix Type Dropdown (for all dynamic matrix blocks)
        for i, m_type_dd in enumerate(matrix_type_dropdowns): 
            def _on_matrix_type_change_closure(selected_type, current_index=i):
                groups = matrix_variable_groups[current_index]
                updates = {
                    groups["standard"]: gr.update(visible=(selected_type == "Standard")),
                    groups["random"]: gr.update(visible=(selected_type == "Random")),
                    groups["setting"]: gr.update(visible=(selected_type == "Setting")),
                    groups["wildcard"]: gr.update(visible=(selected_type == "Wildcard")),
                    groups["lora"]: gr.update(visible=(selected_type == "LoRA")),
                    groups["embedding"]: gr.update(visible=(selected_type == "Embedding")),
                    groups["model"]: gr.update(visible=(selected_type == "Model"))
                }
                return updates
            
            m_type_dd.change(
                fn=_on_matrix_type_change_closure,
                inputs=[m_type_dd], 
                outputs=list(matrix_variable_groups[i].values())
            )
        # 4. Pre-process & Preview Handlers
        def on_preprocess_click(*args):
            # This function will eventually call the core logic to build the job list.
            # For now, it will just generate a dummy preview based on mock data.
            
            # In the final version, this will be replaced by:
            # job_list, grid_info = upm_logic.build_job_list(...)
            
            # --- MOCK DATA for UI DEMO ---
            mock_grid_info = {
                "x_axis": {"options": ["Style A", "Style B", "Style C"]},
                "y_axis": {"options": ["Cat", "Dog"]},
                "page_axes": [
                    {"label": "cfg_scale", "options": ["7.0", "9.0"]},
                    {"label": "sampler_name", "options": ["Euler a", "DPM++"]}
                ]
            }
            # --- END MOCK DATA ---

            x_labels = mock_grid_info.get("x_axis", {}).get("options", [])
            y_labels = mock_grid_info.get("y_axis", {}).get("options", [])
            page_labels = [f"{ax['label']}: {len(ax['options'])} values" for ax in mock_grid_info.get("page_axes", [])]

            preview_image = upm_grid_drawing.draw_preview_grid(x_labels, y_labels, page_labels)
            
            # Status update
            # This will eventually show the real image count and ETA
            num_images = (len(x_labels) or 1) * (len(y_labels) or 1) * (len(page_labels) or 1) # Simplified calc
            status_text = f"Status: Staged {num_images} images. Ready to generate."
            
            return {
                matrix_preview_display: gr.update(value=preview_image),
                calculation_results_display: gr.update(value=status_text),
                generate_btn: gr.update(interactive=True)
            }
        
        # We need to collect ALL inputs for the pre-process step eventually.
        # For now, it takes no inputs since we are using mock data.
        preprocess_btn.click(
            fn=on_preprocess_click,
            inputs=[], # This will be a long list of all UI components later
            outputs=[
                matrix_preview_display,
                calculation_results_display,
                generate_btn
            ]
        )

        # Placeholder for swap_xy_btn click handler
        # This will eventually swap the staged_grid_info and redraw the preview
        swap_xy_btn.click(
            fn=lambda: gr.Info("Swap X/Y button clicked. (Functionality to be implemented)"),
            outputs=None
        )
        # --- Other handlers from previous versions to be integrated/re-wired ---
        # ...

        return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)