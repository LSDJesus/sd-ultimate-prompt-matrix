"""
Ultimate Prompt Matrix v15.1.4
Author: LSDJesus
Changes:
- FIXED: `refresh_static_extras_btn` now correctly populates dropdowns by returning updates in a tuple.
- FIXED: "Paste Last Prompts" button now correctly populates all generation parameters (steps, sampler, cfg, seed, size) by returning a tuple of updates directly from `upm_utils`.
- Overhauled "Matrix Builder" mode selection:
    - Changed main "Builder Mode" from `gr.Radio` to `gr.Dropdown`.
- Refactored "1D Axis" builder UI layout:
    - Arranged input fields (Name/Title, Variables, Syntax Display, Insert Button) on single rows for "Standard", "Random", and "Generation Setting" types.
    - Updated Syntax Shortcut placeholders for "Standard", "Random", and "Wildcard" to match new desired formats (<|...|>, <|~...~|>, <|__...__|>).
    - Removed "Syntax Shortcut" display and "Insert Shortcut to Prompt" button for "LoRA" and "Embedding" 1D axis types, as per new design.
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

# --- CONSTANTS ---
MAX_MATRIX_BLOCKS = 5 # Used for both top-level and LoRA sub-blocks for now

# --- Helper for Generation Settings Dropdown ---
# This list will be used for the "Setting Type" dropdown
# Values correspond to parameter names in processing.py where applicable.
# 'size' is a special combined string.
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
            # Moved and updated introductory markdown text
            gr.Markdown("Define your matrix axes here. *Number of generated images increases multiplicatively with number of variables per matrix and exponentially with dimensions.*")
            
            # State to manage the number of active matrix blocks (for advanced mode)
            num_active_matrix_blocks = gr.State(value=1) 
            
            matrix_block_uis = [] # To store the gr.Group for each matrix block (for advanced mode)
            matrix_type_dropdowns = [] # To store the type dropdowns for event handling (for advanced mode)
            matrix_label_textboxes = [] # To store the label textboxes for event handling (for advanced mode)
            insert_tag_buttons = [] # To store the insert tag buttons for event handling (for advanced mode)
            matrix_variable_groups = [] # To store the conditional variable input groups for each block (for advanced mode)
            matrix_setting_type_dropdowns = [] # To store setting type dropdowns for insert tag fn (for advanced mode)
            matrix_wildcard_name_dropdowns = [] # To store wildcard name dropdowns for insert tag fn (for advanced mode)
            all_lora_sub_managers = [] # To store info for managing LoRA sub-blocks (for advanced mode)

            # New Radio Buttons for Builder Mode (1D, 2D, 3D, 3D+Advanced) - Changed to Radio
            builder_mode = gr.Radio( # Changed from gr.Dropdown to gr.Radio
                ["1D Axis", "2D Grid", "3D Grid", "3D+ (Advanced)"],
                label="Builder Mode",
                value="1D Axis" # Default selection
            )

            # New Markdown for layout description (updates dynamically)
            builder_layout_description = gr.Markdown("Current Layout: **1D Axis** (1 row with X variables)")

            # Dynamically create MAX_MATRIX_BLOCKS (5) matrix definition rows
            for i in range(MAX_MATRIX_BLOCKS):
                # Only the first one is visible by default (for 1D Axis mode)
                # Others become visible based on builder_mode or add_new_matrix_block_btn
                with gr.Group(visible=(i == 0)) as matrix_block: 
                    with gr.Row():
                        enabled_checkbox = gr.Checkbox(label="On", value=(i == 0), scale=0) 
                        m_type_dropdown = gr.Dropdown( # Now a simple dropdown for 'Axis Type'
                            ["Standard", "Random", "Wildcard", "LoRA", "Embedding", "Generation Setting"], 
                            value="Standard", 
                            label="Axis Type", # Changed label from 'Type'
                            scale=1, 
                            elem_id=f"matrix_type_dd_{i}" 
                        )
                        m_label_textbox = gr.Textbox(
                            placeholder="e.g., animal, cfg, style", 
                            label="Reference Label", 
                            scale=2, 
                            elem_id=f"matrix_label_tb_{i}"
                        )
                        insert_tag_btn = gr.Button("Insert Tag to Prompt", scale=1) 
                    
                    # Conditional input areas for variables based on type (nested directly inside the matrix_block)
                    with gr.Column(visible=True) as standard_group: # Default visible for "Standard"
                        with gr.Row():
                            standard_vars_textbox = gr.Textbox(
                                placeholder="e.g., cat, dog, bird", 
                                label="Variables (comma-separated)", 
                                lines=1, 
                                elem_id=f"standard_vars_tb_{i}",
                                scale=4
                            )
                            std_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|Name/title|>", scale=2) # Updated placeholder
                            std_insert_btn = gr.Button("Insert Shortcut to Prompt", scale=1)

                    with gr.Column(visible=False) as random_group:
                        with gr.Row():
                            random_vars_textbox = gr.Textbox(
                                placeholder="e.g., hat, scarf, boots", 
                                label="Variables (comma-separated)", 
                                lines=1, 
                                elem_id=f"random_vars_tb_{i}",
                                scale=4
                            )
                            rand_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="<|~Name/Title~|>", scale=2) # Updated placeholder
                            rand_insert_btn = gr.Button("Insert Shortcut to Prompt", scale=1)
                    
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
                        # Store data for managing LoRA sub-blocks for this matrix block
                        all_lora_sub_managers.append({
                            "num_active_state": num_active_lora_sub_blocks,
                            "sub_block_containers": lora_sub_block_containers,
                            "sub_block_dropdowns": lora_sub_block_dropdowns, # Store dropdowns for refresh
                            "add_button": add_lora_matrix_block_btn,
                            "group_label_name": lora_group_label_name # Store the group's label for event handler
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
                        # Store data for managing Embedding sub-blocks for this matrix block
                        all_lora_sub_managers.append({ # Reusing this list, but should be all_embedding_sub_managers if separate
                            "num_active_state": num_active_embedding_sub_blocks,
                            "sub_block_containers": embedding_sub_block_containers,
                            "sub_block_dropdowns": embedding_sub_block_dropdowns, # Store dropdowns for refresh
                            "add_button": add_embedding_matrix_block_btn,
                            "group_label_name": embedding_group_label_name # Store the group's label for event handler
                        })


                # Collect components for event binding outside the loop
                matrix_block_uis.append(matrix_block)
                matrix_type_dropdowns.append(m_type_dropdown)
                matrix_label_textboxes.append(m_label_textbox)
                insert_tag_buttons.append(insert_tag_btn)
                matrix_variable_groups.append({
                    "standard": standard_group,
                    "random": random_group,
                    "setting": setting_group,
                    "wildcard": wildcard_group,
                    "lora": lora_group,
                    "embedding": embedding_group # Added embedding group
                })
                matrix_setting_type_dropdowns.append(setting_type_dd) # Changed to setting_type_dd
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
            axis_x_display = gr.Textbox(label="X-Axis", interactive=False)
            axis_y_display = gr.Textbox(label="Y-Axis", interactive=False)
            axis_page_display = gr.Textbox(label="Page-Axes", interactive=False)
            random_axis_display = gr.Textbox(label="Random Axes (Not in Grid)", interactive=False)
            swap_xy_btn = gr.Button("Swap X/Y")
            cycle_page_btn = gr.Button("Cycle Page/Y/X")

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
            # Remove .pt extension from selected embedding names
            new_embeddings = [name.replace(".pt", "") for name in selected_embeddings_list]
            existing_embeddings = [e.strip() for e in current_text_box_value.split(',') if e.strip()]
            # Combine and remove duplicates, then sort
            combined = sorted(list(set(existing_embeddings + new_embeddings)))
            return ", ".join(combined)

        def on_add_loras(selected_loras_list, current_text_box_value):
            # Format selected LoRAs into <lora:name:1.0> syntax
            new_loras = [f"<lora:{lora.replace('.safetensors', '').replace('.pt', '')}:1.0>" for lora in selected_loras_list]
            existing_loras = [l.strip() for l in current_text_box_value.split(',') if l.strip()]
            # Combine and remove duplicates, then sort
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

            # Collect all outputs into a list in the correct order for the .click()
            updates = []
            updates.append(gr.update(choices=embeddings_choices)) # static_embeddings_dropdown
            updates.append(gr.update(choices=lora_choices)) # static_loras_dropdown

            # 1D LoRA items
            for item_dd in lora_item_dropdowns:
                updates.append(gr.update(choices=lora_choices))
            # 1D Embedding items
            for item_dd in embedding_item_dropdowns:
                updates.append(gr.update(choices=embeddings_choices))

            # This part is for the Advanced builder (not yet visible/wired, but for future proofing)
            if 'matrix_type_dropdowns' in locals(): # This check is safer if the var might not be created yet
                for lora_manager_data in all_lora_sub_managers: # This might also contain embedding managers
                    for item_dd in lora_manager_data["sub_block_dropdowns"]:
                         updates.append(gr.update(choices=lora_choices if item_dd in lora_item_dropdowns else embeddings_choices)) # Crude check
            return tuple(updates) # Return as tuple for Gradio

        refresh_static_extras_btn.click(
            fn=refresh_static_extras_choices,
            outputs=[
                static_embeddings_dropdown, static_loras_dropdown,
                *[item for item in lora_item_dropdowns], # Unpack individual dropdowns
                *[item for item in embedding_item_dropdowns], # Unpack individual dropdowns
                *[dd for manager in all_lora_sub_managers for dd in manager["sub_block_dropdowns"]] # Unpack advanced mode LoRA/Embedding dropdowns
            ]
        )

        # 2. Prompt & Negative Prompt Button Handlers
        clear_prompt_btn.click(fn=lambda: "", outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", outputs=[negative_prompt])
        paste_prompts_btn.click(
            fn=upm_utils.paste_last_prompts, # Directly call the function, it now returns a tuple of updates
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

        # Handler for "Add New Matrix" button (Only for 3D+ Advanced mode)
        def on_add_new_matrix_block(current_num_active):
            updates = {}
            if current_num_active < MAX_MATRIX_BLOCKS:
                new_num = current_num_active + 1
                updates[matrix_block_uis[new_num - 1]] = gr.update(visible=True) # Make next block visible
                updates[matrix_block_uis[new_num - 1].children[0].children[0]] = gr.update(value=True) # Set its 'On' checkbox to True
                if new_num == MAX_MATRIX_BLOCKS:
                    updates[add_new_matrix_block_btn] = gr.update(interactive=False) # Disable if max reached
                return new_num, updates
            return current_num_active, updates # No change if already at max

        add_new_matrix_block_btn.click(
            fn=on_add_new_matrix_block,
            inputs=[num_active_matrix_blocks],
            outputs=[num_active_matrix_blocks, add_new_matrix_block_btn] + matrix_block_uis + 
                    [block_ui.children[0].children[0] for block_ui in matrix_block_uis] # All 'On' checkboxes
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
                    groups["embedding"]: gr.update(visible=(selected_type == "Embedding"))
                }
                return updates
            
            m_type_dd.change(
                fn=_on_matrix_type_change_closure,
                inputs=[m_type_dd], 
                outputs=[
                    matrix_variable_groups[i]["standard"],
                    matrix_variable_groups[i]["random"],
                    matrix_variable_groups[i]["setting"],
                    matrix_variable_groups[i]["wildcard"],
                    matrix_variable_groups[i]["lora"],
                    matrix_variable_groups[i]["embedding"]
                ]
            )

        # Handler for "Insert Tag to Prompt" button for each matrix block
        for i, insert_btn in enumerate(insert_tag_buttons): 
            insert_btn.click(
                fn=on_insert_tag_to_prompt_builder_block, # Use a centralized handler for all builder blocks
                inputs=[
                    prompt, # The main prompt textbox
                    matrix_type_dropdowns[i], # This block's type
                    matrix_label_textboxes[i], # This block's label
                    matrix_setting_type_dropdowns[i], # This block's setting type dropdown
                    matrix_wildcard_name_dropdowns[i] # This block's wildcard name dropdown
                    # Note: LoRA/Embedding blocks don't insert tags, their buttons are tied to sub-builder actions
                ],
                outputs=[prompt]
            )
        
        # Centralized handler function for inserting tags from Matrix Builder blocks
        def on_insert_tag_to_prompt_builder_block(current_prompt_text, matrix_type, matrix_label, setting_type, wildcard_name):
            tag = ""
            if matrix_type == "Standard":
                tag = f"<|{matrix_label}|>"
            elif matrix_type == "Random":
                tag = f"<|~{matrix_label}~|>"
            elif matrix_type == "Setting":
                tag = f"|Δ:{setting_type}|" # Settings use |Δ:Type| or |Δ:Label|
            elif matrix_type == "Wildcard":
                if wildcard_name: 
                    tag = f"<|__{wildcard_name.replace('.txt', '')}__|>"
            # LoRA and Embedding types don't insert tags directly
            
            if tag:
                return f"{current_prompt_text.strip()} {tag}".strip()
            return current_prompt_text


        # Handler for "Add LoRA/Weight Block" button within each LoRA matrix definition
        for lora_manager_data in all_lora_sub_managers:
            lora_state_component = lora_manager_data["num_active_state"]
            lora_sub_containers = lora_manager_data["sub_block_containers"]
            lora_add_btn_component = lora_manager_data["add_button"]
            lora_group_label_name = lora_manager_data["group_label_name"] # Get the associated label textbox

            def _add_lora_sub_block_closure(current_num_active_sub_blocks, containers=lora_sub_containers, add_button=lora_add_btn_component):
                updates = {}
                if current_num_active_sub_blocks < MAX_MATRIX_BLOCKS:
                    new_num = current_num_active_sub_blocks + 1
                    updates[containers[new_num - 1]] = gr.update(visible=True)
                    if new_num == MAX_MATRIX_BLOCKS:
                        updates[add_button] = gr.update(interactive=False)
                    return new_num, updates
                return current_num_active_sub_blocks, updates

            lora_add_btn_component.click(
                fn=_add_lora_sub_block_closure,
                inputs=[lora_state_component],
                outputs=[lora_state_component, lora_add_btn_component] + lora_sub_containers
            )

        # Logic for LoRA item changes (no longer a displayed syntax shortcut or insert button)
        def on_lora_items_change(lora_matrix_label_name, *args): 
            pass # No updates needed for syntax display, just a trigger for backend later
        
        # NOTE: lora_item_dropdowns and lora_item_weights are from the 1D LoRA builder.
        # Need to dynamically collect these per manager if this handler is for advanced mode blocks too.
        # For now, this is wired only for the 1D LoRA builder.
        lora_item_inputs_for_change = [lora_name] # Start with the name, then all dropdowns/weights
        for item_dd, item_weight in zip(lora_item_dropdowns, lora_item_weights):
            lora_item_inputs_for_change.append(item_dd)
            lora_item_inputs_for_change.append(item_weight)

        for item_input in lora_item_inputs_for_change:
            item_input.change(fn=on_lora_items_change, inputs=lora_item_inputs_for_change, outputs=[])

        # Handler for Embedding add item button (1D Embedding sub-builder)
        embedding_item_active_count = gr.State(value=1)
        embedding_item_containers = [item["row"] for item in embedding_matrix_items]
        embedding_item_dropdowns = [item["dropdown"] for item in embedding_matrix_items]
        embedding_item_weights = [item["weight"] for item in embedding_matrix_items]

        def _add_embedding_item_closure(current_num_active_items):
            updates = {}
            if current_num_active_items < MAX_MATRIX_BLOCKS:
                new_num = current_num_active_items + 1
                updates[embedding_item_containers[new_num - 1]] = gr.update(visible=True)
                if new_num == MAX_MATRIX_BLOCKS:
                    updates[add_embedding_item_btn] = gr.update(interactive=False)
                return new_num, updates
            return current_num_active_items, updates

        add_embedding_item_btn.click(
            fn=_add_embedding_item_closure,
            inputs=[embedding_item_active_count],
            outputs=[embedding_item_active_count, add_embedding_item_btn] + embedding_item_containers
        )

        # Logic for Embedding item changes (no longer a displayed syntax shortcut or insert button)
        def on_embedding_items_change(embedding_matrix_label_name, *args):
            pass # No updates needed for syntax display, just a trigger for backend later
        
        # NOTE: embedding_item_dropdowns and embedding_item_weights are from the 1D Embedding builder.
        # Need to dynamically collect these per manager if this handler is for advanced mode blocks too.
        # For now, this is wired only for the 1D Embedding builder.
        embedding_item_inputs_for_change = [embedding_name] # Start with the name, then all dropdowns/weights
        for item_dd, item_weight in zip(embedding_item_dropdowns, embedding_item_weights):
            embedding_item_inputs_for_change.append(item_dd)
            embedding_item_inputs_for_change.append(item_weight)

        for item_input in embedding_item_inputs_for_change:
            item_input.change(fn=on_embedding_items_change, inputs=embedding_item_inputs_for_change, outputs=[])
        
        return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)