"""
Ultimate Prompt Matrix v15.1.1
Author: LSDJesus
Changes:
- Implemented the full "1D Axis" builder mode within the "Matrix Builder" accordion:
    - Added nested radio buttons ("Standard", "Random", "Wildcard", "LoRA", "Embedding", "Generation Setting") to select 1D axis type.
    - Dynamically displays specific input fields (Name/Title, Variables, Setting Type, Wildcard Name, LoRA/Embedding sub-builders) based on selected 1D axis type.
    - Added syntax display textboxes and "Insert Shortcut to Prompt" buttons for each 1D axis type, generating the correct |matrix:label|, |random:label|, |wildcard:name|, |Δ:setting| tags.
    - Implemented a sub-builder for "LoRA" matrix type, allowing definition of multiple LoRA/Weight combinations with a dynamic "Add LoRA Item" button.
    - Implemented a sub-builder for "Embedding" matrix type, allowing definition of multiple Embedding/Weight combinations with a dynamic "Add Embedding Item" button.
    - Integrated `upm_wildcard_handler` to populate the "Wildcard Name" dropdown for 1D mode.
    - Updated `refresh_static_extras_btn` to also refresh LoRA and Embedding choices within the 1D matrix builder.
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
                paste_prompts_btn = gr.Button("↙️", elem_classes="tool_sm", tooltip="Paste prompts from last generation")
        with gr.Row():
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Enter negative prompts here...", elem_id="upm_negative_prompt")
            with gr.Column(scale=0, min_width=50):
                clear_neg_prompt_btn = gr.Button("🗑️", elem_classes="tool_sm")

        # 4. Matrix Builder (OVERHAULED)
        with gr.Accordion("Matrix Builder", open=True):
            gr.Markdown("Define your matrix axes here. Remember: **|Δ:param|** axes must come before **|matrix:prompt|** axes for live previews to work.")
            
            # State to manage the number of active matrix blocks
            num_active_matrix_blocks = gr.State(value=1) 
            
            matrix_block_uis = [] # To store the gr.Group for each matrix block
            matrix_type_dropdowns = [] # To store the type dropdowns for event handling
            matrix_label_textboxes = [] # To store the label textboxes for event handling
            insert_tag_buttons = [] # To store the insert tag buttons for event handling
            matrix_variable_groups = [] # To store the conditional variable input groups for each block
            matrix_setting_type_dropdowns = [] # To store setting type dropdowns for insert tag fn
            matrix_wildcard_name_dropdowns = [] # To store wildcard name dropdowns for insert tag fn
            all_lora_sub_managers = [] # To store info for managing LoRA sub-blocks

            # New Radio Buttons for Matrix Mode within the builder (1D, 2D, 3D, 3D+Advanced)
            builder_mode = gr.Radio(
                ["1D Axis", "2D Grid", "3D Grid", "3D+ (Advanced)"],
                label="Builder Mode",
                value="1D Axis" # Default selection
            )

            # --- Start UI for 1D Axis Builder Mode ---
            with gr.Column(visible=True) as builder_1d_ui:
                # Nested Radio Buttons for the 1D Axis Type
                one_d_axis_type_radio = gr.Radio(
                    ["Standard", "Random", "Wildcard", "LoRA", "Embedding", "Generation Setting"],
                    label="1D Axis Type",
                    value="Standard" # Default selection for 1D mode
                )

                # --- Dynamic UI for each 1D Axis Type ---
                with gr.Column(visible=True) as one_d_standard_ui:
                    with gr.Row():
                        std_name = gr.Textbox(label="Name/Title", placeholder="e.g., animal, style")
                        std_vars = gr.Textbox(label="Variables (comma-separated)", placeholder="e.g., cat, dog, bird")
                    with gr.Row():
                        std_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |matrix:animal|")
                        std_insert_btn = gr.Button("Insert Shortcut to Prompt")

                with gr.Column(visible=False) as one_d_random_ui:
                    with gr.Row():
                        rand_name = gr.Textbox(label="Name/Title", placeholder="e.g., headwear, object")
                        rand_vars = gr.Textbox(label="Variables (comma-separated)", placeholder="e.g., hat, scarf, boots")
                    with gr.Row():
                        rand_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |random:headwear|")
                        rand_insert_btn = gr.Button("Insert Shortcut to Prompt")

                with gr.Column(visible=False) as one_d_wildcard_ui:
                    wc_name_dropdown = gr.Dropdown(
                        label="Wildcard Name",
                        choices=upm_wildcard_handler.get_wildcard_files(), # Dynamically load wildcard files
                        value=None
                    )
                    wc_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |wildcard:subjects|")
                    wc_insert_btn = gr.Button("Insert Shortcut to Prompt")

                with gr.Column(visible=False) as one_d_lora_ui:
                    gr.Markdown("Define LoRA combinations for this matrix label:")
                    lora_name = gr.Textbox(label="Name/Title", placeholder="e.g., LoRA_Styles")
                    lora_matrix_items = [] # To store the individual LoRA + Weight sets
                    lora_item_active_count = gr.State(value=1) # State for this specific Add LoRA Item button
                    lora_item_containers = [] # List of gr.Row components
                    lora_item_dropdowns = []
                    lora_item_weights = []

                    for j in range(MAX_MATRIX_BLOCKS): # Using MAX_MATRIX_BLOCKS as the limit for LoRA items too
                        with gr.Row(visible=(j==0)) as lora_item_row: # First item visible
                            lora_dd = gr.Dropdown(
                                label=f"LoRA {j+1}",
                                choices=[], # Initialized empty, populated by refresh
                                multiselect=False, # Single LoRA per line for explicit weight
                                scale=2
                            )
                            lora_weight = gr.Slider(label="Weight", minimum=-2.0, maximum=2.0, value=1.0, step=0.05, scale=1)
                            lora_item_containers.append(lora_item_row)
                            lora_item_dropdowns.append(lora_dd)
                            lora_item_weights.append(lora_weight)
                    add_lora_item_btn = gr.Button("Add LoRA Item")
                    lora_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |matrix:LoRA_Styles|")
                    lora_insert_btn = gr.Button("Insert Shortcut to Prompt")
                    gr.Markdown("Note: LoRAs will be appended to the end of the prompt for each image.")

                with gr.Column(visible=False) as one_d_embedding_ui:
                    gr.Markdown("Define Embedding combinations for this matrix label:")
                    embedding_name = gr.Textbox(label="Name/Title", placeholder="e.g., Embedding_Details")
                    embedding_matrix_items = []
                    embedding_item_active_count = gr.State(value=1)
                    embedding_item_containers = []
                    embedding_item_dropdowns = []
                    embedding_item_weights = []

                    for j in range(MAX_MATRIX_BLOCKS):
                        with gr.Row(visible=(j==0)) as embedding_item_row:
                            embedding_dd = gr.Dropdown(
                                label=f"Embedding {j+1}",
                                choices=[], # Initialized empty, populated by refresh
                                multiselect=False,
                                scale=2
                            )
                            embedding_weight = gr.Slider(label="Weight", minimum=-2.0, maximum=2.0, value=1.0, step=0.05, scale=1)
                            embedding_item_containers.append(embedding_item_row)
                            embedding_item_dropdowns.append(embedding_dd)
                            embedding_item_weights.append(embedding_weight)
                    add_embedding_item_btn = gr.Button("Add Embedding Item")
                    embedding_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |matrix:Embedding_Details|")
                    embedding_insert_btn = gr.Button("Insert Shortcut to Prompt")
                    gr.Markdown("Note: Embeddings will be appended to the beginning of the prompt for each image.")

                with gr.Column(visible=False) as one_d_setting_ui:
                    setting_name = gr.Textbox(label="Name/Title (Optional)", placeholder="e.g., CFG_Values")
                    setting_type_dd = gr.Dropdown(
                        label="Setting Type",
                        choices=GENERATION_SETTINGS_CHOICES, # Use predefined list
                        value="cfg_scale"
                    )
                    setting_vars = gr.Textbox(label="Variables (comma-separated)", placeholder="e.g., 7.0, 9.0 (for CFG) or 512x768, 768x512 (for Size)")
                    setting_syntax_display = gr.Textbox(label="Syntax Shortcut", interactive=False, placeholder="e.g., |Δ:CFG_Values| or |Δ:cfg_scale|")
                    setting_insert_btn = gr.Button("Insert Shortcut to Prompt")
                    gr.Markdown("Note: This does not need to be added to the prompt. It will automatically apply the setting.")

                # Function to handle 1D axis type changes
                def on_one_d_axis_type_change(axis_type):
                    updates = {
                        one_d_standard_ui: gr.update(visible=(axis_type == "Standard")),
                        one_d_random_ui: gr.update(visible=(axis_type == "Random")),
                        one_d_wildcard_ui: gr.update(visible=(axis_type == "Wildcard")),
                        one_d_lora_ui: gr.update(visible=(axis_type == "LoRA")),
                        one_d_embedding_ui: gr.update(visible=(axis_type == "Embedding")),
                        one_d_setting_ui: gr.update(visible=(axis_type == "Generation Setting")),
                    }
                    return updates

                one_d_axis_type_radio.change(
                    fn=on_one_d_axis_type_change,
                    inputs=one_d_axis_type_radio,
                    outputs=[
                        one_d_standard_ui, one_d_random_ui, one_d_wildcard_ui,
                        one_d_lora_ui, one_d_embedding_ui, one_d_setting_ui
                    ]
                )
            # --- End UI for 1D Axis Builder Mode ---

            # Placeholder columns for other builder modes
            with gr.Column(visible=False) as builder_2d_ui:
                gr.Markdown("**(2D Grid Builder UI Placeholder)**")

            with gr.Column(visible=False) as builder_3d_ui:
                gr.Markdown("**(3D Grid Builder UI Placeholder)**")

            with gr.Column(visible=False) as builder_advanced_ui:
                gr.Markdown("**(3D+ Advanced Builder UI Placeholder)**")

            # Function to handle main builder mode changes
            def on_builder_mode_change(mode):
                return {
                    builder_1d_ui: gr.update(visible=(mode == "1D Axis")),
                    builder_2d_ui: gr.update(visible=(mode == "2D Grid")),
                    builder_3d_ui: gr.update(visible=(mode == "3D Grid")),
                    builder_advanced_ui: gr.update(visible=(mode == "3D+ (Advanced)")),
                }

            # Connect the builder mode radio buttons to the function
            builder_mode.change(
                fn=on_builder_mode_change,
                inputs=builder_mode,
                outputs=[
                    builder_1d_ui,
                    builder_2d_ui,
                    builder_3d_ui,
                    builder_advanced_ui,
                ]
            )
        
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
            # Ensure attributes exist before trying to access .keys() or iterate
            embeddings_choices = list(shared.sd_embeddings.embeddings.keys()) if hasattr(shared, 'sd_embeddings') and shared.sd_embeddings else []
            lora_choices = [lora.name for lora in sd_models.loras] if hasattr(sd_models, 'loras') else []

            updates = {
                static_embeddings_dropdown: gr.update(choices=embeddings_choices),
                static_loras_dropdown: gr.update(choices=lora_choices),
            }
            # Also refresh LoRA choices in the Matrix Builder's 1D LoRA sub-builder
            # and in any other matrix blocks (2D, 3D, Advanced) if they have LoRA dropdowns
            for item in lora_item_dropdowns: # 1D LoRA items
                updates[item] = gr.update(choices=lora_choices)
            for item in embedding_item_dropdowns: # 1D Embedding items
                updates[item] = gr.update(choices=embeddings_choices)

            # This part is for the Advanced builder (not yet visible/wired, but for future proofing)
            # if matrix_type_dropdowns is defined and contains elements, iterate through its LORA sections
            if 'matrix_type_dropdowns' in locals():
                for lora_manager_data in all_lora_sub_managers:
                    for item in lora_manager_data["sub_block_containers"]: # Assuming sub_block_containers elements have 'dropdown' children
                        if len(item.children) > 0 and hasattr(item.children[0], 'choices'): # Check if it's a dropdown and has choices
                             updates[item.children[0]] = gr.update(choices=lora_choices)
            return updates

        refresh_static_extras_btn.click(
            fn=refresh_static_extras_choices,
            outputs=[
                static_embeddings_dropdown, static_loras_dropdown,
                *[item for item in lora_item_dropdowns], # Unpack individual dropdowns
                *[item for item in embedding_item_dropdowns] # Unpack individual dropdowns
                # Dynamically added advanced mode LoRA dropdowns will be handled when they are created and appended to matrix_type_dropdowns
            ]
        )

        # 2. Prompt & Negative Prompt Button Handlers
        clear_prompt_btn.click(fn=lambda: "", outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", outputs=[negative_prompt])
        paste_prompts_btn.click(
            fn=upm_utils.paste_last_prompts,
            outputs=[prompt, negative_prompt]
        )

        # 3. Matrix Builder Handlers

        # Handler for "Add New Matrix" button (Top-level blocks in Advanced mode - currently not used for 1D, 2D, 3D)
        def on_add_new_matrix_block(current_num_active):
            updates = {}
            if current_num_active < MAX_MATRIX_BLOCKS:
                new_num = current_num_active + 1
                # Make the next block visible
                updates[matrix_block_uis[new_num - 1]] = gr.update(visible=True)
                # Set its 'On' checkbox to True (accessing through children)
                updates[matrix_block_uis[new_num - 1].children[0].children[0]] = gr.update(value=True)
                # If max reached, disable the add button
                if new_num == MAX_MATRIX_BLOCKS:
                    updates[add_new_matrix_block_btn] = gr.update(interactive=False)
                return new_num, updates
            return current_num_active, updates # No change if already at max

        # add_new_matrix_block_btn.click( # Temporarily commented out as this button is not yet visible in any mode.
        #     fn=on_add_new_matrix_block,
        #     inputs=[num_active_matrix_blocks],
        #     outputs=[num_active_matrix_blocks, add_new_matrix_block_btn] + matrix_block_uis + 
        #             [block_ui.children[0].children[0] for block_ui in matrix_block_uis] # All 'On' checkboxes
        # )

        # Handler for each Matrix Type Dropdown (Top-level blocks in Advanced mode - currently not used for 1D, 2D, 3D)
        # for i, m_type_dd in enumerate(matrix_type_dropdowns): # These components are commented out from the main loop
        #     def _on_matrix_type_change_closure(selected_type, current_index=i):
        #         groups = matrix_variable_groups[current_index]
        #         updates = {
        #             groups["standard"]: gr.update(visible=(selected_type == "Standard")),
        #             groups["random"]: gr.update(visible=(selected_type == "Random")),
        #             groups["setting"]: gr.update(visible=(selected_type == "Setting")),
        #             groups["wildcard"]: gr.update(visible=(selected_type == "Wildcard")),
        #             groups["lora"]: gr.update(visible=(selected_type == "LoRA"))
        #         }
        #         return updates
            
        #     m_type_dd.change(
        #         fn=_on_matrix_type_change_closure,
        #         inputs=[m_type_dd], 
        #         outputs=[
        #             matrix_variable_groups[i]["standard"],
        #             matrix_variable_groups[i]["random"],
        #             matrix_variable_groups[i]["setting"],
        #             matrix_variable_groups[i]["wildcard"],
        #             matrix_variable_groups[i]["lora"]
        #         ]
        #     )

        # Handler for "Insert Tag to Prompt" button for each matrix block (Top-level blocks in Advanced mode)
        # def on_insert_tag_to_prompt(current_prompt_text, matrix_type, matrix_label, setting_type, wildcard_name): # This handler is for the main matrix blocks, not 1D mode
        #     tag = ""
        #     if matrix_type == "Standard":
        #         tag = f"|matrix:{matrix_label}|"
        #     elif matrix_type == "Random":
        #         tag = f"|random:{matrix_label}|"
        #     elif matrix_type == "Setting":
        #         tag = f"|Δ:{setting_type}|"
        #     elif matrix_type == "Wildcard":
        #         if wildcard_name: # Ensure a wildcard is selected
        #             tag = f"|wildcard:{wildcard_name.replace('.txt', '')}|"
        #         else:
        #             return current_prompt_text # Do not insert if no wildcard selected
        #     elif matrix_type == "LoRA":
        #         tag = f"|Loras:{matrix_label}|" # Use the defined label for LoRA matrix

        #     if tag:
        #         # Append the tag to the current prompt, ensuring a space if needed
        #         return f"{current_prompt_text.strip()} {tag}".strip()
        #     return current_prompt_text

        # for i, insert_btn in enumerate(insert_tag_buttons): # These components are commented out from the main loop
        #     insert_btn.click(
        #         fn=on_insert_tag_to_prompt,
        #         inputs=[
        #             prompt, # The main prompt textbox
        #             matrix_type_dropdowns[i], # This block's type
        #             matrix_label_textboxes[i], # This block's label
        #             matrix_setting_type_dropdowns[i], # This block's setting type dropdown
        #             matrix_wildcard_name_dropdowns[i] # This block's wildcard name dropdown
        #         ],
        #         outputs=[prompt]
        #     )
        
        # Handler for "Add LoRA/Weight Block" button within each LoRA matrix definition (Top-level blocks in Advanced mode)
        # for lora_manager_data in all_lora_sub_managers: # These components are commented out from the main loop
        #     lora_state_component = lora_manager_data["num_active_state"]
        #     lora_sub_containers = lora_manager_data["sub_block_containers"]
        #     lora_add_btn_component = lora_manager_data["add_button"]

        #     def _add_lora_sub_block_closure(current_num_active_sub_blocks, containers=lora_sub_containers, add_button=lora_add_btn_component):
        #         updates = {}
        #         if current_num_active_sub_blocks < MAX_MATRIX_BLOCKS:
        #             new_num = current_num_active_sub_blocks + 1
        #             updates[containers[new_num - 1]] = gr.update(visible=True)
        #             if new_num == MAX_MATRIX_BLOCKS:
        #                 updates[add_button] = gr.update(interactive=False)
        #             return new_num, updates
        #         return current_num_active_sub_blocks, updates

        #     lora_add_btn_component.click(
        #         fn=_add_lora_sub_block_closure,
        #         inputs=[lora_state_component],
        #         outputs=[lora_state_component, lora_add_btn_component] + lora_sub_containers
        #     )

        # --- Handlers for 1D Axis Builder ---
        # Syntax display and insert tag for Standard Type
        def on_std_change(name_text):
            if name_text:
                return gr.update(value=f"|matrix:{name_text}|")
            return gr.update(value="")
        std_name.change(fn=on_std_change, inputs=std_name, outputs=std_syntax_display)
        std_insert_btn.click(fn=lambda text, tag_text: f"{text.strip()} {tag_text}".strip() if tag_text else text, inputs=[prompt, std_syntax_display], outputs=prompt)
        
        # Syntax display and insert tag for Random Type
        def on_rand_change(name_text):
            if name_text:
                return gr.update(value=f"|random:{name_text}|")
            return gr.update(value="")
        rand_name.change(fn=on_rand_change, inputs=rand_name, outputs=rand_syntax_display)
        rand_insert_btn.click(fn=lambda text, tag_text: f"{text.strip()} {tag_text}".strip() if tag_text else text, inputs=[prompt, rand_syntax_display], outputs=prompt)

        # Syntax display and insert tag for Wildcard Type
        def on_wc_dropdown_change(selected_wildcard_file):
            if selected_wildcard_file:
                # Remove .txt extension for the tag
                return gr.update(value=f"|wildcard:{selected_wildcard_file.replace('.txt', '')}|")
            return gr.update(value="")
        wc_name_dropdown.change(fn=on_wc_dropdown_change, inputs=wc_name_dropdown, outputs=wc_syntax_display)
        wc_insert_btn.click(fn=lambda text, tag_text: f"{text.strip()} {tag_text}".strip() if tag_text else text, inputs=[prompt, wc_syntax_display], outputs=prompt)

        # Syntax display and insert tag for Generation Setting Type
        def on_setting_change(setting_type_value, name_text): # Added name_text as input
            tag = ""
            if setting_type_value:
                # If a name is provided, use that for the tag, otherwise use the setting type
                tag = f"|Δ:{name_text if name_text else setting_type_value}|"
            return gr.update(value=tag) # Update the display
        
        setting_type_dd.change(fn=on_setting_change, inputs=[setting_type_dd, setting_name], outputs=setting_syntax_display)
        setting_name.change(fn=on_setting_change, inputs=[setting_type_dd, setting_name], outputs=setting_syntax_display) # Update if name changes too
        setting_insert_btn.click(fn=lambda text, tag_text: f"{text.strip()} {tag_text}".strip() if tag_text else text, inputs=[prompt, setting_syntax_display], outputs=prompt)
        
        # Handler for LoRA add item button (1D LoRA sub-builder)
        # Using already defined lora_item_active_count etc.
        def _add_lora_item_closure(current_num_active_items):
            updates = {}
            if current_num_active_items < MAX_MATRIX_BLOCKS:
                new_num = current_num_active_items + 1
                updates[lora_item_containers[new_num - 1]] = gr.update(visible=True)
                if new_num == MAX_MATRIX_BLOCKS:
                    updates[add_lora_item_btn] = gr.update(interactive=False)
                return new_num, updates
            return current_num_active_items, updates

        add_lora_item_btn.click(
            fn=_add_lora_item_closure,
            inputs=[lora_item_active_count],
            outputs=[lora_item_active_count, add_lora_item_btn] + lora_item_containers
        )

        # Logic to generate LoRA syntax string
        def on_lora_items_change(lora_matrix_label_name, *args): # Added lora_matrix_label_name
            # args will contain (selected_lora_1, weight_1, selected_lora_2, weight_2, ...)
            # based on how many lora_item_dropdowns and lora_item_weights are passed in inputs
            lora_parts_values = []
            for i in range(MAX_MATRIX_BLOCKS):
                # Ensure the current item's row is visible before trying to access its dropdown/weight values
                if i*2 < len(args) and lora_item_containers[i].visible: 
                    lora_name = args[i*2]
                    lora_weight = args[i*2+1]
                    if lora_name and lora_name != "None":
                        lora_parts_values.append(f"<lora:{lora_name}:{lora_weight}>")
            
            # The syntax display should be |Loras:Label| not the full lora string
            if lora_matrix_label_name:
                return gr.update(value=f"|Loras:{lora_matrix_label_name}|") 
            return gr.update(value="")


        # We need to collect all dropdowns and sliders for this change handler
        lora_item_inputs_for_change = [lora_name] # Start with the name, then all dropdowns/weights
        for item in lora_matrix_items:
            lora_item_inputs_for_change.append(item["dropdown"])
            lora_item_inputs_for_change.append(item["weight"])

        for item in lora_item_inputs_for_change:
            item.change(fn=on_lora_items_change, inputs=lora_item_inputs_for_change, outputs=lora_syntax_display)

        # Handler for Embedding add item button (1D Embedding sub-builder)
        # Using already defined embedding_item_active_count etc.
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

        # Logic to generate Embedding syntax string
        def on_embedding_items_change(embedding_matrix_label_name, *args): # Added embedding_matrix_label_name
            # embedding_parts_values = [] # Not directly used for syntax display, but useful for backend
            # for i in range(MAX_MATRIX_BLOCKS):
            #     if i*2 < len(args) and embedding_item_containers[i].visible: # Check if this item is active/visible
            #         embedding_name = args[i*2]
            #         embedding_weight = args[i*2+1]
            #         if embedding_name and embedding_name != "None":
            #             embedding_parts_values.append(f"({embedding_name}:{embedding_weight})")
            
            # The syntax display should be |Embeddings:Label| not the full embedding string
            if embedding_matrix_label_name:
                return gr.update(value=f"|Embeddings:{embedding_matrix_label_name}|")
            return gr.update(value="")
        
        # We need to collect all dropdowns and sliders for this change handler
        embedding_item_inputs_for_change = [embedding_name] # Start with the name, then all dropdowns/weights
        for item in embedding_matrix_items:
            embedding_item_inputs_for_change.append(item["dropdown"])
            embedding_item_inputs_for_change.append(item["weight"])

        for item in embedding_item_inputs_for_change:
            item.change(fn=on_embedding_items_change, inputs=embedding_item_inputs_for_change, outputs=embedding_syntax_display)

        embedding_insert_btn.click(fn=lambda text, tag_text: f"{tag_text} {text.strip()}".strip() if tag_text else text, inputs=[prompt, embedding_syntax_display], outputs=prompt)
        lora_insert_btn.click(fn=lambda text, tag_text: f"{text.strip()} {tag_text}".strip() if tag_text else text, inputs=[prompt, lora_syntax_display], outputs=prompt)
        
        return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)