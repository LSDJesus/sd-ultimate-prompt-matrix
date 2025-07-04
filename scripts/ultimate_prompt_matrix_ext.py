"""
Ultimate Prompt Matrix v15.1.0
Author: LSDJesus
Changes:
- Fixed ModuleNotFoundError by explicitly adding the script's directory to sys.path.
- Overhauled "Static LoRA/Embedding" section:
    - Implemented multi-select dropdowns for available Embeddings and LoRAs.
    - Added "Add Selected" buttons to transfer choices to dedicated textboxes.
    - Introduced "Clear" buttons for Embeddings and LoRAs textboxes.
    - Added a "Refresh LoRAs/Embeddings" button to update dropdown choices.
    - Integrated `modules.sd_models` for dynamic loading of available LoRAs and Textual Inversions.
- Overhauled "Matrix Builder" section:
    - Removed the fixed `matrix_builder_rows` and `matrix_row_slider`.
    - Implemented a dynamic "Add New Matrix" button to add up to 5 matrix definition blocks.
    - Each matrix block now includes:
        - "On" checkbox, "Type" dropdown ("Standard", "Random", "Setting", "Wildcard", "LoRA"), and "Reference Label" textbox.
        - Conditional input fields that appear based on the selected "Matrix Type" (e.g., "Variables" for Standard/Random, "Setting Type" for Setting, "Wildcard Name" for Wildcard, LoRA sub-builder for LoRA).
    - Integrated `upm_wildcard_handler` to populate the "Wildcard Name" dropdown.
    - Implemented a sub-builder for "LoRA" matrix type, allowing definition of multiple LoRA/Weight combinations per matrix.
    - Added "Insert Tag to Prompt" buttons for each matrix block to help users insert correct syntax.
- Enhanced Prompt & Negative Prompt section:
    - Added "Clear" buttons next to both Prompt and Negative Prompt textboxes.
    - Integrated "Paste Last Prompts" button using `upm_utils` for convenience.
"""
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers, sd_schedulers
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
            seed_behavior = gr.Radio(["Iterate Per Image", "Iterate Per Row", "Fixed", "Random"], label="Seed Behavior", value="Iterate Per Image", scale=2)
            seed = gr.Number(label="Start Seed", value=-1, precision=0, interactive=True)
        
        # 2. Static LoRA/Embedding Section (OVERHAULED)
        with gr.Accordion("Static LoRA/Embedding", open=False):
            gr.Markdown("These are applied globally to every image generated, prepended to the main prompt.")
            
            with gr.Row():
                static_embeddings_dropdown = gr.Dropdown(
                    label="Embeddings", 
                    choices=[ti.name for ti in modules.sd_models.get_available_text_inversions()], 
                    multiselect=True, 
                    scale=3
                )
                add_selected_embeddings_btn = gr.Button("Add Selected Embeddings", scale=1)
                
                static_loras_dropdown = gr.Dropdown(
                    label="LoRAs", 
                    choices=[l.name for l in modules.sd_models.get_available_loras()], 
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

            for i in range(MAX_MATRIX_BLOCKS):
                with gr.Group(visible=(i == 0)) as matrix_block: # First block visible by default
                    with gr.Row():
                        enabled_checkbox = gr.Checkbox(label="On", value=(i == 0), scale=0) 
                        m_type_dropdown = gr.Dropdown(
                            ["Standard", "Random", "Setting", "Wildcard", "LoRA"], 
                            value="Standard", 
                            label="Type", 
                            scale=1, 
                            elem_id=f"matrix_type_dd_{i}" # Add unique ID for easier JS if needed
                        )
                        m_label_textbox = gr.Textbox(
                            placeholder="e.g., animal, cfg, style", 
                            label="Reference Label", 
                            scale=2, 
                            elem_id=f"matrix_label_tb_{i}"
                        )
                        insert_tag_btn = gr.Button("Insert Tag to Prompt", scale=1) 
                    
                    # Conditional input areas for variables based on type
                    with gr.Column(visible=True) as standard_group: # Default visible for "Standard"
                        standard_vars_textbox = gr.Textbox(
                            placeholder="e.g., cat, dog, bird", 
                            label="Variables (comma-separated)", 
                            lines=1, 
                            elem_id=f"standard_vars_tb_{i}"
                        )
                    
                    with gr.Column(visible=False) as random_group:
                        random_vars_textbox = gr.Textbox(
                            placeholder="e.g., hat, scarf, boots", 
                            label="Variables (comma-separated)", 
                            lines=1, 
                            elem_id=f"random_vars_tb_{i}"
                        )
                    
                    with gr.Column(visible=False) as setting_group:
                        with gr.Row():
                            setting_type_dropdown = gr.Dropdown(
                                label="Setting Type",
                                choices=["cfg_scale", "steps", "width", "height", "sampler_name", "scheduler", "seed", "denoising_strength", "checkpoint"],
                                value="cfg_scale",
                                scale=1, 
                                elem_id=f"setting_type_dd_{i}"
                            )
                            setting_vars_textbox = gr.Textbox(
                                placeholder="e.g., 7.0, 9.0 (for CFG) or dreamshaper, cyberrealistic (for Checkpoint)", 
                                label="Variables (comma-separated)", 
                                lines=1, 
                                scale=2, 
                                elem_id=f"setting_vars_tb_{i}"
                            )
                    
                    with gr.Column(visible=False) as wildcard_group:
                        wildcard_name_dropdown = gr.Dropdown(
                            label="Wildcard Name",
                            choices=upm_wildcard_handler.get_wildcard_files(),
                            value=None,
                            elem_id=f"wildcard_dropdown_{i}"
                        )
                    
                    with gr.Column(visible=False) as lora_group:
                        gr.Markdown("Define LoRA combinations for this matrix label:")
                        # State for managing sub-blocks within this specific LoRA matrix block
                        num_active_lora_sub_blocks = gr.State(value=0) 
                        
                        lora_sub_block_containers = [] 
                        # Max 5 sub-blocks per LoRA matrix (using MAX_MATRIX_BLOCKS as a convenient limit)
                        for j in range(MAX_MATRIX_BLOCKS): 
                            with gr.Group(visible=False) as lora_sub_block: 
                                with gr.Row():
                                    lora_matrix_dropdown = gr.Dropdown(
                                        label=f"LoRAs (Block {j+1})", 
                                        choices=[l.name for l in modules.sd_models.get_available_loras()], # Populated LoRA choices
                                        multiselect=True, 
                                        scale=2,
                                        elem_id=f"lora_matrix_dd_{i}_{j}"
                                    ) 
                                    lora_matrix_weight_slider = gr.Slider(
                                        label="Weight", 
                                        minimum=-2.0, maximum=2.0, value=1.0, step=0.05, 
                                        scale=1,
                                        elem_id=f"lora_matrix_weight_sl_{i}_{j}"
                                    )
                                lora_sub_block_containers.append(lora_sub_block)
                        
                        add_lora_matrix_block_btn = gr.Button("Add LoRA/Weight Block", elem_id=f"add_lora_block_btn_{i}")
                        
                        # Store data for managing LoRA sub-blocks for this matrix block
                        all_lora_sub_managers.append({
                            "num_active_state": num_active_lora_sub_blocks,
                            "sub_block_containers": lora_sub_block_containers,
                            "add_button": add_lora_matrix_block_btn
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
                    "lora": lora_group
                })
                matrix_setting_type_dropdowns.append(setting_type_dropdown)
                matrix_wildcard_name_dropdowns.append(wildcard_name_dropdown)

            # Button to add new top-level matrix blocks
            add_new_matrix_block_btn = gr.Button("Add New Matrix", elem_id="add_new_matrix_block_btn")


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
            return {
                static_embeddings_dropdown: gr.update(choices=[ti.name for ti in modules.sd_models.get_available_text_inversions()]),
                static_loras_dropdown: gr.update(choices=[l.name for l in modules.sd_models.get_available_loras()])
            }

        refresh_static_extras_btn.click(
            fn=refresh_static_extras_choices,
            outputs=[static_embeddings_dropdown, static_loras_dropdown]
        )

        # 2. Prompt & Negative Prompt Button Handlers
        clear_prompt_btn.click(fn=lambda: "", outputs=[prompt])
        clear_neg_prompt_btn.click(fn=lambda: "", outputs=[negative_prompt])
        paste_prompts_btn.click(
            fn=upm_utils.paste_last_prompts,
            outputs=[prompt, negative_prompt]
        )

        # 3. Matrix Builder Handlers

        # Handler for "Add New Matrix" button
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

        add_new_matrix_block_btn.click(
            fn=on_add_new_matrix_block,
            inputs=[num_active_matrix_blocks],
            outputs=[num_active_matrix_blocks, add_new_matrix_block_btn] + matrix_block_uis + 
                    [block_ui.children[0].children[0] for block_ui in matrix_block_uis] # All 'On' checkboxes
        )

        # Handler for each Matrix Type Dropdown
        for i, m_type_dd in enumerate(matrix_type_dropdowns):
            def _on_matrix_type_change_closure(selected_type, current_index=i):
                groups = matrix_variable_groups[current_index]
                updates = {
                    groups["standard"]: gr.update(visible=(selected_type == "Standard")),
                    groups["random"]: gr.update(visible=(selected_type == "Random")),
                    groups["setting"]: gr.update(visible=(selected_type == "Setting")),
                    groups["wildcard"]: gr.update(visible=(selected_type == "Wildcard")),
                    groups["lora"]: gr.update(visible=(selected_type == "LoRA"))
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
                    matrix_variable_groups[i]["lora"]
                ]
            )

        # Handler for "Insert Tag to Prompt" button for each matrix block
        def on_insert_tag_to_prompt(current_prompt_text, matrix_type, matrix_label, setting_type, wildcard_name):
            tag = ""
            if matrix_type == "Standard":
                tag = f"|matrix:{matrix_label}|"
            elif matrix_type == "Random":
                tag = f"|random:{matrix_label}|"
            elif matrix_type == "Setting":
                tag = f"|Δ:{setting_type}|"
            elif matrix_type == "Wildcard":
                if wildcard_name: # Ensure a wildcard is selected
                    tag = f"|wildcard:{wildcard_name.replace('.txt', '')}|"
                else:
                    return current_prompt_text # Do not insert if no wildcard selected
            elif matrix_type == "LoRA":
                tag = f"|Loras:{matrix_label}|" # Use the defined label for LoRA matrix

            if tag:
                # Append the tag to the current prompt, ensuring a space if needed
                return f"{current_prompt_text.strip()} {tag}".strip()
            return current_prompt_text

        for i, insert_btn in enumerate(insert_tag_buttons):
            insert_btn.click(
                fn=on_insert_tag_to_prompt,
                inputs=[
                    prompt, # The main prompt textbox
                    matrix_type_dropdowns[i], # This block's type
                    matrix_label_textboxes[i], # This block's label
                    matrix_setting_type_dropdowns[i], # This block's setting type dropdown
                    matrix_wildcard_name_dropdowns[i] # This block's wildcard name dropdown
                ],
                outputs=[prompt]
            )
        
        # Handler for "Add LoRA/Weight Block" button within each LoRA matrix definition
        for lora_manager_data in all_lora_sub_managers:
            lora_state_component = lora_manager_data["num_active_state"]
            lora_sub_containers = lora_manager_data["sub_block_containers"]
            lora_add_btn_component = lora_manager_data["add_button"]

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
        
        # This is the function that A1111 calls to add the tab to the UI.
        return [(ui_component, "Ultimate Matrix", "ultimate_matrix")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)