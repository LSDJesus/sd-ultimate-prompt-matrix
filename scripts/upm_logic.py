import re
import random
from itertools import product

REX_REFTAG = re.compile(r'\|(matrix|random|Δ)\:([^|]+)\|')
REX_LEGACY_MATRIX = re.compile(r'(<(?!lora:)([^>]+)>)')
REX_LEGACY_PARAM = re.compile(r'<(?!random\b)([a-zA-Z_]+)\(([^)]+)\)>')

def parse_axis_references(prompt):
    """Parses the new reference tags from a prompt string."""
    axes = []
    for match in REX_REFTAG.finditer(prompt):
        axes.append({
            "token": match.group(0),
            "type": match.group(1),
            "label": match.group(2).strip(),
            "start": match.start()
        })
    return sorted(axes, key=lambda x: x['start'])

def build_job_list(p, seed_behavior, matrix_definitions):
    """
    HYBRID MODEL REWRITE
    This function now returns a list of "page jobs".
    Each page job is a dictionary containing everything needed to run a batch,
    including a list of sub-prompts for the X/Y grid.
    """
    definitions_map = {d['label']: d for d in matrix_definitions}

    # 1. Parse and enrich references from prompts
    pos_refs = parse_axis_references(p.prompt)
    neg_refs = parse_axis_references(p.negative_prompt)
    for ref in pos_refs: ref['origin'] = 'positive'
    for ref in neg_refs: ref['origin'] = 'negative'
    
    all_axes = []
    prompt_refs = sorted(pos_refs + neg_refs, key=lambda x: x['start'])
    for ref in prompt_refs:
        if ref['label'] in definitions_map:
            full_axis_def = {**definitions_map[ref['label']], **ref}
            all_axes.append(full_axis_def)
        else:
            print(f"[UPM Warning] Reference '{ref['token']}' has no definition and will be ignored.")

    if not all_axes:
        # No matrix defined, return a single "page job" with one prompt.
        return [{"prompts": [p.prompt], "params": {}, "seeds": [p.seed]}], None

    # 2. Separate axes and VALIDATE the hybrid model rules.
    random_axes = [axis for axis in all_axes if axis['type'] == 'random']
    param_axes = [axis for axis in all_axes if axis['type'] == 'Δ']
    prompt_axes = [axis for axis in all_axes if axis['type'] == 'matrix']

    # Validation: ALL parameter axes must come BEFORE ALL prompt axes.
    # This enforces the "Params on Z-axis only" rule.
    last_param_pos = max([ax['start'] for ax in param_axes]) if param_axes else -1
    first_prompt_pos = min([ax['start'] for ax in prompt_axes]) if prompt_axes else float('inf')

    if last_param_pos > first_prompt_pos:
        print("[UPM CRITICAL ERROR] Invalid axis order for Hybrid Mode.")
        print("All parameter axes (|Δ:...|) must appear before all prompt axes (|matrix:...|) in the prompt.")
        return [], {"error": "Invalid axis order for Hybrid Mode. All |Δ:...| tags must come before all |matrix:...| tags."}

    # 3. Define page, X, and Y axes based on the new rules.
    # Page axes are now ONLY parameter axes.
    # X and Y axes are ONLY prompt axes.
    x_axis = prompt_axes[-1] if len(prompt_axes) > 0 else None
    y_axis = prompt_axes[-2] if len(prompt_axes) > 1 else None
    
    page_axes = param_axes # Page axes ARE the parameter axes.

    x_opts = x_axis['options'] if x_axis else ['']
    y_opts = y_axis['options'] if y_axis else ['']
    
    page_options_list = [axis['options'] for axis in page_axes]
    page_combinations = list(product(*page_options_list)) if page_options_list else [()]

    # 4. Build the list of page_jobs
    page_job_list = []
    base_seed = p.seed if p.seed != -1 else random.randint(0, 2**32 - 1)

    for page_idx, page_vals in enumerate(page_combinations):
        # This is a single page job, representing one batch.
        page_job = {"prompts": [], "params": {}, "seeds": []}
        
        # Apply page-level parameters
        page_prompt = p.prompt
        page_neg_prompt = p.negative_prompt
        if page_axes:
            for axis, value in zip(page_axes, page_vals):
                page_job['params'][axis['label']] = value
                # Also substitute in prompt in case it's used for display, will be cleaned later
                if axis['origin'] == 'positive': page_prompt = page_prompt.replace(axis['token'], "", 1)
                else: page_neg_prompt = page_neg_prompt.replace(axis['token'], "", 1)
        
        # Now, build the list of prompts for the X/Y grid of THIS page
        for row_idx, y_val in enumerate(y_opts):
            for col_idx, x_val in enumerate(x_opts):
                final_prompt = page_prompt
                final_neg_prompt = page_neg_prompt

                # Assign X and Y values
                if y_axis:
                    if y_axis['origin'] == 'positive': final_prompt = final_prompt.replace(y_axis['token'], y_val, 1)
                    else: final_neg_prompt = final_neg_prompt.replace(y_axis['token'], y_val, 1)
                if x_axis:
                    if x_axis['origin'] == 'positive': final_prompt = final_prompt.replace(x_axis['token'], x_val, 1)
                    else: final_neg_prompt = final_neg_prompt.replace(x_axis['token'], x_val, 1)

                # Assign random values
                for r_axis in random_axes:
                    chosen_val = random.choice(r_axis['options'])
                    if r_axis['origin'] == 'positive': final_prompt = final_prompt.replace(r_axis['token'], chosen_val, 1)
                    else: final_neg_prompt = final_neg_prompt.replace(r_axis['token'], chosen_val, 1)
                
                # Clean up any remaining tags and add to the list for this page
                job_prompt = re.sub(REX_REFTAG, '', final_prompt).strip(" ,")
                job_neg_prompt = re.sub(REX_REFTAG, '', final_neg_prompt).strip(" ,")
                page_job['prompts'].append(f"{job_prompt} --neg {job_neg_prompt}")

                # Assign seeds for this page's batch
                image_index = (row_idx * len(x_opts)) + col_idx
                if seed_behavior == "Fixed": page_job['seeds'].append(base_seed + page_idx)
                elif seed_behavior == "Iterate Per Image": page_job['seeds'].append(base_seed + (page_idx * len(y_opts) * len(x_opts)) + image_index)
                elif seed_behavior == "Iterate Per Row": page_job['seeds'].append(base_seed + (page_idx * len(y_opts)) + row_idx)
                elif seed_behavior == "Random": page_job['seeds'].append(-1)

        page_job_list.append(page_job)

    # 5. Prepare grid_info for the UI and grid drawing
    grid_info = {"x_axis": x_axis, "y_axis": y_axis, "page_axes": page_axes, "page_combinations": page_combinations}
    if grid_info.get('x_axis'): grid_info['x_axis']['param_name'] = 'prompt'
    if grid_info.get('y_axis'): grid_info['y_axis']['param_name'] = 'prompt'
    for axis in grid_info.get('page_axes', []): axis['param_name'] = axis['label']

    return page_job_list, grid_info