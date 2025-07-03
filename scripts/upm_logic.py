import re
import random
from itertools import product

# New regex to find reference tags like |matrix:animal|, |Δ(cfg)|, |Loras:style1|
# It captures the type ('matrix', 'Δ', 'Loras', 'random', etc.) and the label ('animal', 'cfg', 'style1')
REX_REFTAG = re.compile(r'\|(matrix|random|Δ)\:([^|]+)\|')

# Keep old regex for cleaning up any legacy syntax if it slips through, just in case
REX_LEGACY_MATRIX = re.compile(r'(<(?!lora:)([^>]+)>)')
REX_LEGACY_PARAM = re.compile(r'<(?!random\b)([a-zA-Z_]+)\(([^)]+)\)>')

def parse_axis_references(prompt):
    """
    Parses the new reference tags from a prompt string.
    Example: "a photo of a |matrix:animal|" -> [{'token': '|matrix:animal|', 'type': 'matrix', 'label': 'animal', 'start': 15}]
    """
    axes = []
    for match in REX_REFTAG.finditer(prompt):
        axes.append({
            "token": match.group(0),
            "type": match.group(1),
            "label": match.group(2).strip(),
            "start": match.start()
        })
    return sorted(axes, key=lambda x: x['start'])

def build_job_list(p, seed_behavior, matrix_definitions, axis_overrides=None):
    """
    The core logic. It now uses structured matrix_definitions from the UI.
    - matrix_definitions: A list of dicts, e.g.,
      [{'label': 'animal', 'type': 'matrix', 'options': ['cat', 'dog']},
       {'label': 'cfg', 'type': 'Δ', 'options': ['7', '9']}]
    """
    # Create a lookup dictionary for fast access to definitions by their label
    definitions_map = {d['label']: d for d in matrix_definitions}

    # 1. Parse references from prompts
    pos_refs = parse_axis_references(p.prompt)
    neg_refs = parse_axis_references(p.negative_prompt)
    for ref in pos_refs: ref['origin'] = 'positive'
    for ref in neg_refs: ref['origin'] = 'negative'

    # 2. Combine and enrich references with their full definitions
    all_axes = []
    prompt_refs = sorted(pos_refs + neg_refs, key=lambda x: x['start'])
    for ref in prompt_refs:
        if ref['label'] in definitions_map:
            # Merge the reference info (like 'token' and 'origin') with the full definition
            full_axis_def = {**definitions_map[ref['label']], **ref}
            all_axes.append(full_axis_def)
        else:
            print(f"[UPM Warning] Reference '{ref['token']}' found in prompt but has no definition in the Matrix Builder. It will be ignored.")

    final_axes_order = all_axes

    if not final_axes_order:
        # If no valid axes are found, just clean the prompt and return a single job
        cleaned_prompt = re.sub(REX_REFTAG, '', p.prompt)
        cleaned_neg_prompt = re.sub(REX_REFTAG, '', p.negative_prompt)
        return [{"pos_prompt": cleaned_prompt.strip(), "neg_prompt": cleaned_neg_prompt.strip(), "params": {}, "seed": p.seed}], None

    # 3. Separate axes into grid, page, and random
    random_axes = [axis for axis in final_axes_order if axis.get('type') == 'random']
    grid_axes = [axis for axis in final_axes_order if axis.get('type') != 'random']

    x_axis = grid_axes[-1] if len(grid_axes) > 0 else None
    y_axis = grid_axes[-2] if len(grid_axes) > 1 else None
    page_axes = grid_axes[:-2] if len(grid_axes) > 2 else []

    x_opts = x_axis['options'] if x_axis else ['']
    y_opts = y_axis['options'] if y_axis else ['']
    page_options_list = [axis['options'] for axis in page_axes]
    page_combinations = list(product(*page_options_list)) if page_options_list else [()]

    # 4. Build the job list
    job_list = []
    base_seed = p.seed if p.seed != -1 else random.randint(0, 2**32 - 1)

    for page_idx, page_vals in enumerate(page_combinations):
        for row_idx, y_val in enumerate(y_opts):
            for col_idx, x_val in enumerate(x_opts):
                job = {"pos_prompt": p.prompt, "neg_prompt": p.negative_prompt, "params": {}}

                # Handle random axes first
                for r_axis in random_axes:
                    chosen_val = random.choice(r_axis['options'])
                    key = 'pos_prompt' if r_axis['origin'] == 'positive' else 'neg_prompt'
                    job[key] = job[key].replace(r_axis['token'], chosen_val, 1)

                # Assign values for grid/page axes
                assignments = []
                if page_axes: assignments.extend(zip(page_axes, page_vals))
                if y_axis: assignments.append((y_axis, y_val))
                if x_axis: assignments.append((x_axis, x_val))

                for axis, value in assignments:
                    if axis['type'] == 'Δ':
                        job['params'][axis['label']] = value
                    else: # 'matrix' is a prompt substitution
                        key = 'pos_prompt' if axis['origin'] == 'positive' else 'neg_prompt'
                        job[key] = job[key].replace(axis['token'], value, 1)

                # Assign Seed
                image_index = (page_idx * len(y_opts) * len(x_opts)) + (row_idx * len(x_opts)) + col_idx
                if seed_behavior == "Fixed": job['seed'] = base_seed
                elif seed_behavior == "Iterate Per Image": job['seed'] = base_seed + image_index
                elif seed_behavior == "Iterate Per Row": job['seed'] = base_seed + (page_idx * len(y_opts)) + row_idx
                elif seed_behavior == "Random": job['seed'] = -1

                job_list.append(job)

    # 5. Final cleanup of all prompts
    for job in job_list:
        # Remove any remaining reference tags and legacy syntax
        job['pos_prompt'] = re.sub(REX_REFTAG, '', job['pos_prompt'])
        job['neg_prompt'] = re.sub(REX_REFTAG, '', job['neg_prompt'])
        job['pos_prompt'] = re.sub(REX_LEGACY_PARAM, '', re.sub(REX_LEGACY_MATRIX, '', job['pos_prompt'])).strip(' ,')
        job['neg_prompt'] = re.sub(REX_LEGACY_PARAM, '', re.sub(REX_LEGACY_MATRIX, '', job['neg_prompt'])).strip(' ,')

    # Prepare info for the grid drawing function
    grid_info = {"x_axis": x_axis, "y_axis": y_axis, "page_axes": page_axes, "page_combinations": page_combinations}

    # Add 'param_name' key for correct label formatting later
    if grid_info.get('x_axis'):
        grid_info['x_axis']['param_name'] = grid_info['x_axis']['label'] if grid_info['x_axis']['type'] == 'Δ' else 'prompt'
    if grid_info.get('y_axis'):
        grid_info['y_axis']['param_name'] = grid_info['y_axis']['label'] if grid_info['y_axis']['type'] == 'Δ' else 'prompt'
    for axis in grid_info.get('page_axes', []):
        axis['param_name'] = axis['label'] if axis['type'] == 'Δ' else 'prompt'

    return job_list, grid_info