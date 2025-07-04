import re
import os
import modules.scripts as scripts
import modules.shared as shared 
import gradio as gr # Added: Import gradio for gr.update()

def paste_last_prompts():
    # Initialize values to gr.update() to signify "no change" if parameter not found
    prompt_val = gr.update()
    neg_prompt_val = gr.update()
    steps_val = gr.update()
    sampler_name_val = gr.update()
    scheduler_val = gr.update()
    cfg_scale_val = gr.update()
    seed_val = gr.update()
    width_val = gr.update()
    height_val = gr.update()
    model_name_val = gr.update() # Will be applied via shared.opts, not direct UI update
    vae_name_val = gr.update()   # Will be applied via shared.opts, not direct UI update

    info_sources = [
        (hasattr(shared, 'last_info') and shared.last_info, "shared.last_info"),
        (os.path.exists(os.path.join(scripts.basedir(), "params.txt")), "params.txt")
    ]

    info_string = None
    source_name = None

    for has_info, current_source_name in info_sources:
        if has_info:
            try:
                if current_source_name == "shared.last_info":
                    info_string = shared.last_info
                elif current_source_name == "params.txt":
                    params_file_path = os.path.join(scripts.basedir(), "params.txt")
                    with open(params_file_path, 'r', encoding='utf-8') as f:
                        info_string = f.read()
                
                source_name = current_source_name
                break # Found a valid info source, break the loop
            except Exception as e:
                print(f"[Ultimate Matrix] Error reading {current_source_name}: {e}")
                info_string = None # Reset info_string if reading failed

    if not info_string:
        print("[Ultimate Matrix] Could not find last generation info from any source.")
        # Return gr.update() for all UI components when no info is found, in the correct order
        return (prompt_val, neg_prompt_val, steps_val, sampler_name_val, scheduler_val, 
                cfg_scale_val, seed_val, width_val, height_val)

    # Parse Prompts
    neg_match = re.search(r'Negative prompt: (.*?)\n(?:Steps:|$)', info_string, re.DOTALL)
    if neg_match:
        neg_prompt_val = neg_match.group(1).strip()
        prompt_val = info_string.split('Negative prompt:')[0].strip()
    else:
        prompt_val = info_string.split('\nSteps:')[0].strip() if '\nSteps:' in info_string else info_string.strip()

    # Parse other parameters
    param_regexes = {
        'steps': r'Steps: (\d+)',
        'sampler_name': r'Sampler: ([^,\n]+)',
        'scheduler': r'Schedule type: ([^,\n]+)',
        'cfg_scale': r'CFG scale: ([\d.]+)',
        'seed': r'Seed: (-?\d+)',
        'size': r'Size: (\d+x\d+)', # To be split into width/height
        'model_name': r'Model: ([^,\n]+)',
        'vae_name': r'Module 1: ([^,\n]+)' 
    }

    for param, regex in param_regexes.items():
        match = re.search(regex, info_string)
        if match:
            value = match.group(1).strip()
            if param == 'size':
                try:
                    width_str, height_str = value.split('x')
                    width_val = int(width_str)
                    height_val = int(height_str)
                except ValueError:
                    print(f"[Ultimate Matrix] Could not parse size: {value}")
            elif param == 'steps':
                steps_val = int(value)
            elif param == 'seed':
                seed_val = int(value)
            elif param == 'cfg_scale':
                cfg_scale_val = float(value)
            elif param == 'sampler_name':
                sampler_name_val = value
            elif param == 'scheduler':
                scheduler_val = value
            elif param == 'model_name':
                model_name_val = value
            elif param == 'vae_name':
                vae_name_val = value

    print(f"[Ultimate Matrix] Successfully loaded parameters from {source_name}.")
    
    # Apply Model and VAE via shared.opts (these don't update UI dropdowns directly)
    if model_name_val is not gr.update():
        shared.opts.data['sd_model_checkpoint'] = model_name_val
        print(f"[Ultimate Matrix] Pasted Model: {model_name_val} (will apply on next generation)")
    if vae_name_val is not gr.update():
        shared.opts.data['sd_vae'] = vae_name_val
        print(f"[Ultimate Matrix] Pasted VAE: {vae_name_val} (will apply on next generation)")

    # Return values in the exact order expected by ultimate_prompt_matrix_ext.py's .click outputs
    return (prompt_val, neg_prompt_val, steps_val, sampler_name_val, scheduler_val, 
            cfg_scale_val, seed_val, width_val, height_val)