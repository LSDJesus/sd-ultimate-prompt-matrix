import re

def sanitize_filename(text):
    if not text: return ""
    return re.sub(r'[\\/*?:"<>|]', '', text)[:100]

def paste_last_prompts():
    # IMPORTANT: Import modules.shared inside the function to get the live, up-to-date object.
    import modules.shared as shared 
    
    # Defensive check: Does the data even exist?
    if hasattr(shared, 'last_info') and shared.last_info:
        try:
            # The rest of the parsing logic...
            ...
            return pos_prompt_part, neg_prompt_part
        except Exception as e:
            # Fails gracefully if the text format is unexpected
            print(f"[Ultimate Matrix] Error parsing last infotext: {e}")
            return "", ""
            
    # Fails gracefully if last_info is missing entirely
    print("[Ultimate Matrix] Paste button: shared.last_info not found or is empty.")
    return "", ""