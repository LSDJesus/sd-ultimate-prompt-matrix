import os
import modules.scripts as scripts

# --- CONSTANTS ---
# We define the potential locations for wildcard files.
# We check our own extension's folder first, then the standard A1111/Forge locations.
A1111_ROOT = os.path.abspath(os.path.join(scripts.basedir(), "..", ".."))
WILDCARD_DIRS = [
    os.path.join(scripts.basedir(), "wildcards"), # Our own extension's wildcard folder
    os.path.join(A1111_ROOT, "scripts", "wildcards"), # A common secondary location
    os.path.join(A1111_ROOT, "extensions", "sd-dynamic-prompts", "wildcards") # The official location
]

# --- FUNCTIONS ---

def get_wildcard_files():
    """
    Scans all known wildcard directories and returns a list of unique .txt files.
    This function discovers what wildcards are available to be chosen from a dropdown.
    """
    found_files = set()
    for directory in WILDCARD_DIRS:
        if os.path.exists(directory):
            try:
                files = [f for f in os.listdir(directory) if f.endswith('.txt')]
                for f in files:
                    found_files.add(f)
            except Exception as e:
                print(f"[UPM Wildcard Handler] Warning: Could not read directory {directory}. Error: {e}")
    
    # Return a sorted list for consistent UI display
    return sorted(list(found_files))

def get_wildcard_content(filename):
    """
    Finds a specific wildcard file by name in the known directories and returns its content as a list of strings.
    This function reads the lines from a selected wildcard file.
    """
    if not filename:
        return []

    for directory in WILDCARD_DIRS:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read lines, strip whitespace, and ignore comments or empty lines
                    lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
                    return lines
            except Exception as e:
                print(f"[UPM Wildcard Handler] Error reading file {file_path}. Error: {e}")
                return [] # Return empty list on error
    
    print(f"[UPM Wildcard Handler] Warning: Wildcard file '{filename}' not found in any known directory.")
    return []