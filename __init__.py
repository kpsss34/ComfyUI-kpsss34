"""
ComfyUI Sana Custom Node Package
- Modified to load nodes from both sana_nodes.py and i2iFlash.py
"""

# Import and rename mappings from the original sana_nodes.py to avoid conflicts
try:
    from .sana_nodes import NODE_CLASS_MAPPINGS as SANA_CLASS_MAPPINGS
    from .sana_nodes import NODE_DISPLAY_NAME_MAPPINGS as SANA_DISPLAY_MAPPINGS
    # Keep WEB_DIRECTORY as it is, assuming it's unique
    from .sana_nodes import WEB_DIRECTORY
    sana_loaded = True
except ImportError:
    print("Warning: Could not import from sana_nodes.py. Skipping.")
    SANA_CLASS_MAPPINGS = {}
    SANA_DISPLAY_MAPPINGS = {}
    WEB_DIRECTORY = "js" # A default fallback
    sana_loaded = False


# Import and rename mappings from your new i2iFlash.py
try:
    from .i2iFlash import NODE_CLASS_MAPPINGS as I2IFLASH_CLASS_MAPPINGS
    from .i2iFlash import NODE_DISPLAY_NAME_MAPPINGS as I2IFLASH_DISPLAY_MAPPINGS
    i2iflash_loaded = True
except ImportError:
    print("Warning: Could not import from i2iFlash.py. Check the file for errors.")
    I2IFLASH_CLASS_MAPPINGS = {}
    I2IFLASH_DISPLAY_MAPPINGS = {}
    i2iflash_loaded = False


# Merge the dictionaries from both files into final mappings for ComfyUI
NODE_CLASS_MAPPINGS = {**SANA_CLASS_MAPPINGS, **I2IFLASH_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**SANA_DISPLAY_MAPPINGS, **I2IFLASH_DISPLAY_MAPPINGS}

# This special variable tells ComfyUI what to export to the main program
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Optional: Add a confirmation message in the console
print("---")
print("Loading Custom Nodes: ComfyUI-kpsss34-Sana")
if sana_loaded:
    print("  - Successfully loaded nodes from sana_nodes.py")
if i2iflash_loaded:
    print("  - Successfully loaded nodes from i2iFlash.py")
print(f"  - Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print("---")
