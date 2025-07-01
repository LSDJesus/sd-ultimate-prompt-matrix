# Ultimate Prompt Matrix Extension for AUTOMATIC1111 & Forge

An advanced prompt matrix extension for AUTOMATIC1111's Web UI and its popular forks. This extension creates its own dedicated tab, providing a powerful and uncluttered interface for generating complex image grids.

![UI Mockup](https://raw.githubusercontent.com/LSDJesus/sd-ultimate-prompt-matrix/main/UImockup.png)

## Why is this an Extension?
By running in its own dedicated tab, the Ultimate Prompt Matrix has a clean, spacious interface that will never be cut off or squished, no matter how many other extensions like ControlNet or ADetailer you have installed. This provides a vastly superior user experience and solves common UI space issues in forks like Forge.

## Features
- **Dedicated UI Tab:** A clean, professional, and spacious interface that is always easy to use.
- **Two Powerful Modes:**
    - **Permutation Mode:** For substitutions (e.g., `a <cat|dog>`). Perfect for A/B testing.
    - **Combination Mode:** For additive tags (e.g., `base, <tag1>, <tag2>`). Perfect for testing keyword influence.
- **Quality-of-Life Buttons:** Includes `🗑️` to clear prompts, `↙️` to paste prompts from the last generation, and `🔃` to refresh the current settings display.
- **Comprehensive Grid Control:**
    - Fully annotated grids with large, readable X, Y, and Page axis labels.
    - A "Mega-Grid" to summarize multi-dimensional tests.
    - Options to toggle annotations, margins, and use descriptive filenames.
- **Advanced Functionality:**
    - A "Dry Run" mode to validate your prompt logic *before* generating.
    - Always-on `<random(...)>` syntax for controlled chaos.
    - Optional integration with the Dynamic Prompts extension.
- **LoRA-Safe Syntax:** The `<...>` parser is designed to ignore `<lora:...>` syntax automatically.

## Installation
Installation is now incredibly simple via the Web UI:
1.  In your Web UI, go to the **Extensions** tab.
2.  Click on the **Install from URL** sub-tab.
3.  Paste the following URL into the "URL for extension's git repository" field:
    ```
    https://github.com/LSDJesus/sd-ultimate-prompt-matrix
    ```
4.  Click the **"Install"** button.
5.  Wait for the confirmation that the extension has installed, then go to the **Installed** tab.
6.  Click **"Apply and restart UI"**.

A new tab named **"Ultimate Matrix"** will now appear at the top of your Web UI.

## Usage
Go to the new "Ultimate Matrix" tab, enter your prompts, choose your settings, and click "Generate".

-   **Permutation Example:** `a <fantasy|sci-fi> painting of a <cat|dog>`
-   **Combination Example:** `a beautiful landscape, <masterpiece>, <cinematic lighting>`
-   **Random Example:** `a <random(cat|dog|bird)>`

## License
This project is licensed under the **MIT License**.

---
## Acknowledgements

This script was developed with the assistance of Google's AI.