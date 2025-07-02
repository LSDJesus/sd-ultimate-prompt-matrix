# Ultimate Prompt Matrix Extension for AUTOMATIC1111 & Forge

An advanced prompt matrix and parameter testing suite for AUTOMATIC1111, Forge, and other popular forks. This extension provides a powerful, dedicated UI tab for generating complex image grids and discovering the perfect settings for your concepts.

![UI Mockup](uimockup.png)

## Key Features

### 1. Full Parameter Matrixing
The flagship feature of this extension. Go beyond prompt variations and create grids that test different generation parameters against each other. Find the best settings for your subject with unparalleled precision.

- **Syntax:** `A cat <cfg(5|7.5|10)> <steps(20|30)>`
- **Supported Parameters:**
    - `cfg(...)` - CFG Scale
    - `steps(...)` - Sampling Steps
    - `sampler(...)` - The Sampler (e.g., `DPM++ 2M Karras`)
    - `scheduler(...)` - The Scheduler (e.g., `Karras`)
    - `checkpoint(...)` - The Checkpoint/Model file (e.g., `my_model.safetensors`)
- **Combine Everything:** Mix prompt and parameter matrices freely: `A <cat|dog> in a hat <cfg(7|9)>`

### 2. Intelligent & Powerful Grid Generation
- **Automatic Prompt Detection:** The script automatically detects matrix syntax in **both positive and negative prompts**. There's no need to tell it what to do—it just works. If you have matrices in both prompts, it will generate the full combinatorial product.
- **Axis Control & Visualizer:** Before you generate, click **"Pre-process & Calculate"** to see a preview of your grid axes. Use the "Swap" and "Cycle" buttons to rearrange the X, Y, and Page axes to get the exact grid layout you want without re-typing your prompt.
- **Advanced Seed Control:** A "Seed Behavior" dropdown gives you fine-grained control:
    - `Iterate Per Image`: Standard `seed, seed+1, ...` behavior.
    - `Iterate Per Row`: **(Highly Useful!)** All images in a row share the same seed. Perfect for keeping a character consistent while changing one variable (like clothing or background) on the X-axis.
    - `Fixed`: Every image uses the exact same seed.
    - `Random`: Every image gets a completely random seed.

### 3. Workflow & Safety Features
- **Preset System:** Save your entire setup—prompts, settings, sliders, and all—to a named preset. Load it back with a single click to re-run complex tests anytime.
- **Batch Pre-Calculator:** See exactly how many images will be generated and get an **estimated total runtime** *before* you start.
- **Large Batch Warning:** The main "Generate" button is disabled for very large jobs, preventing accidental resource lock-up. An override button lets you proceed with caution.
- **Dry Run Mode:** Validate your logic by printing a full list of all jobs (prompts, parameters, and seeds) to the console without using any GPU time.

### 4. UI & Quality of Life
- **Dedicated UI Tab:** A clean, professional, and spacious interface.
- **Tabbed Output:** Results are organized into a "Summary Grids" tab and an "All Images" tab for easy review.
- **LoRA Matrix Builder:** A visual interface to build complex LoRA matrices (`<lora:A:1|lora:B:1>`) and insert them into your prompt.
- **Help & Syntax Guide:** A built-in accordion explains all the syntax and features.

## Syntax Quick Reference

- **Prompt Matrix:** `a photo of a <cat|dog|bird>`
- **Parameter Matrix:** `a photo of a cat <cfg(5|7.5|10)>`
- **Random Word:** `a photo of a <random(cat|dog|bird)>`

The script reads matrix tags from your prompts from **end to beginning**:
- The **last** tag becomes the **X-axis**.
- The **second-to-last** tag becomes the **Y-axis**.
- **All other tags** become **Page-axes**.

## Installation
1.  In your Web UI, go to the **Extensions** tab.
2.  Click on the **Install from URL** sub-tab.
3.  Paste the following URL into the "URL for extension's git repository" field:
    ```
    https://github.com/LSDJesus/sd-ultimate-prompt-matrix
    ```
4.  Click the **"Install"** button.
5.  Wait for the confirmation message, then go to the **Installed** tab and click **"Apply and restart UI"**.

A new tab named **"Ultimate Matrix"** will now appear at the top of your Web UI.

## License
This project is licensed under the **MIT License**.

---
## Acknowledgements

This script was developed with the assistance of Google's AI.