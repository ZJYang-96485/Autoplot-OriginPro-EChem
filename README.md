# Autoplot-OriginPro-EChem

Author: Zihan (Joe) Yang, joeyang2 AT illinois.edu

This repository is utilized for developing automated graphing tools via OriginPro, with a specialization in Electrochemistry experiments, with extra supports on data cleaning, data analysis, statistical modeling, and machine learning.

## Requirements

Before running the plotting scripts, make sure **OriginPro is already installed and activated on your Windows computer**. The Python package `originpro` only provides the connection between Python and OriginPro. It does **not** install the OriginPro software itself.

Required software:

- Windows computer (Mac currently does not support OriginPro)

- OriginPro installed locally

- Python 3.9 or newer

- Git

- VS Code


## Initialize the APP.

Kindly using this automation tool with your dataset via these following steps,

1. Please check out the 'requirements.txt' file. Make sure the environment is properly loaded.

#Open this repository via your own VSCode terminal:

Run 'pip install -r requirements.txt'

2. Use the Automation tool via terminal:

Run 'python app.py'

3. To use Subscript and superscript: Copy paste and edit here. Author Note: Most of them have been adapted in the code.

E / V vs. RHE, NaNO₃, H₂O₂, mA cm⁻², mol L⁻¹, s⁻¹

In addition, LaTeX is enabled.

## AI Assistant Setup (Optional, Charge to OpenAI with a very small amount)

The AI Plot Assistant uses the OpenAI API. Each user must **provide their own OpenAI API key**.

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here

AI_MODEL=gpt-5.4-mini
AI_PLOT_MODEL=gpt-5.4-mini
AI_WORKFLOW_MODEL=gpt-5.4-mini
```

If you are new to the OpenAI API, go to `https://platform.openai.com` and create your own API key. You also need to set up billing or add credits in your OpenAI API account before the assistant can make requests.


## AI Plotting Details

The AI Plot Assistant supports several common scientific plotting style presets:

- Nature-style figures
- Science-style figures
- ACS-style figures
- RSC-style figures
- Elsevier-style figures
- IEEE-style figures

These presets are intended to provide convenient starting points for scientific plotting. **They are not official journal templates**, so users should still check the final formatting requirements of the target journal before submission.

Additional general presets include publication, thesis, presentation, poster, monochrome, colorblind-safe, and dark-background styles.

## AI Plot Assistant Prompt Template

The AI Plot Assistant is designed to help users fill plotting settings more efficiently. It is only for graph generation and plot formatting. It does not clean, merge, average, or restructure raw experimental data.

Users should prepare their dataset first. This includes combining files, assigning condition names, converting variables, averaging replicates, and creating any columns required for plotting. After the dataset is ready, the AI Plot Assistant can help generate the plotting configuration.

## Guideline 1: Using AI to Create a Data-Processing Prompt

This software is designed to process different electrochemical datasets, but the correct workflow depends strongly on the file structure, naming convention, experiment type, and required conversions. Users should not reuse a prompt from another dataset without adapting it.

When asking an AI assistant to create a data-processing prompt, provide the experimental structure instead of only asking for a generic workflow. The AI should be asked to identify file types, readable columns, condition groups, replicate groups, sequence order, stitching rules, conversion formulas, and expected output datasets.

A good data-processing prompt should clearly specify:

* total number of files
* expected number of files per condition or subset
* file type, when known
* whether files are independent replicates or sequential segments
* how conditions should be inferred from filenames or columns
* how sequence numbers should be interpreted
* which numbers in filenames should be ignored, such as rpm, layer count, dates, or sample IDs
* required unit conversions
* electrode area, reference potential shift, pH correction, or other electrochemical constants
* required output columns
* whether the software should stop and report diagnostics when files are missing or ambiguous

The prompt should instruct the software not to silently guess when file assignment is ambiguous. It should also prevent unsafe operations such as stretching, mirroring, extrapolating, or fabricating missing data.

Recommended AI instruction:

1. Consult your own AI for prompt suggestion.

“Create a data-processing prompt for this software. The prompt should inspect the uploaded files, infer the safest workflow, define condition mapping, determine whether files should be stitched or averaged, apply the required electrochemical conversions, and stop with diagnostics if the file structure is ambiguous. Do not make assumptions that are not supported by filenames, metadata, or column names.”

2. Run workflow with prompts in this software app.

## Guideline 2: Using AI to Create a Plotting Prompt

Plotting prompts should be created after the data-processing workflow is defined. A plotting prompt should describe how to visualize the processed dataset, not how to repair or reinterpret incorrectly processed data.

When asking an AI assistant to create a plotting prompt, specify which processed dataset should be used, which columns should be mapped to X, Y, and grouping variables, and whether the plot should show raw curves, stitched curves, averaged curves, or replicate statistics.

A good plotting prompt should clearly specify:

* which dataset type to use, such as `combined_data` or `averaged_replicates`
* X column
* Y column
* group or condition column
* whether to show raw files, stitched sequences, averaged curves, or error bars
* axis labels and units
* axis ranges only when supported by the data
* legend labels and order
* color rules
* marker and line style
* grid, frame, tick, and typography preferences
* whether a secondary or custom axis is scientifically justified
* what the software should do when required columns, conditions, or data coverage are missing

The plotting prompt should not ask the software to fix incomplete processing. If a condition does not cover the requested range, or if a required column is missing, the plotter should stop and report the issue instead of stretching curves, smoothing artifacts, or creating misleading figures.

Recommended AI instruction:

1. Consult with your own AI.

“Create a plotting prompt for this software based on the processed dataset. The prompt should map the correct columns, preserve the experimental data shape, apply publication-style formatting, and include validation checks for required columns, conditions, and data coverage. The prompt should stop with diagnostics instead of generating a misleading plot when the selected dataset is incomplete or incompatible.”

2. Run in this software.


The Nature-style preset is intended as a convenient starting point for clean, compact scientific figures. It is not an official journal template. Users should always check the final formatting requirements of the target journal before submission.


### Questions or Suggestions can be forwarded to joeyang2 AT illinois.edu

