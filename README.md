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

3. To use Subscript and superscript: Copy paste and edit here.

E / V vs. RHE

NaNO₃

H₂O₂

mA cm⁻²

mol L⁻¹

s⁻¹

## AI Assistant Setup (Optional, Charge to OpenAI with a very small amount)

The AI Plot Assistant uses the OpenAI API. Each user must **provide their own OpenAI API key**.

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
AI_MODEL=gpt-5.4-mini
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

## ## AI Plot Assistant Prompt Template

The AI Plot Assistant is designed to help users fill plotting settings more efficiently. It is only for graph generation and plot formatting. It does not clean, merge, average, or restructure raw experimental data.

Users should prepare their dataset first. This includes combining files, assigning condition names, converting variables, averaging replicates, and creating any columns required for plotting. After the dataset is ready, the AI Plot Assistant can help generate the plotting configuration.

### General Prompt Template

```text
Create a [style] plot.

Use [X column name] as the X column and [Y column name] as the Y column.
Use a [line / scatter / bar / histogram / box] plot.
If needed, group the plot by [group column name].

Set the X-axis label to "[X-axis label]".
Set the Y-axis label to "[Y-axis label]".
Set the plot title to "[plot title]" or leave it empty.

Set X range from [minimum] to [maximum].
Set Y range from [minimum] to [maximum].

Use [color] for the line color and [color] for the marker color.
Use line width [value], marker size [value], and opacity [value].

Show or hide the legend.
Show or hide grid lines.
Use a full frame or open frame.
Use inward, outward, or inout ticks.

Set X tick mode to [auto / uniform / custom].
Set Y tick mode to [auto / uniform / custom].

Use figure width [value], figure height [value], and DPI [value].
```

### Example: Nature-Style Electrochemistry Plot

```text
Create a Nature-style electrochemistry line plot.

Use global_time_min as the X column and j_mA_cm2 as the Y column.
Use a line plot with small markers.

Set the X-axis label to "$t$ / min".
Set the Y-axis label to "$j$ / mA cm$^{-2}$".
Leave the plot title empty.

Set X range from 0 to 160.
Set Y range from -95 to 20.

Use black for the line color and black for the marker color.
Use line width 1.2, marker size 4, and opacity 1.0.

Do not show the legend unless there is more than one curve.
Show a full frame.
Use inward ticks.
Show top, bottom, left, and right ticks.
Do not show grid lines.

Use figure width 3.5, figure height 2.6, and DPI 300.
Use normal axis label weight.
Use axis label font size 7.
Use tick label font size 6.
Use title font size 7.
Use legend font size 6.
```

The Nature-style preset is intended as a convenient starting point for clean, compact scientific figures. It is not an official journal template. Users should always check the final formatting requirements of the target journal before submission.


### Questions or Suggestions can be forwarded to joeyang2 AT illinois.edu

