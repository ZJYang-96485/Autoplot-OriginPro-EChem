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

3. AI Assistant Setup (Optional, Charge to OpenAI with a very small amount)

The AI Plot Assistant uses the OpenAI API. Each user must **provide their own OpenAI API key**.

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
AI_MODEL=gpt-5.4-mini

4. Questions or Suggestions can be forwarded to joeyang2 AT illinois.edu

