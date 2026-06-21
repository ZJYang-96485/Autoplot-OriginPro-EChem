import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_MODEL = os.getenv("AI_MODEL", "gpt-5.4-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PLOT_CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "plot_title": {"type": "string"},
        "x_column": {"type": "string"},
        "y_column": {"type": "string"},
        "plot_type": {
            "type": "string",
            "enum": ["line", "scatter", "line_scatter", "bar"]
        },
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "x_min": {"type": ["number", "null"]},
        "x_max": {"type": ["number", "null"]},
        "y_min": {"type": ["number", "null"]},
        "y_max": {"type": ["number", "null"]},
        "template": {"type": "string"},
        "export_format": {
            "type": "string",
            "enum": ["png", "pdf", "both"]
        },
        "notes": {"type": "string"}
    },
    "required": [
        "plot_title",
        "x_column",
        "y_column",
        "plot_type",
        "x_label",
        "y_label",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "template",
        "export_format",
        "notes"
    ]
}

def parse_plot_request(user_request, column_names):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    columns_text = ", ".join(column_names)

    response = client.responses.create(
        model=AI_MODEL,
        instructions=(
            "You are an AI assistant for an OriginPro automatic plotting platform. "
            "Convert the user's request into a plot configuration. "
            "Only choose x_column and y_column from the provided column names. "
            "Use null for axis limits when the user does not specify them. "
            "Do not invent columns. "
            "Use default as the template unless the user requests a specific template."
        ),
        input=(
            f"Available columns: {columns_text}\n\n"
            f"User request: {user_request}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "plot_config",
                "schema": PLOT_CONFIG_SCHEMA,
                "strict": True
            }
        }
    )

    config = json.loads(response.output_text)

    if config["x_column"] not in column_names:
        raise ValueError(f"Invalid x_column: {config['x_column']}")

    if config["y_column"] not in column_names:
        raise ValueError(f"Invalid y_column: {config['y_column']}")

    return config