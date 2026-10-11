import re
import os
import sys
import json
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """Execute code and capture stdout and stderr."""
    stdout = StringIO()
    stderr = StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<string>", "exec"), {})

        return {
            "success": True,
            "output": stdout.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }



def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """Ask the LLM to identify the faulty source-code line."""

    prompt = f"""
Analyze the Python code and traceback below.

Return the exact line number(s) in the submitted code that caused
the exception. Count lines starting at 1.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return JSON: {{"error_lines": [3]}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify Python source-code error line numbers "
                    "using the traceback. Return valid JSON only, "
                    'with the key "error_lines" containing integers.'
                )
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("AI returned an empty response")

    result = ErrorAnalysis.model_validate_json(content)

    line_count = len(code.splitlines())

    return sorted({
        line for line in result.error_lines
        if 1 <= line <= line_count
    })



@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    error_lines = []

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )
    except Exception as e:
        # Show the actual AI failure in the terminal.
        print(f"AI analysis failed: {type(e).__name__}: {e}")

    # Fallback: extract source line numbers from the traceback.
    if not error_lines:
        matches = re.findall(
            r'File "<string>", line (\d+)',
            execution["output"]
        )

        line_count = len(request.code.splitlines())

        error_lines = sorted({
            int(line) for line in matches
            if 1 <= int(line) <= line_count
        })

    return {
        "error": error_lines,
        "result": execution["output"]
    }

@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}
