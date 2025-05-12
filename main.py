import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import urllib.parse
import time

import httpx
import PyPDF2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv
import openai

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

if not ASSISTANT_ID:
    raise ValueError("ASSISTANT_ID must be set in the .env file")

app = FastAPI(title="Resume Data Extractor")


class ResumeRequest(BaseModel):
    resume_url: HttpUrl = Field(..., description="URL to the resume PDF file")


async def download_pdf(url: str) -> Path:
    try:
        url_str = str(url)

        async with httpx.AsyncClient() as client:
            response = await client.get(url_str, follow_redirects=True)
            response.raise_for_status()

            fd, path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)

            return Path(path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download PDF: {str(e)}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")


async def process_with_assistant(text: str, assistant_id: str) -> Dict[str, Any]:
    try:
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=text
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(
                    status_code=500, detail=f"Assistant run failed with status: {run_status.status}")
            time.sleep(1)

        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        for message in messages.data:
            if message.role == "assistant":
                for content in message.content:
                    if content.type == "text":
                        import json
                        try:
                            text_content = content.text.value

                            if "```json" in text_content:
                                json_str = text_content.split(
                                    "```json")[1].split("```")[0].strip()
                            elif "```" in text_content:
                                json_str = text_content.split(
                                    "```")[1].split("```")[0].strip()
                            else:
                                json_str = text_content.strip()

                            json.loads(json_str)
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            raise HTTPException(
                                status_code=500, detail="Failed to parse JSON from assistant response")
                break

        raise HTTPException(
            status_code=500, detail="No valid response from assistant")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process with assistant: {str(e)}")


@app.post("/extract-resume")
async def extract_resume_data(request: ResumeRequest):
    pdf_path = await download_pdf(request.resume_url)

    try:
        text = extract_text_from_pdf(pdf_path)
        result = await process_with_assistant(text, ASSISTANT_ID)
        return result
    finally:
        os.unlink(pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
