import os
import json
import requests
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from libertune.adapter.base import Adapter
from hugging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOST=os.getenv('HOST')
PORT = os.getenv('PORT')
STATIC_DIR = os.getenv('STATIC_DIR')
MISTRAL_DIR = os.getenv('MISTRAL_DIR')
MIXTRAL_DIR = os.getenv('MIXTRAL_DIR')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

app.mount("/static", StaticFiles(directory="static"), name="static")

class Load_Data_Request(BaseModel):
    data: str

@app.get('/')
def root():
    return {"message": "/docs for more info"}

@app.get('/data')
def download_file(load_data_request: Load_Data_Request):
    url = load_data_request.data
    filename = url.split("/")[-1]
    response = requests.get(url)
        # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

class HuggingFaceRequest(BaseModel):
    model: str
    data_filepath: str
    tokeen: str

@app.get('/huggingface')
def hugging_face_download(hugging_face_request: HuggingFaceRequest):
    url = hugging_face_request.data_filepath
    filename = url.split("/")[-1]
    response = requests.get(url)
        # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

class TrainModelRequest(BaseModel):
    model: str
    data_filepath: str
    
@app.get('/train')
def train_model():
    pass

class TrainAdapterConfig(BaseModel):
    input_dim: Optional[int]
    output_dim: Optional[int]
    base_model: Optional[str]
    dropout: Optional[float]

class TrainAdapterRequest(BaseModel):
    model: str
    data_filepath: str
    config: TrainAdapterConfig
    
@app.get('/adapter')
def trainAdapter(train_adapter_request: TrainAdapterRequest):
    if not train_adapter_request.config:
        raise ValueError("Missing config")
    try:
        adapter = Adapter(
            input_dim=train_adapter_request.config.input_dim=256,
            output_dim=train_adapter_request.config.output_dim=256,
            base_model=train_adapter_request.config.base_model='bert-base-uncased',
            dropout=train_adapter_request.config.dropout=0.1,
        )
        adapter.build()
    except Exception as e:
        print(e)




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)