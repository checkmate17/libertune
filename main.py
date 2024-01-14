import os
import json
import requests
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from libertune.adapter.core_adapter import Adapter
from libertune.model.mistral import MistralConfig, configuration_mistral, MistralModel
from libertune.model.mixtral import MixtralConfig, configuration_mixtral, MixtralModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
STATIC_DIR = os.getenv("STATIC_DIR")
MISTRAL_DIR = os.getenv("MISTRAL_DIR")
MIXTRAL_DIR = os.getenv("MIXTRAL_DIR")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


app.mount("/static", StaticFiles(directory="static"), name="static")


class Load_Data_Request(BaseModel):
    data: str


@app.get("/")
def root():
    return {"message": "/docs for more info"}


@app.get("/data")
def download_file(load_data_request: Load_Data_Request):
    url = load_data_request.data
    filename = url.split("/")[-1]
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


class HuggingFaceRequest(BaseModel):
    model: str
    data_filepath: str
    token: str


@app.get("/huggingface")
def hugging_face_download(hugging_face_request: HuggingFaceRequest):
    url = hugging_face_request.data_filepath
    filename = url.split("/")[-1]
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


class TrainModelRequest(BaseModel):
    model: str
    data_filepath: str


class TrainingManager(BaseModel):
    model: Union[MistralModel, MixtralModel]
    config: Union[MistralConfig, MixtralConfig]
    weights_path: str


class TrainAdapterConfig(BaseModel):
    input_dim: Optional[int]
    output_dim: Optional[int]
    base_model: Optional[str]
    dropout: Optional[float]


class TrainAdapterRequest(BaseModel):
    model: str
    data_filepath: str
    config: TrainAdapterConfig


@app.post("/adapter")
def trainAdapter(train_adapter_request: TrainAdapterRequest):
    if not train_adapter_request:
        raise ValueError("Missing train_adapter_request")
    if not train_adapter_request.config:
        raise ValueError("Missing config")
    if (
        not train_adapter_request.config.input_dim
        or not train_adapter_request.config.output_dim
        or not train_adapter_request.config.base_model
        or not train_adapter_request.config.dropout
    ):
        raise ValueError("Missing input_dim")
    try:
        adapter = Adapter(
            train_adapter_request.config.input_dim,
            train_adapter_request.config.output_dim,
            train_adapter_request.config.base_model,
            train_adapter_request.config.dropout,
        )
        adapter.build()
        return JSONResponse(content=adapter.to_json_string(), status_code=200)
    except Exception as e:
        return JSONResponse(content=str(e), status_code=500)


@app.post("/mixtral/config")
def configure_mixtral(config_request: Union[MixtralConfig, MistralConfig]):
    config = None
    try:
        if config_request.model_type == "mixtral":
            config = configuration_mixtral.MixtralConfig(**config_request.dict())
        if config_request.model_type == "mistral":
            config = configuration_mistral.MistralConfig(**config_request.dict())
        print(config)

        return JSONResponse(content=config, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content=str(e), status_code=500)


@app.post("/load_model")
def modeling_model(model_request: Union[MixtralConfig, MistralConfig]):
    try:
        model = None
        if model_request.model_type == "mixtral":
            model = MixtralModel(**model_request.model_dump())
        if model_request.model_type == "mistral":
            model = MistralModel(**model_request.model_dump())

        print(model)

        return JSONResponse(content=model, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content=str(e), status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
