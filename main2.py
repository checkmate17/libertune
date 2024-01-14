import os
import requests
import uvicorn
from pydantic import BaseModel
from typing import Optional, Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from libertune.adapter.adapter import Adapter
from libertune.model.mistral import MistralConfig, MistralModel
from libertune.model.mistral.configuration_mistral import (
    MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP,
)
from libertune.model.mixtral import MixtralConfig, MixtralModel
from libertune.model.mixtral.configuration_mixtral import (
    MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP,
)

MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP[
    "mistralai/Mistral-7B-v0.1"
] = "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json"
MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP[
    "Mistral-7B-Instruct-v0.2"
] = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP[
    "mistral-ai/Mixtral-8x7B"
] = "https://huggingface.co/mistral-ai/Mixtral-8x7B/resolve/main/config.json"
MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP[
    "Mixtral-8x7B-Instruct-v0.1"
] = "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/config.json"


app = FastAPI()

app.add_middleware(
    middleware_class=CORSMiddleware,
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
    response = requests.get(url, timeout=6000)
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
    response = requests.get(url, timeout=6000)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


@app.get("/get_models")
def get_model_list():
    try:
        models = {
            "Mistral": list(MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
            "Mixtral": list(MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
        }
        return JSONResponse(content=models, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


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
            config = MixtralConfig(**config_request.dict())
        if config_request.model_type == "mistral":
            config = MistralConfig(**config_request.dict())
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
