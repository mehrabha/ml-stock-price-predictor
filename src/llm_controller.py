import docker
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from security import authenticate


load_dotenv()

BASE_ENDPOINT = os.getenv("BASE_PATH")
IMAGE_NAME = os.getenv("IMAGE_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
LLM_URL = os.getenv("LLM_URL")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT"))
PORT = int(os.getenv("PORT"))

app = FastAPI()
router = APIRouter(prefix=BASE_ENDPOINT)
docker_client = docker.from_env()

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are quantitative trading AI"
    temperature: float = .3
    cache_prompt: bool

@router.post("/start")
async def start_llm(user: str = Depends(authenticate)):
    # Builds the image if needed then launches the container

    try:
        docker_client.images.get(IMAGE_NAME)
        print(f"LLM image found, image={IMAGE_NAME}")
    except docker.errors.ImageNotFound:
        try:
            print(f"LLM image not found. Building from Dockerfile...")

            docker_client.images.build(os.getcwd(), tag=IMAGE_NAME, rm=True)
            print("Build complete.")
        except docker.errors.BuildError as e:
            print(f"Build failed for image={IMAGE_NAME}")

            for line in e.build_log:
                if 'stream' in line:
                    print(line['stream'].strip())
                elif 'error' in line:
                    print(line['error'].strip())
        except docker.errors.APIError as e:
            print(f"Docker server error: {e}")
    
    try:
        print(f"Starting container={CONTAINER_NAME}")
        container = docker_client.containers.get(CONTAINER_NAME)
        if container.status == 'running':
            print(f"Container={CONTAINER_NAME} already running! Refreshing...")
            container.stop()
        container.start()
    except docker.errors.NotFound:
        models_path = os.path.join(os.getcwd(), "models")

        # Create fresh container with GPU access and Volume mapping
        docker_client.containers.run(
            IMAGE_NAME,
            name=CONTAINER_NAME,
            detach=True,    # Run in background
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[['gpu']])],  # enable gpu
            ports={'8080/tcp': 8080},
            volumes={models_path: {"bind": "/models", "mode": "rw"}},    # mount models folder containing the image
            mem_limit="14g",
            shm_size="2g"
        )

    return await wait_for_ready()

@router.post("/chat")
async def chat(request: ChatRequest, user: str = Depends(authenticate)):
    # Invokes LLM with prompt

    req_body = {
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.prompt}
        ],
        "temperature": request.temperature,
        "cache_prompt": request.cache_prompt
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as http_client:    
        try:
            resp = await http_client.post(LLM_URL, json=req_body)
            resp.raise_for_status()
            response_json = resp.json()

            message = response_json["choices"][0]["message"]
            final_content = message["content"].strip()
            reasoning = message["reasoning_content"].strip()
            usage = response_json["usage"]

            return {
                "content": final_content,
                "reasoning": reasoning,
                "usage": usage
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error invoking LLM: {e}")

@router.post("/stop")
async def stop_llm(user : str = Depends(authenticate)):
    # Shuts down LLM container and release GPU

    try:
        container = docker_client.containers.get(CONTAINER_NAME)
        container.stop()
        return {"status": "terminated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping LLM: {e}")

async def wait_for_ready():
    # Polls the LLM health endpoint for 60 seconds before giving up

    for _ in range(60):
        try:
            async with httpx.AsyncClient(timeout=3.0) as http_client:
                res = await http_client.get("http://localhost:8080/health")
                if res.status_code == 200:
                    return {"status": "ready"}
                else:
                    raise HTTPException(status_code=503)
        except:
            await asyncio.sleep(2)
    
    raise HTTPException(status_code=504, detail="LLM failed to start in time.")

if __name__ == "__main__":
    import uvicorn

    app.include_router(router=router)
    
    # Runs the controller on port 5000
    uvicorn.run(app, host="0.0.0.0", port=PORT)