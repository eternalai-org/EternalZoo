"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import logging
import httpx
import asyncio
import time
import json
import uuid
# Import configuration settings
from json_repair import repair_json
from typing import Dict, Any, Optional
from eternal_zoo.config import DEFAULT_CONFIG
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from eternal_zoo.manager import EternalZooManager, EternalZooServiceError

# Import schemas from schema.py
from eternal_zoo.schema import (
    Choice,
    Message,
    ModelCard,
    ModelList,
    ModelPermission,
    LoraConfigRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatCompletionChunk,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()



# Add CORS middleware to allow localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:*", "http://127.0.0.1", "http://127.0.0.1:*"],  # Allow localhost only
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize EternalZoo Manager for service management
eternal_zoo_manager = EternalZooManager()


# Performance constants from config
IDLE_TIMEOUT = DEFAULT_CONFIG.performance.IDLE_TIMEOUT
UNLOAD_CHECK_INTERVAL = DEFAULT_CONFIG.performance.UNLOAD_CHECK_INTERVAL
UNLOAD_LOG_INTERVAL = DEFAULT_CONFIG.performance.UNLOAD_LOG_INTERVAL
UNLOAD_MAX_CONSECUTIVE_ERRORS = DEFAULT_CONFIG.performance.UNLOAD_MAX_CONSECUTIVE_ERRORS
UNLOAD_ERROR_SLEEP_MULTIPLIER = DEFAULT_CONFIG.performance.UNLOAD_ERROR_SLEEP_MULTIPLIER
STREAM_CLEANUP_INTERVAL = DEFAULT_CONFIG.performance.STREAM_CLEANUP_INTERVAL
STREAM_CLEANUP_ERROR_SLEEP = DEFAULT_CONFIG.performance.STREAM_CLEANUP_ERROR_SLEEP
STREAM_STALE_TIMEOUT = DEFAULT_CONFIG.performance.STREAM_STALE_TIMEOUT
MODEL_SWITCH_VERIFICATION_DELAY = DEFAULT_CONFIG.performance.MODEL_SWITCH_VERIFICATION_DELAY
MODEL_SWITCH_MAX_RETRIES = DEFAULT_CONFIG.performance.MODEL_SWITCH_MAX_RETRIES
MODEL_SWITCH_STREAM_TIMEOUT = DEFAULT_CONFIG.performance.MODEL_SWITCH_STREAM_TIMEOUT
QUEUE_BACKPRESSURE_TIMEOUT = DEFAULT_CONFIG.performance.QUEUE_BACKPRESSURE_TIMEOUT
PROCESS_CHECK_INTERVAL = DEFAULT_CONFIG.performance.PROCESS_CHECK_INTERVAL
SHUTDOWN_TASK_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_TASK_TIMEOUT
SHUTDOWN_SERVER_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_SERVER_TIMEOUT
SHUTDOWN_CLIENT_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_CLIENT_TIMEOUT
SERVICE_START_TIMEOUT = DEFAULT_CONFIG.performance.SERVICE_START_TIMEOUT
POOL_CONNECTIONS = DEFAULT_CONFIG.performance.POOL_CONNECTIONS
POOL_KEEPALIVE = DEFAULT_CONFIG.performance.POOL_KEEPALIVE
HTTP_TIMEOUT = DEFAULT_CONFIG.performance.HTTP_TIMEOUT
STREAM_TIMEOUT = DEFAULT_CONFIG.performance.STREAM_TIMEOUT
MAX_RETRIES = DEFAULT_CONFIG.performance.MAX_RETRIES
RETRY_DELAY = DEFAULT_CONFIG.performance.RETRY_DELAY
MAX_QUEUE_SIZE = DEFAULT_CONFIG.performance.MAX_QUEUE_SIZE
HEALTH_CHECK_INTERVAL = DEFAULT_CONFIG.performance.HEALTH_CHECK_INTERVAL
STREAM_CHUNK_SIZE = DEFAULT_CONFIG.performance.STREAM_CHUNK_SIZE

# Utility functions
def get_service_info() -> Dict[str, Any]:
    """Get service info from EternalZooManager with error handling."""
    try:
        return eternal_zoo_manager.get_service_info()
    except EternalZooServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

def convert_request_to_dict(request) -> Dict[str, Any]:
    """Convert request object to dictionary, supporting both Pydantic v1 and v2."""
    return request.model_dump() if hasattr(request, "model_dump") else request.dict()

def validate_model_field(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate that the model field is present and matches one of the available model hashes."""

    requested_model = request_data["model"]
    
    try:
        service_info = get_service_info()
        models = list(service_info.get("models", {}).keys())
        
        # Check if the requested model hash is in the models dictionary
        if requested_model not in models:
            logger.warning(f"Requested model '{requested_model}' not found in available models: {models}. Using model {models[0]} instead.")
            return service_info, models[0] 
        
        logger.debug(f"Model validation passed for '{requested_model}'")
        
        # Return the service info to avoid redundant calls
        return service_info, requested_model
            
    except HTTPException as e:
        # If we can't get service info (503), it means no model is running
        if e.status_code == 503:
            logger.warning(f"Service info not available, requested model '{requested_model}' cannot be validated")
            raise HTTPException(
                status_code=400,
                detail="Service is not running"
            )
        # Re-raise other HTTPExceptions
        raise

def generate_request_id() -> str:
    """Generate a short request ID for tracking."""
    return str(uuid.uuid4())[:8]

def generate_chat_completion_id() -> str:
    """Generate a chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex}"

# Service Functions
class ServiceHandler:
    """Handler class for making requests to the underlying service."""
    
    @staticmethod
    def _create_vision_error_response(request: ChatCompletionRequest, content: str):
        """Create error response for vision requests when multimodal is not supported."""
        if request.stream:
            async def error_stream():
                chunk = {
                    "id": generate_chat_completion_id(),
                    "choices": [{
                        "delta": {"content": content},
                        "finish_reason": "stop",
                        "index": 0
                    }],
                    "created": int(time.time()),
                    "model": request.model,
                    "object": "chat.completion.chunk"
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return ChatCompletionResponse(
                id=generate_chat_completion_id(),
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[Choice(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content=content)
                )]
            )
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest):
        """Generate a response for chat completion requests, supporting both streaming and non-streaming."""
        service_info = get_service_info()
        ai_services = service_info.get("ai_services", [])
        model_id = request.model
        chat_models = []

        for ai_service in ai_services:
            task = ai_service["task"]
            if task == "chat":
                chat_models.append(ai_service)

        if len(chat_models) == 0:
            raise HTTPException(status_code=404, detail=f"No chat model found")
        
        if len(chat_models) > 1:
            for chat_model in chat_models:
                if chat_model["model_id"] == model_id:
                    port = chat_model["port"]
                    break
                else:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        if request.is_vision_request():
            if not service_info.get("multimodal", False):
                content = "Unfortunately, I'm not equipped to interpret images at this time. Please provide a text description if possible."
                return ServiceHandler._create_vision_error_response(request, content)
                
        request.clean_messages()
        request.enhance_tool_messages()
        request_dict = convert_request_to_dict(request)
        
        if request.stream:
            # For streaming requests, generate a stream ID 
            stream_id = generate_request_id()
            
            logger.debug(f"Creating streaming response for model {request.model} with stream ID {stream_id}")
            
            # Registration happens inside the generator to avoid race conditions
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict, stream_id),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        logger.debug(f"Making non-streaming request for model {request.model}")
        response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)
        return ChatCompletionResponse(
            id=response_data.get("id", generate_chat_completion_id()),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest, service_info: Optional[Dict[str, Any]] = None):
        """Generate a response for embedding requests."""
        # Use provided service_info or get it if not provided
        if service_info is None:
            service_info = get_service_info()
        
        port = get_service_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=response_data.get("data", []),
            model=request.model
        )
    
    @staticmethod
    async def generate_image_response(request: ImageGenerationRequest, service_info: Optional[Dict[str, Any]] = None):
        """Generate a response for image generation requests."""
        # Use provided service_info or get it if not provided
        if service_info is None:
            service_info = get_service_info()
        
        port = get_service_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(port, "/v1/images/generations", request_dict)
        return ImageGenerationResponse(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", [])
        )
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a non-streaming API call to the specified endpoint and return the JSON response."""
        try:
            response = await app.state.client.post(
                f"http://localhost:{port}{endpoint}", 
                json=data,
                timeout=HTTP_TIMEOUT
            )
            logger.info(f"Received response with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Error: {response.status_code} - {error_text}")
                if response.status_code < 500:
                    raise HTTPException(status_code=response.status_code, detail=error_text)
            
            # Cache JSON parsing to avoid multiple calls
            json_response = response.json()
            return json_response
            
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def _stream_generator(port: int, data: Dict[str, Any], stream_id: str):
        """Generator for streaming responses from the service."""
        try:
            # Register stream at the start of actual streaming to avoid race conditions
            await RequestProcessor.register_stream(stream_id)
            logger.debug(f"Starting stream {stream_id}")
            
            buffer = ""
            tool_calls = {}
            
            def _extract_json_data(line: str) -> Optional[str]:
                """Extract JSON data from SSE line, return None if not valid data."""
                line = line.strip()
                if not line or line.startswith(': ping'):
                    return None
                if line.startswith('data: '):
                    return line[6:].strip()
                return None
            
            def _process_tool_call_delta(delta_tool_call, tool_calls: dict):
                """Process tool call delta and update tool_calls dict."""
                tool_call_index = str(delta_tool_call.index)
                if tool_call_index not in tool_calls:
                    tool_calls[tool_call_index] = {"arguments": ""}
                
                if delta_tool_call.id is not None:
                    tool_calls[tool_call_index]["id"] = delta_tool_call.id
                    
                function = delta_tool_call.function
                if function.name is not None:
                    tool_calls[tool_call_index]["name"] = function.name
                if function.arguments is not None:
                    tool_calls[tool_call_index]["arguments"] += function.arguments
            
            def _create_tool_call_chunks(tool_calls: dict, chunk_obj):
                """Create tool call chunks for final output - yields each chunk separately."""
                chunk_obj_copy = chunk_obj.copy()
                
                for tool_call_index, tool_call in tool_calls.items():
                    try:
                        tool_call_obj = json.loads(repair_json(json.dumps(tool_call)))
                        tool_call_id = tool_call_obj.get("id", None)
                        tool_call_name = tool_call_obj.get("name", "")
                        tool_call_arguments = tool_call_obj.get("arguments", "")
                        if tool_call_arguments == "":
                            tool_call_arguments = "{}"
                        function_call = ChoiceDeltaFunctionCall(
                            name=tool_call_name,
                            arguments=tool_call_arguments
                        )
                        delta_tool_call = ChoiceDeltaToolCall(
                            index=int(tool_call_index),
                            id=tool_call_id,
                            function=function_call,
                            type="function"
                        )
                        chunk_obj_copy.choices[0].delta.content = None
                        chunk_obj_copy.choices[0].delta.tool_calls = [delta_tool_call]  
                        chunk_obj_copy.choices[0].finish_reason = "tool_calls"
                        yield f"data: {chunk_obj_copy.json()}\n\n"
                    except Exception as e:
                        logger.error(f"Failed to create tool call chunk in {stream_id}: {e}")
                        chunk_obj_copy.choices[0].delta.content = None
                        chunk_obj_copy.choices[0].delta.tool_calls = []
                        yield f"data: {chunk_obj_copy.json()}\n\n"
                        
            try:
                async with app.state.client.stream(
                    "POST", 
                    f"http://localhost:{port}/v1/chat/completions", 
                    json=data,
                    timeout=STREAM_TIMEOUT
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = f"data: {{\"error\":{{\"message\":\"{error_text.decode()}\",\"code\":{response.status_code}}}}}\n\n"
                        logger.error(f"Streaming error for {stream_id}: {response.status_code} - {error_text.decode()}")
                        yield error_msg
                        return
                    
                    async for chunk in response.aiter_bytes():
                        buffer += chunk.decode('utf-8', errors='replace')
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            json_str = _extract_json_data(line)
                            
                            if json_str is None:
                                continue
                                
                            if json_str == '[DONE]':
                                yield 'data: [DONE]\n\n'
                                continue
                            
                            try:
                                chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                                choice = chunk_obj.choices[0]
                                
                                # Handle finish reason - output accumulated tool calls
                                if choice.finish_reason and tool_calls:
                                    for tool_call_chunk in _create_tool_call_chunks(tool_calls, chunk_obj):
                                        yield tool_call_chunk

                                    yield f"data: [DONE]\n\n"
                                    return
                                
                                # Handle tool call deltas
                                if choice.delta.tool_calls:
                                    _process_tool_call_delta(choice.delta.tool_calls[0], tool_calls)
                                else:
                                    # Regular content chunk
                                    yield f"data: {chunk_obj.json()}\n\n"
                                        
                            except Exception as e:
                                logger.error(f"Failed to parse streaming chunk in {stream_id}: {e}")
                                # Pass through unparseable data (except ping messages)
                                if not line.strip().startswith(': ping'):
                                    yield f"data: {line}\n\n"
                                
                    # Process any remaining buffer content
                    if buffer.strip():
                        json_str = _extract_json_data(buffer)
                        if json_str and json_str != '[DONE]':
                            try:
                                chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                                yield f"data: {chunk_obj.json()}\n\n"
                            except Exception as e:
                                logger.error(f"Failed to parse trailing chunk in {stream_id}: {e}")
                        elif json_str == '[DONE]':
                            yield 'data: [DONE]\n\n'
                            
            except Exception as e:
                logger.error(f"Streaming error for {stream_id}: {e}")
                yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
                
        except Exception as e:
            logger.error(f"Critical error in stream generator {stream_id}: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
        finally:
            # Always unregister the stream when done
            try:
                await RequestProcessor.unregister_stream(stream_id)
                logger.debug(f"Stream {stream_id} completed and unregistered")
            except Exception as e:
                logger.error(f"Error unregistering stream {stream_id}: {e}")


# Request Processor
class RequestProcessor:
    """Process requests sequentially using a queue to accommodate single-threaded backends."""
    
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    processing_lock = asyncio.Lock()
    
    # Track active streams to prevent model switching during streaming
    active_streams = set()
    active_streams_lock = asyncio.Lock()
    stream_timestamps = {}  # Track when streams were registered
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/v1/images/generations": (ImageGenerationRequest, ServiceHandler.generate_image_response),
        "/images/generations": (ImageGenerationRequest, ServiceHandler.generate_image_response),
    }
    
    @staticmethod
    async def register_stream(stream_id: str):
        """Register an active stream to prevent model switching."""
        async with RequestProcessor.active_streams_lock:
            RequestProcessor.active_streams.add(stream_id)
            RequestProcessor.stream_timestamps[stream_id] = time.time()
            logger.debug(f"Registered active stream {stream_id}, total active: {len(RequestProcessor.active_streams)}")
    
    @staticmethod
    async def unregister_stream(stream_id: str):
        """Unregister a completed stream."""
        async with RequestProcessor.active_streams_lock:
            RequestProcessor.active_streams.discard(stream_id)
            RequestProcessor.stream_timestamps.pop(stream_id, None)
            logger.debug(f"Unregistered stream {stream_id}, total active: {len(RequestProcessor.active_streams)}")
    
    @staticmethod
    async def has_active_streams() -> bool:
        """Check if there are any active streams."""
        async with RequestProcessor.active_streams_lock:
            return len(RequestProcessor.active_streams) > 0
    
    @staticmethod
    async def terminate_active_streams():
        """Forcefully terminate all active streams."""
        async with RequestProcessor.active_streams_lock:
            terminated_count = len(RequestProcessor.active_streams)
            RequestProcessor.active_streams.clear()
            RequestProcessor.stream_timestamps.clear()
            logger.warning(f"Force terminated {terminated_count} active streams")
    
    @staticmethod
    async def wait_for_streams_to_complete(timeout: float = MODEL_SWITCH_STREAM_TIMEOUT, force_terminate: bool = False):
        """Wait for all active streams to complete before proceeding."""
        start_time = time.time()
        initial_count = len(RequestProcessor.active_streams)
        
        if initial_count > 0:
            logger.info(f"Waiting for {initial_count} active streams to complete (timeout: {timeout}s)")
        
        check_interval = 0.1
        last_log_time = start_time
        
        while await RequestProcessor.has_active_streams():
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed > timeout:
                remaining_streams = len(RequestProcessor.active_streams)
                if force_terminate:
                    logger.warning(f"Timeout waiting for streams to complete after {elapsed:.1f}s, "
                                  f"force terminating {remaining_streams} active streams")
                    # Log the active stream IDs for debugging
                    async with RequestProcessor.active_streams_lock:
                        active_stream_ids = list(RequestProcessor.active_streams)
                        logger.warning(f"Force terminating stream IDs: {active_stream_ids}")
                    
                    await RequestProcessor.terminate_active_streams()
                    break
                else:
                    logger.error(f"Timeout waiting for streams to complete after {elapsed:.1f}s, "
                                f"{remaining_streams} still active. Refusing to proceed without force_terminate=True")
                    # Log the active stream IDs for debugging
                    async with RequestProcessor.active_streams_lock:
                        active_stream_ids = list(RequestProcessor.active_streams)
                        logger.error(f"Active stream IDs: {active_stream_ids}")
                    return False
            
            # Log progress every 5 seconds
            if current_time - last_log_time >= 5.0:
                remaining_streams = len(RequestProcessor.active_streams)
                logger.info(f"Still waiting for {remaining_streams} streams to complete "
                           f"(elapsed: {elapsed:.1f}s/{timeout}s)")
                last_log_time = current_time
            
            await asyncio.sleep(check_interval)
        
        final_count = len(RequestProcessor.active_streams)
        if initial_count > 0:
            logger.info(f"Stream wait completed. Initial: {initial_count}, Final: {final_count}")
        
        return True
    
    @staticmethod
    async def cleanup_stale_streams():
        """Clean up any stale streams that might be stuck in the active list."""
        current_time = time.time()
        stale_streams = []
        
        async with RequestProcessor.active_streams_lock:
            for stream_id in list(RequestProcessor.active_streams):
                if stream_id in RequestProcessor.stream_timestamps:
                    if current_time - RequestProcessor.stream_timestamps[stream_id] > STREAM_STALE_TIMEOUT:
                        stale_streams.append(stream_id)
                else:
                    # Stream without timestamp is considered stale
                    stale_streams.append(stream_id)
            
            for stream_id in stale_streams:
                RequestProcessor.active_streams.discard(stream_id)
                RequestProcessor.stream_timestamps.pop(stream_id, None)
                logger.warning(f"Cleaned up stale stream {stream_id}")
            
            if stale_streams:
                logger.warning(f"Cleaned up {len(stale_streams)} stale streams")
            elif RequestProcessor.active_streams:
                logger.info(f"No stale streams found, {len(RequestProcessor.active_streams)} active streams are healthy")
    
    @staticmethod
    async def _verify_model_switch(model_requested: str, request_id: str, max_retries: int = MODEL_SWITCH_MAX_RETRIES) -> bool:
        """Verify model switch with retry logic."""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(MODEL_SWITCH_VERIFICATION_DELAY)
                
                updated_service_info = eternal_zoo_manager.get_service_info()
                updated_models = updated_service_info.get("models", {})
                
                if model_requested in updated_models and updated_models[model_requested].get("active", False):
                    logger.debug(f"[{request_id}] Model switch verification successful (attempt {attempt + 1})")
                    return True
                    
            except Exception as e:
                logger.warning(f"[{request_id}] Verification attempt {attempt + 1} failed: {e}")
        
        logger.error(f"[{request_id}] Model switch verification failed after {max_retries} attempts")
        return False
    
    @staticmethod
    async def _add_to_queue_with_backpressure(item, timeout: float = QUEUE_BACKPRESSURE_TIMEOUT):
        """Add item to queue with timeout and backpressure handling."""
        try:
            await asyncio.wait_for(
                RequestProcessor.queue.put(item),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            current_size = RequestProcessor.queue.qsize()
            logger.error(f"Queue full (size: {current_size}), request timed out after {timeout}s")
            raise HTTPException(
                status_code=503, 
                detail=f"Service overloaded. Queue size: {current_size}/{MAX_QUEUE_SIZE}"
            )
    
    @staticmethod
    async def _ensure_model_active_in_queue(model_requested: str, request_id: str) -> bool:
        """
        Ensure the requested model is active within the queue processing context.
        This method is called within the processing lock to ensure atomic model switching.
        
        Args:
            model_requested (str): The model hash requested by the client
            request_id (str): The request ID for logging
            
        Returns:
            bool: True if the model is active or was successfully switched to
        """
        try:
            # Get current service info
            service_info = get_service_info()
            models = service_info.get("models", {})
            
            # Check if the requested model exists
            if model_requested not in models:
                logger.error(f"[{request_id}] Requested model {model_requested} not found in available models")
                return False
            
            model_info = models[model_requested]
            
            # Check if model is already active
            if model_info.get("active", False):
                logger.debug(f"[{request_id}] Model {model_requested} is already active")
                return True
                
            # Model exists but not active, need to switch
            logger.info(f"[{request_id}] Model switch required to {model_requested}")
            
            # Wait for any active streams to complete before switching
            if await RequestProcessor.has_active_streams():
                stream_count = len(RequestProcessor.active_streams)
                logger.info(f"[{request_id}] Waiting for {stream_count} active streams to complete before model switch")
                if not await RequestProcessor.wait_for_streams_to_complete(timeout=MODEL_SWITCH_STREAM_TIMEOUT, force_terminate=True):
                    logger.error(f"[{request_id}] Failed to wait for streams to complete")
                    return False
                logger.info(f"[{request_id}] All streams completed, proceeding with model switch")
            
            # Perform the model switch
            logger.info(f"[{request_id}] Switching to model {model_requested}")
            switch_start_time = time.time()
            
            # Get current active model for logging
            active_model = eternal_zoo_manager.get_active_model()
            
            # Perform the model switch
            success = await eternal_zoo_manager.switch_model(model_requested)
            
            switch_duration = time.time() - switch_start_time
            
            if success:
                logger.info(f"[{request_id}] Successfully switched from {active_model} to {model_requested} "
                           f"(switch time: {switch_duration:.2f}s)")
                
                # Update app state with new service info and verify the switch
                try:
                    # Use the new verification method with retry logic
                    if await RequestProcessor._verify_model_switch(model_requested, request_id):
                        # Update app state with new service info after successful verification
                        updated_service_info = eternal_zoo_manager.get_service_info()
                        app.state.service_info = updated_service_info
                        return True
                    else:
                        logger.error(f"[{request_id}] Model switch verification failed - model not active after switch")
                        return False
                        
                except Exception as e:
                    logger.error(f"[{request_id}] Error verifying model switch: {str(e)}")
                    return False
            else:
                logger.error(f"[{request_id}] Failed to switch to model {model_requested} "
                           f"(attempted for {switch_duration:.2f}s)")
                return False
                
        except Exception as e:
            logger.error(f"[{request_id}] Error ensuring model active: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def process_request(endpoint: str, request_data: Dict[str, Any]):
        """Process a request by adding it to the queue and waiting for the result."""
        request_id = generate_request_id()
        queue_size = RequestProcessor.queue.qsize()
        
        # Validate that model field is present and get service info
        service_info, validated_model = validate_model_field(request_data)
        request_data["model"] = validated_model
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        start_wait_time = time.time()
        future = asyncio.Future()
        # Pass service_info to avoid redundant lookups
        queue_item = (endpoint, request_data, future, request_id, start_wait_time, service_info)
        await RequestProcessor._add_to_queue_with_backpressure(queue_item)
        
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: Dict[str, Any]):
        """Process a request directly without queueing for administrative endpoints."""
        request_id = generate_request_id()
        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}")
        
        # Validate that model field is present and get service info
        service_info, validated_model = validate_model_field(request_data)
        request_data["model"] = validated_model
        
        start_time = time.time()
        if endpoint in RequestProcessor.MODEL_ENDPOINTS:
            model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
            request_obj = model_cls(**request_data)
            
            # Ensure model is active before processing (for direct requests)
            if hasattr(request_obj, 'model') and request_obj.model:
                logger.debug(f"[{request_id}] Ensuring model {request_obj.model} is active for direct request")
                
                # Use the same centralized model switching logic as the queue
                if not await RequestProcessor._ensure_model_active_in_queue(request_obj.model, request_id):
                    error_msg = f"Model {request_obj.model} is not available or failed to switch"
                    logger.error(f"[{request_id}] {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Refresh service info after potential model switch
                service_info = get_service_info()
                logger.debug(f"[{request_id}] Model {request_obj.model} confirmed active for direct request")
            
            # Process the request with the updated service info
            result = await handler(request_obj, service_info)
            
            process_time = time.time() - start_time
            logger.info(f"[{request_id}] Direct request completed for endpoint {endpoint} (time: {process_time:.2f}s)")
            
            return result
        else:
            logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
            raise HTTPException(status_code=404, detail="Endpoint not found")
    
    @staticmethod
    async def worker():
        """Enhanced worker function with better error recovery."""
        logger.info("Request processor worker started")
        processed_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                endpoint, request_data, future, request_id, start_wait_time, service_info = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint} "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                # Process the request within the lock to ensure sequential execution
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()
                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
                        try:
                            request_obj = model_cls(**request_data)
                            
                            # Check if this is a streaming request
                            is_streaming = hasattr(request_obj, 'stream') and request_obj.stream
                            if is_streaming:
                                logger.debug(f"[{request_id}] Processing streaming request for model {request_obj.model}")
                            
                            # Ensure model is active before processing (within the lock)
                            if hasattr(request_obj, 'model') and request_obj.model:
                                logger.debug(f"[{request_id}] Ensuring model {request_obj.model} is active")
                                
                                # Check current active streams before model switching
                                active_stream_count = len(RequestProcessor.active_streams)
                                if active_stream_count > 0:
                                    logger.info(f"[{request_id}] Found {active_stream_count} active streams before model check")
                                
                                if not await RequestProcessor._ensure_model_active_in_queue(request_obj.model, request_id):
                                    error_msg = f"Model {request_obj.model} is not available or failed to switch"
                                    logger.error(f"[{request_id}] {error_msg}")
                                    future.set_exception(HTTPException(status_code=400, detail=error_msg))
                                    continue
                                
                                # Refresh service info after potential model switch
                                service_info = get_service_info()
                                logger.debug(f"[{request_id}] Model {request_obj.model} confirmed active, proceeding with request")
                            
                            # Process the request with the updated service info
                            result = await handler(request_obj, service_info)
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint} "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"[{request_id}] Handler error for {endpoint}: {str(e)}")
                            future.set_exception(e)
                    else:
                        logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
                        future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
                
                # Reset consecutive errors on successful processing
                consecutive_errors = 0
                
                # Log periodic status about queue health
                if processed_count % 10 == 0:
                    active_stream_count = len(RequestProcessor.active_streams)
                    logger.info(f"Queue status: current size={queue_size}, processed={processed_count}, active streams={active_stream_count}")
                
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Worker error (consecutive: {consecutive_errors}/{max_consecutive_errors}): {str(e)}")
                
                # If we have too many consecutive errors, pause before continuing
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive worker errors, pausing for recovery")
                    await asyncio.sleep(5)  # Brief pause before continuing
                    consecutive_errors = 0  # Reset counter after recovery pause
                    
                    # Clean up any potential issues
                    try:
                        await RequestProcessor.cleanup_stale_streams()
                    except Exception as cleanup_error:
                        logger.error(f"Error during worker recovery cleanup: {cleanup_error}")
                
                # Mark the task as done to prevent queue from getting stuck
                try:
                    RequestProcessor.queue.task_done()
                except ValueError:
                    # task_done() called more times than items in queue
                    pass

async def stream_cleanup_task():
    """Periodic cleanup of stale streams."""
    logger.info("Stream cleanup task started")
    
    while True:
        try:
            await asyncio.sleep(STREAM_CLEANUP_INTERVAL)
            
            # Clean up stale streams
            await RequestProcessor.cleanup_stale_streams()
            
        except asyncio.CancelledError:
            logger.info("Stream cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in stream cleanup task: {str(e)}", exc_info=True)
            # Wait a bit longer before retrying on critical errors
            await asyncio.sleep(STREAM_CLEANUP_ERROR_SLEEP)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware that adds a header with the processing time for the request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Startup event handler: initialize the HTTP client and start the worker task."""
    # Create an asynchronous HTTP client with optimized connection pooling
    limits = httpx.Limits(
        max_connections=POOL_CONNECTIONS,
        max_keepalive_connections=POOL_CONNECTIONS,
        keepalive_expiry=POOL_KEEPALIVE
    )
    app.state.client = httpx.AsyncClient(
        limits=limits,
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(
            retries=MAX_RETRIES,
            verify = True # SSL verification for local connections
        )
    )
    
    # Initialize the last request time
    app.state.last_request_time = time.time()
    
    # Start background tasks
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    # TEMPORARILY DISABLED: Dynamic unload functionality
    # app.state.unload_checker_task = asyncio.create_task(unload_checker())
    app.state.stream_cleanup_task = asyncio.create_task(stream_cleanup_task())
    
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Optimized shutdown event with proper resource cleanup and error handling.
    """
    logger.info("Starting application shutdown...")
    shutdown_start_time = time.time()
    
    # Phase 1: Cancel background tasks gracefully
    tasks_to_cancel = []
    task_names = []
    
    for task_attr in ["worker_task", "stream_cleanup_task"]:  # TEMPORARILY DISABLED: "unload_checker_task"
        if hasattr(app.state, task_attr):
            task = getattr(app.state, task_attr)
            if not task.done():
                task_names.append(task_attr)
                tasks_to_cancel.append(task)
                task.cancel()
    
    if tasks_to_cancel:
        logger.info(f"Cancelling background tasks: {', '.join(task_names)}")
        try:
            # Wait for tasks to complete cancellation with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=SHUTDOWN_TASK_TIMEOUT
            )
            logger.info("Background tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Background task cancellation timed out, proceeding with shutdown")
        except Exception as e:
            logger.error(f"Error during background task cancellation: {str(e)}")
    
    # Phase 2: Clean up EternalZoo server
    try:
        service_info = get_service_info()
        if "pid" in service_info:
            pid = service_info.get("pid")
            logger.info(f"Terminating EternalZoo server (PID: {pid}) during shutdown...")
            
            # Use the optimized kill method with timeout
            kill_success = await asyncio.wait_for(
                eternal_zoo_manager.kill_ai_server(),
                timeout=SHUTDOWN_SERVER_TIMEOUT
            )
            
            if kill_success:
                logger.info("EternalZoo server terminated successfully during shutdown")
            else:
                logger.warning("EternalZoo server termination failed during shutdown")
        else:
            logger.debug("No EternalZoo server PID found, skipping termination")
            
    except HTTPException:
        logger.debug("Service info not available during shutdown (expected)")
    except asyncio.TimeoutError:
        logger.error("EternalZoo server termination timed out during shutdown")
    except Exception as e:
        logger.error(f"Error terminating EternalZoo server during shutdown: {str(e)}")
    
    # Phase 3: Close HTTP client connections
    if hasattr(app.state, "client"):
        try:
            logger.info("Closing HTTP client connections...")
            await asyncio.wait_for(app.state.client.aclose(), timeout=SHUTDOWN_CLIENT_TIMEOUT)
            logger.info("HTTP client closed successfully")
        except asyncio.TimeoutError:
            logger.warning("HTTP client close timed out")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {str(e)}")
    
    # Phase 4: Clean up any remaining request queue and streams
    if hasattr(RequestProcessor, 'queue'):
        try:
            # Clean up any remaining active streams
            await RequestProcessor.cleanup_stale_streams()
            
            queue_size = RequestProcessor.queue.qsize()
            if queue_size > 0:
                logger.warning(f"Request queue still has {queue_size} pending requests during shutdown")
                # Cancel any pending futures in the queue
                pending_requests = []
                while not RequestProcessor.queue.empty():
                    try:
                        _, _, future, request_id, _, _ = RequestProcessor.queue.get_nowait()
                        if not future.done():
                            future.cancel()
                            pending_requests.append(request_id)
                    except asyncio.QueueEmpty:
                        break
                
                if pending_requests:
                    logger.info(f"Cancelled {len(pending_requests)} pending requests")
        except Exception as e:
            logger.error(f"Error cleaning up request queue: {str(e)}")
    
    shutdown_duration = time.time() - shutdown_start_time
    logger.info(f"Application shutdown complete (duration: {shutdown_duration:.2f}s)")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint that bypasses the request queue for immediate response."""
    return {"status": "ok"}
    
@app.post("/update/lora")
async def update_lora(request: LoraConfigRequest):
    """Update the LoRA for a given model hash."""
    request_dict = convert_request_to_dict(request)
    if eternal_zoo_manager.update_lora(request_dict):
        return {"status": "ok", "message": "LoRA updated successfully"}
    else:
        return {"status": "error", "message": "Failed to update LoRA"}

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)

@app.post("/images/generations")
@app.post("/v1/images/generations")
async def image_generations(request: ImageGenerationRequest):
    """Endpoint for image generation requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/images/generations", request_dict)

@app.get("/models", response_model=ModelList)
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Provides a list of available models, compatible with OpenAI's /v1/models endpoint.
    Returns all models in multi-model service including main and on-demand models.
    """
    try:
        service_info = get_service_info()
    except HTTPException as e:
        if e.status_code == 503:
            logger.info("/v1/models: Service information not available. No model loaded or /update not called.")
            return ModelList(data=[])
        logger.error(f"/v1/models: Unexpected HTTPException while fetching service_info: {e.detail}")
        raise

    model_cards = []
    models = service_info.get("models", {})

    if models:
        logger.info(f"/v1/models: Multi-model service detected with {len(models)} models")
        for model_hash, model_info in models.items():
            metadata = model_info.get("metadata", {})
            folder_name = metadata.get("folder_name", "")
            active = model_info.get("active", False)
            is_on_demand = model_info.get("on_demand", False)
            task = metadata.get("task", "chat")
            lora_config = model_info.get("lora_config", None)
            context_length = model_info.get("context_length", None)
            base_model_path = model_info.get("base_model_path", None)
            local_model_path = model_info.get("local_model_path", None)
            local_projector_path = model_info.get("local_projector_path", None)
            parent = model_info.get("parent", None)
            permission = model_info.get("permission", None)
            created = metadata.get("created", int(time.time()))
            owned_by = metadata.get("owned_by", "user")
            multimodal = metadata.get("multimodal", False)

            model_id = folder_name if folder_name else model_hash
            raw_ram_value = metadata.get("ram")
            parsed_ram_value = None

            if isinstance(raw_ram_value, (int, float)):
                parsed_ram_value = float(raw_ram_value)
            elif isinstance(raw_ram_value, str):
                try:
                    parsed_ram_value = float(raw_ram_value.lower().replace("gb", "").strip())
                except ValueError:
                    logger.warning(f"/v1/models: Could not parse RAM value '{raw_ram_value}' to float for model {model_id}")

            model_card = ModelCard(
                id=model_hash,
                object="model",
                created=created,
                owned_by=owned_by,
                active=active,
                root=model_id,
                parent=parent,
                permission=permission if permission is not None else [ModelPermission()],
                ram=parsed_ram_value,
                folder_name=folder_name,
                lora_config=lora_config,
                on_demand=is_on_demand,
                task=task,
                multimodal=multimodal,
                context_length=context_length,
                base_model_path=base_model_path,
                local_model_path=local_model_path,
                local_projector_path=local_projector_path
            )
            model_cards.append(model_card)
            status = "🟢 Active" if active else ("🔴 On-demand" if is_on_demand else "⚪ Unknown")
            logger.debug(f"/v1/models: Added model {model_id} ({model_hash[:16]}...) - {status}")
    else:
        model_hash = service_info.get("hash")
        folder_name_from_info = service_info.get("folder_name")
        task = service_info.get("task", "chat")
        lora_config = service_info.get("lora_config", None)
        context_length = service_info.get("context_length", None)
        base_model_path = service_info.get("base_model_path", None)
        local_model_path = service_info.get("local_model_path", None)
        local_projector_path = service_info.get("local_projector_path", None)
        parent = service_info.get("parent", None)
        permission = service_info.get("permission", None)
        created = service_info.get("created", int(time.time()))
        owned_by = service_info.get("owned_by", "user")
        multimodal = service_info.get("multimodal", False)
        if not model_hash:
            logger.warning("/v1/models: No model hash found in service_info, though service_info itself was present. Returning empty list.")
            return ModelList(data=[])
        model_id = folder_name_from_info if folder_name_from_info else model_hash
        raw_ram_value = service_info.get("ram")
        parsed_ram_value = None
        if isinstance(raw_ram_value, (int, float)):
            parsed_ram_value = float(raw_ram_value)
        elif isinstance(raw_ram_value, str):
            try:
                parsed_ram_value = float(raw_ram_value.lower().replace("gb", "").strip())
            except ValueError:
                logger.warning(f"/v1/models: Could not parse RAM value '{raw_ram_value}' to float.")
        model_card = ModelCard(
            id=model_hash,
            object="model",
            created=created,
            owned_by=owned_by,
            root=model_id,
            parent=parent,
            permission=permission if permission is not None else [ModelPermission()],
            ram=parsed_ram_value,
            folder_name=folder_name_from_info,
            lora_config=lora_config,
            on_demand=None,
            task=task,
            context_length=context_length,
            base_model_path=base_model_path,
            local_model_path=local_model_path,
            local_projector_path=local_projector_path,
            active = True,
            multimodal=multimodal
        )
        model_cards.append(model_card)
        logger.info(f"/v1/models: Single-model service - returning model {model_id}")
    logger.info(f"/v1/models: Returning {len(model_cards)} models")
    return ModelList(data=model_cards)