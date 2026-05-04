import json
import traceback
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from schemas import RunRequest
from graph import initial_state, step_plan, step_search, step_synthesize, get_llm
from memory import RollingBuffer
from settings import settings
from constants import (
    ALLOWED_ORIGINS,
    ALLOWED_ORIGIN_REGEX,
    SSE_HEADERS,
    PROGRESS_PLANNING,
    PROGRESS_PLAN_COMPLETE,
    PROGRESS_SEARCHING,
    PROGRESS_SEARCH_COMPLETE,
    PROGRESS_SYNTHESIZING,
    PROGRESS_SYNTHESIS_COMPLETE,
    PROGRESS_DONE,
)
from exceptions import APIKeyError
from logger import logger

app = FastAPI(
    title="Deep Research Agent",
    description="LangGraph-powered research agent with streaming support",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = RollingBuffer(max_len=settings.MAX_MESSAGES)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/run")
async def run(req: RunRequest, request: Request):
    logger.info("API", "🚀 /run endpoint called")
    logger.info("API", f"Origin: {request.headers.get('origin')}")
    logger.info("API", f"Query: {req.query}")
    logger.info("API", f"Config: {req.config.model_dump()}")
    logger.info("API", f"Memory messages: {len(req.messages)}")

    memory.extend([m.model_dump() for m in req.messages])

    state = initial_state(
        query=req.query,
        config=req.config.model_dump(),
        messages=memory.as_messages()
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            try:
                actual_provider = settings.get_available_provider(state["config"]["provider"])
                if actual_provider != state["config"]["provider"]:
                    yield _sse_event("log", {
                        "msg": f"⚠️  Gemini API key not found, using OpenAI instead"
                    })
                    state["config"]["provider"] = actual_provider
            except (ValueError, APIKeyError) as e:
                yield _sse_event("error", {"message": str(e)})
                return

            try:
                settings.validate_search_requirements()
            except (ValueError, APIKeyError) as e:
                yield _sse_event("error", {"message": str(e)})
                return

            logger.info("API", "→ Sending planning status...")
            yield _sse_event("status", {"phase": "planning"})
            yield _sse_event("log", {"msg": "Starting planning phase..."})
            yield _sse_event("progress", {"percent": PROGRESS_PLANNING})

            logger.info("API", "→ Planning...")
            loop = asyncio.get_event_loop()
            state1 = await loop.run_in_executor(None, step_plan, state)
            plan = state1["plan"]
            logger.info("API", f"Plan generated: {len(plan)} chars")

            yield _sse_event("plan", {"text": plan})
            yield _sse_event("log", {"msg": "Planning complete"})
            yield _sse_event("progress", {"percent": PROGRESS_PLAN_COMPLETE})

            logger.info("API", "→ Sending searching status...")
            yield _sse_event("status", {"phase": "searching"})
            yield _sse_event("log", {"msg": "Starting search phase..."})
            yield _sse_event("progress", {"percent": PROGRESS_SEARCHING})

            logger.info("API", "→ Searching...")
            state2 = await loop.run_in_executor(None, step_search, state1)
            sources = state2["sources"]
            logger.info("API", f"Found {len(sources)} unique sources")

            yield _sse_event("sources", {"count": len(sources), "top": sources[:5]})
            yield _sse_event("log", {"msg": f"Found {len(sources)} unique sources"})
            yield _sse_event("progress", {"percent": PROGRESS_SEARCH_COMPLETE})

            logger.info("API", "→ Sending synthesizing status...")
            yield _sse_event("status", {"phase": "synthesizing"})
            yield _sse_event("log", {"msg": "Starting synthesis..."})
            yield _sse_event("progress", {"percent": PROGRESS_SYNTHESIZING})

            logger.info("API", "→ Synthesizing report...")
            state3 = await loop.run_in_executor(None, step_synthesize, state2)
            report = state3["report"]
            logger.info("API", f"Report generated: {len(report['content'])} chars")

            yield _sse_event("log", {"msg": "Synthesis complete"})
            yield _sse_event("progress", {"percent": PROGRESS_SYNTHESIS_COMPLETE})

            logger.info("API", "→ Sending final report...")
            yield _sse_event("done", {"report": report})
            yield _sse_event("progress", {"percent": PROGRESS_DONE})
            logger.success("API", "🏁 Done.")

        except (ValueError, APIKeyError) as e:
            logger.error("API", f"Validation error: {e}")
            yield _sse_event("error", {"message": str(e)})
        except Exception as e:
            logger.error("API", f"Error in event_generator: {e}")
            traceback.print_exc()
            yield _sse_event("error", {"message": f"An error occurred: {str(e)}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )



@app.post("/run_sync")
async def run_sync(req: RunRequest):
    try:
        actual_provider = settings.get_available_provider(req.config.provider)
        if actual_provider != req.config.provider:
            logger.warning(
                "API",
                f"Provider fallback: {req.config.provider} → {actual_provider}"
            )

        settings.validate_search_requirements()

        memory.extend([m.model_dump() for m in req.messages])
        state = initial_state(req.query, req.config.model_dump(), memory.as_messages())
        state["config"]["provider"] = actual_provider

        state1 = step_plan(state)
        state2 = step_search(state1)
        state3 = step_synthesize(state2)

        return JSONResponse(content={
            "report": state3["report"],
            "plan": state1["plan"],
            "sources": state2["sources"],
            "actual_provider": actual_provider
        })

    except (ValueError, APIKeyError) as e:
        logger.error("API", f"Validation error in run_sync: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )
    except Exception as e:
        logger.error("API", f"Error in run_sync: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )



@app.post("/chat")
async def chat(req: RunRequest):
    logger.info("API", "💬 /chat endpoint called")
    logger.info("API", f"Query: {req.query}")
    logger.info("API", f"Context messages: {len(req.messages)}")

    memory.extend([m.model_dump() for m in req.messages])

    try:
        actual_provider = settings.get_available_provider(req.config.provider)
        if actual_provider != req.config.provider:
            logger.warning(
                "API",
                f"Chat provider fallback: {req.config.provider} → {actual_provider}"
            )

        llm = get_llm(actual_provider, req.config.model)

        messages = []
        for msg in memory.as_messages():
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": req.query})

        response = llm.invoke(messages)

        return JSONResponse(content={
            "response": response.content,
            "mode": "chat",
            "actual_provider": actual_provider
        })

    except (ValueError, APIKeyError) as e:
        logger.error("API", f"Chat validation error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )
    except Exception as e:
        logger.error("API", f"Chat error: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "service": "Deep Research Agent",
        "version": "1.0.0",
        "features": {
            "openai": bool(settings.OPENAI_API_KEY),
            "gemini": settings.has_gemini,
            "tavily": settings.has_tavily,
            "serpapi": settings.has_serp_api,
            "dual_search": settings.can_use_dual_search
        }
    })



@app.on_event("startup")
async def startup_event():
    """Log application startup and configuration status."""
    logger.success("API", "🚀 Deep Research Agent starting up...")
    logger.info("API", f"OpenAI configured: {bool(settings.OPENAI_API_KEY)}")
    logger.info("API", f"Gemini configured: {settings.has_gemini}")
    logger.info("API", f"Tavily configured: {settings.has_tavily}")
    logger.info("API", f"SerpAPI configured: {settings.has_serp_api}")
    logger.info("API", f"Dual search available: {settings.can_use_dual_search}")
    logger.success("API", "✅ Application ready to accept requests")