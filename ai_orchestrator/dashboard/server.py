"""FastAPI server for AI Orchestrator Dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    AgentActionRequest,
    AgentActionResponse,
    AgentConfig,
    AgentMetrics,
    AgentRole,
    AgentStatus,
    DashboardConfig,
    DashboardMetrics,
    PhaseStatus,
    PhaseTimeouts,
    RunWorkflowRequest,
    RunWorkflowResponse,
    UpdateConfigRequest,
    UpdateConfigResponse,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# In-memory state (would be persisted in production)
_config: DashboardConfig | None = None
_workflows: dict[str, WorkflowStatus] = {}
_metrics: DashboardMetrics = DashboardMetrics()
_websocket_connections: list[WebSocket] = []


def get_default_agents() -> dict[str, AgentConfig]:
    """Get default agent configurations."""
    return {
        "claude": AgentConfig(
            name="claude",
            display_name="Claude (Anthropic)",
            enabled=True,
            roles=[AgentRole.PLANNER, AgentRole.REVIEWER, AgentRole.IMPLEMENTER],
            priority=1,
            model="claude-sonnet-4-20250514",
            status=AgentStatus.AVAILABLE,
        ),
        "codex": AgentConfig(
            name="codex",
            display_name="Codex (OpenAI)",
            enabled=False,
            roles=[AgentRole.PLANNER, AgentRole.REVIEWER, AgentRole.IMPLEMENTER],
            priority=2,
            model="codex-latest",
            status=AgentStatus.DISABLED,
        ),
        "gemini": AgentConfig(
            name="gemini",
            display_name="Gemini (Google)",
            enabled=False,
            roles=[AgentRole.PLANNER, AgentRole.REVIEWER, AgentRole.RESEARCHER],
            priority=2,
            model="gemini-2.5-pro",
            status=AgentStatus.DISABLED,
        ),
        "kilocode": AgentConfig(
            name="kilocode",
            display_name="Kilocode (OpenRouter)",
            enabled=False,
            roles=[AgentRole.REVIEWER],
            priority=3,
            model="mistralai/devstral-2512:free",
            status=AgentStatus.DISABLED,
        ),
    }


def get_default_config() -> DashboardConfig:
    """Get default dashboard configuration."""
    return DashboardConfig(
        agents=get_default_agents(),
    )


def load_config() -> DashboardConfig:
    """Load configuration from disk or return defaults."""
    global _config
    if _config is not None:
        return _config

    config_path = Path.home() / ".ai_orchestrator" / "dashboard_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            _config = DashboardConfig.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            _config = get_default_config()
    else:
        _config = get_default_config()

    return _config


def save_config(config: DashboardConfig) -> None:
    """Save configuration to disk."""
    global _config
    _config = config

    config_dir = Path.home() / ".ai_orchestrator"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "dashboard_config.json"

    with open(config_path, "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=str)


async def broadcast_update(event_type: str, data: dict[str, Any]) -> None:
    """Broadcast an update to all connected WebSocket clients."""
    if not _websocket_connections:
        return

    message = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now(UTC).isoformat()}, default=str)

    disconnected = []
    for ws in _websocket_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        _websocket_connections.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    load_config()
    logger.info("Dashboard server started")
    yield
    # Shutdown
    logger.info("Dashboard server shutting down")


# Create FastAPI app
app = FastAPI(
    title="AI Orchestrator Dashboard",
    description="Control panel for the Multi-AI Code Orchestration System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# === HTML Routes ===

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard page."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Dashboard template not found</h1>", status_code=500)


# === Configuration API ===

@app.get("/api/config", response_model=DashboardConfig)
async def get_config():
    """Get current configuration."""
    return load_config()


@app.put("/api/config", response_model=UpdateConfigResponse)
async def update_config(request: UpdateConfigRequest):
    """Update configuration."""
    try:
        save_config(request.config)
        await broadcast_update("config_updated", request.config.model_dump(mode="json"))
        return UpdateConfigResponse(
            success=True,
            message="Configuration updated successfully",
            config=request.config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/reset", response_model=UpdateConfigResponse)
async def reset_config():
    """Reset configuration to defaults."""
    config = get_default_config()
    save_config(config)
    await broadcast_update("config_reset", config.model_dump(mode="json"))
    return UpdateConfigResponse(
        success=True,
        message="Configuration reset to defaults",
        config=config,
    )


# === Agent Management API ===

@app.get("/api/agents", response_model=dict[str, AgentConfig])
async def get_agents():
    """Get all agent configurations."""
    config = load_config()
    return config.agents


@app.get("/api/agents/{agent_name}", response_model=AgentConfig)
async def get_agent(agent_name: str):
    """Get a specific agent configuration."""
    config = load_config()
    if agent_name not in config.agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return config.agents[agent_name]


@app.put("/api/agents/{agent_name}", response_model=AgentConfig)
async def update_agent(agent_name: str, agent_config: AgentConfig):
    """Update a specific agent configuration."""
    config = load_config()
    if agent_name not in config.agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    config.agents[agent_name] = agent_config
    save_config(config)
    await broadcast_update("agent_updated", {"agent": agent_name, "config": agent_config.model_dump(mode="json")})
    return agent_config


@app.post("/api/agents/{agent_name}/action", response_model=AgentActionResponse)
async def agent_action(agent_name: str, request: AgentActionRequest):
    """Perform an action on an agent (enable, disable, test, reset)."""
    config = load_config()
    if agent_name not in config.agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = config.agents[agent_name]

    if request.action == "enable":
        agent.enabled = True
        agent.status = AgentStatus.AVAILABLE
        message = f"Agent '{agent_name}' enabled"
    elif request.action == "disable":
        agent.enabled = False
        agent.status = AgentStatus.DISABLED
        message = f"Agent '{agent_name}' disabled"
    elif request.action == "reset_circuit_breaker":
        agent.status = AgentStatus.AVAILABLE
        message = f"Circuit breaker reset for agent '{agent_name}'"
    elif request.action == "test":
        # Simulate a test - in production, would actually test the CLI
        message = f"Agent '{agent_name}' test successful (simulated)"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    config.agents[agent_name] = agent
    save_config(config)
    await broadcast_update("agent_action", {"agent": agent_name, "action": request.action})

    return AgentActionResponse(
        success=True,
        message=message,
        agent_status=agent,
    )


# === Timeout Configuration API ===

@app.get("/api/timeouts")
async def get_timeouts():
    """Get timeout configuration."""
    config = load_config()
    return config.timeouts


@app.put("/api/timeouts")
async def update_timeouts(timeouts: dict[str, Any]):
    """Update timeout configuration."""
    config = load_config()

    # Update per-agent timeouts
    for agent_name in ["claude", "codex", "gemini", "kilocode"]:
        if agent_name in timeouts:
            agent_timeouts = timeouts[agent_name]
            setattr(config.timeouts, agent_name, PhaseTimeouts(**agent_timeouts))

    # Update global timeouts
    if "max_total_workflow" in timeouts:
        config.timeouts.max_total_workflow = timeouts["max_total_workflow"]
    if "max_single_operation" in timeouts:
        config.timeouts.max_single_operation = timeouts["max_single_operation"]
    if "timeout_action" in timeouts:
        config.timeouts.timeout_action = timeouts["timeout_action"]
    if "retry_on_timeout" in timeouts:
        config.timeouts.retry_on_timeout = timeouts["retry_on_timeout"]

    save_config(config)
    await broadcast_update("timeouts_updated", config.timeouts.model_dump(mode="json"))
    return config.timeouts


# === Research Configuration API ===

@app.get("/api/research")
async def get_research_config():
    """Get research configuration."""
    config = load_config()
    return {
        "research": config.research.model_dump(mode="json", exclude={"api_key"}),
        "web_search": config.web_search.model_dump(mode="json", exclude={"api_key"}),
    }


@app.put("/api/research")
async def update_research_config(research_config: dict[str, Any]):
    """Update research configuration."""
    config = load_config()

    if "research" in research_config:
        for key, value in research_config["research"].items():
            if hasattr(config.research, key):
                setattr(config.research, key, value)

    if "web_search" in research_config:
        for key, value in research_config["web_search"].items():
            if hasattr(config.web_search, key):
                setattr(config.web_search, key, value)

    save_config(config)
    await broadcast_update("research_updated", research_config)
    return {
        "research": config.research.model_dump(mode="json", exclude={"api_key"}),
        "web_search": config.web_search.model_dump(mode="json", exclude={"api_key"}),
    }


# === Prompt Enhancement API ===

@app.get("/api/prompt-enhancement")
async def get_prompt_enhancement():
    """Get prompt enhancement configuration."""
    config = load_config()
    return config.prompt_enhancement


@app.put("/api/prompt-enhancement")
async def update_prompt_enhancement(enhancement: dict[str, Any]):
    """Update prompt enhancement configuration."""
    config = load_config()

    for key, value in enhancement.items():
        if hasattr(config.prompt_enhancement, key):
            setattr(config.prompt_enhancement, key, value)

    save_config(config)
    await broadcast_update("prompt_enhancement_updated", config.prompt_enhancement.model_dump(mode="json"))
    return config.prompt_enhancement


# === Workflow Configuration API ===

@app.get("/api/workflow")
async def get_workflow_config():
    """Get workflow configuration."""
    config = load_config()
    return {
        "iteration": config.iteration.model_dump(mode="json"),
        "human_loop": config.human_loop.model_dump(mode="json"),
        "resilience": config.resilience.model_dump(mode="json"),
        "incremental_review": config.incremental_review.model_dump(mode="json"),
        "post_checks": config.post_checks.model_dump(mode="json"),
        "agent_assignment": config.agent_assignment.model_dump(mode="json"),
    }


@app.put("/api/workflow")
async def update_workflow_config(workflow: dict[str, Any]):
    """Update workflow configuration."""
    config = load_config()

    if "iteration" in workflow:
        for key, value in workflow["iteration"].items():
            if hasattr(config.iteration, key):
                setattr(config.iteration, key, value)

    if "human_loop" in workflow:
        for key, value in workflow["human_loop"].items():
            if hasattr(config.human_loop, key):
                setattr(config.human_loop, key, value)

    if "resilience" in workflow:
        for key, value in workflow["resilience"].items():
            if hasattr(config.resilience, key):
                setattr(config.resilience, key, value)

    if "incremental_review" in workflow:
        for key, value in workflow["incremental_review"].items():
            if hasattr(config.incremental_review, key):
                setattr(config.incremental_review, key, value)

    if "post_checks" in workflow:
        for key, value in workflow["post_checks"].items():
            if hasattr(config.post_checks, key):
                setattr(config.post_checks, key, value)

    if "agent_assignment" in workflow:
        for key, value in workflow["agent_assignment"].items():
            if hasattr(config.agent_assignment, key):
                setattr(config.agent_assignment, key, value)

    save_config(config)
    await broadcast_update("workflow_updated", workflow)
    return await get_workflow_config()


# === Workflow Execution API ===

@app.post("/api/workflow/run", response_model=RunWorkflowResponse)
async def run_workflow(request: RunWorkflowRequest):
    """Start a new workflow execution."""
    workflow_id = str(uuid4())

    # Create initial workflow status
    workflow = WorkflowStatus(
        workflow_id=workflow_id,
        prompt=request.prompt,
        project_path=request.project_path or ".",
        current_phase="init",
        started_at=datetime.now(UTC),
        phases=[
            PhaseStatus(phase="init", status="in_progress", started_at=datetime.now(UTC)),
            PhaseStatus(phase="planning", status="pending"),
            PhaseStatus(phase="reviewing", status="pending"),
            PhaseStatus(phase="implementing", status="pending"),
            PhaseStatus(phase="post_checks", status="pending"),
        ],
    )

    _workflows[workflow_id] = workflow
    await broadcast_update("workflow_started", workflow.model_dump(mode="json"))

    return RunWorkflowResponse(
        workflow_id=workflow_id,
        status="started",
        message=f"Workflow {workflow_id} started successfully",
    )


@app.get("/api/workflow/status/{workflow_id}", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow."""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return _workflows[workflow_id]


@app.get("/api/workflow/active", response_model=list[WorkflowStatus])
async def get_active_workflows():
    """Get all active workflows."""
    active = [w for w in _workflows.values() if w.current_phase not in ("completed", "failed")]
    return active


@app.post("/api/workflow/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    workflow = _workflows[workflow_id]
    workflow.current_phase = "failed"
    workflow.errors.append("Workflow cancelled by user")
    await broadcast_update("workflow_cancelled", {"workflow_id": workflow_id})

    return {"success": True, "message": f"Workflow {workflow_id} cancelled"}


# === Metrics API ===

@app.get("/api/metrics", response_model=DashboardMetrics)
async def get_metrics():
    """Get dashboard metrics."""
    # Calculate metrics from workflows
    metrics = DashboardMetrics(
        total_workflows=len(_workflows),
        completed_workflows=sum(1 for w in _workflows.values() if w.current_phase == "completed"),
        failed_workflows=sum(1 for w in _workflows.values() if w.current_phase == "failed"),
        active_workflows=sum(1 for w in _workflows.values() if w.current_phase not in ("completed", "failed")),
    )

    # Add agent metrics
    config = load_config()
    for agent_name, agent in config.agents.items():
        metrics.agents[agent_name] = AgentMetrics(
            name=agent_name,
            circuit_breaker_state="closed" if agent.status == AgentStatus.AVAILABLE else "open",
        )

    return metrics


@app.get("/api/metrics/agents", response_model=dict[str, AgentMetrics])
async def get_agent_metrics():
    """Get metrics for all agents."""
    config = load_config()
    metrics = {}
    for agent_name, agent in config.agents.items():
        metrics[agent_name] = AgentMetrics(
            name=agent_name,
            circuit_breaker_state="closed" if agent.status == AgentStatus.AVAILABLE else "open",
        )
    return metrics


# === WebSocket for Real-time Updates ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    _websocket_connections.append(websocket)

    try:
        # Send initial state
        config = load_config()
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "data": {
                "config": config.model_dump(mode="json"),
                "workflows": {k: v.model_dump(mode="json") for k, v in _workflows.items()},
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }, default=str))

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Handle ping/pong for keepalive
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": datetime.now(UTC).isoformat()}))

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _websocket_connections:
            _websocket_connections.remove(websocket)


# === Export/Import Configuration ===

@app.get("/api/config/export")
async def export_config():
    """Export current configuration as JSON."""
    config = load_config()
    return config.model_dump(mode="json")


@app.post("/api/config/import")
async def import_config(config_data: dict[str, Any]):
    """Import configuration from JSON."""
    try:
        config = DashboardConfig.model_validate(config_data)
        save_config(config)
        await broadcast_update("config_imported", config.model_dump(mode="json"))
        return {"success": True, "message": "Configuration imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
