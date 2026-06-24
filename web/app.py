#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi>=0.110", "uvicorn[standard]>=0.27", "python-multipart>=0.0.9",
#   "jinja2>=3.1", "slowapi>=0.1.9",
# ]
# ///
"""paraligilo web — minimal upload form + background alignment job + download.

Run:  ./web/app.py  (binds 0.0.0.0:8088)
Or:   uvicorn web.app:app --reload
"""
import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

ROOT = Path(__file__).resolve().parent.parent     # paraligilo/
RUN_PY = ROOT / "run.py"
JOBS_DIR = Path(os.environ.get("PARALIGILO_JOBS", "/tmp/paraligilo-jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_CONCURRENT = int(os.environ.get("PARALIGILO_MAX_CONCURRENT", "2"))
JOB_TTL_HOURS = int(os.environ.get("PARALIGILO_TTL_HOURS", "24"))
MAX_UPLOAD_MB = int(os.environ.get("PARALIGILO_MAX_MB", "30"))
SERVER_FALLBACK_KEY = os.environ.get("OPENAI_API_KEY")  # used for "free quota" if set

# Rate limits — override via env to tighten/loosen.
RL_ALIGN     = os.environ.get("PARALIGILO_RL_ALIGN",     "10/hour")
RL_ALIGN_AI  = os.environ.get("PARALIGILO_RL_ALIGN_AI",  "3/hour")   # when server-key used
RL_STATUS    = os.environ.get("PARALIGILO_RL_STATUS",    "120/minute")
RL_DOWNLOAD  = os.environ.get("PARALIGILO_RL_DOWNLOAD",  "60/hour")

def _client_ip(request: Request) -> str:
    # Honour X-Forwarded-For when behind nginx (set proxy_set_header in nginx).
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=_client_ip, default_limits=["200/hour"])

# In-memory job registry. Persists across requests within one uvicorn process.
JOBS: dict[str, dict] = {}
RUN_SEM = asyncio.Semaphore(MAX_CONCURRENT)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # janitor: prune old jobs
    async def janitor():
        while True:
            await asyncio.sleep(3600)
            cutoff = time.time() - JOB_TTL_HOURS * 3600
            for job_id, j in list(JOBS.items()):
                if j["created_at"] < cutoff:
                    shutil.rmtree(j["dir"], ignore_errors=True)
                    JOBS.pop(job_id, None)
    task = asyncio.create_task(janitor())
    yield
    task.cancel()


app = FastAPI(title="paraligilo", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _new_job_dir() -> tuple[str, Path]:
    job_id = secrets.token_urlsafe(10)
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=False)
    return job_id, d


async def _save_upload(uf: UploadFile, dst: Path) -> int:
    total = 0
    with dst.open("wb") as f:
        while chunk := await uf.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File too large (>{MAX_UPLOAD_MB}MB)")
            f.write(chunk)
    return total


async def _run_pipeline(job_id: str, api_key: str, mode: str, llm: bool, base_url: str = ""):
    async with RUN_SEM:
        j = JOBS[job_id]
        j["status"] = "running"
        j["started_at"] = time.time()
        env = os.environ.copy()
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        else:
            env.pop("OPENAI_API_KEY", None)
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env.pop("OPENAI_BASE_URL", None)
        cmd = [
            str(RUN_PY),
            str(j["orig_path"]),
            str(j["trans_path"]),
            "--name", "book",
            "--workdir", str(j["dir"]),
            "--mode", mode,
            "--sample", "100",
        ]
        if llm:
            cmd.append("--llm")
        log_path = j["dir"] / "run.log"
        try:
            with log_path.open("wb") as logf:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                )
                j["pid"] = proc.pid
                rc = await proc.wait()
            j["return_code"] = rc
            j["finished_at"] = time.time()
            if rc == 0:
                j["status"] = "done"
            else:
                j["status"] = "failed"
        except Exception as e:
            j["status"] = "failed"
            j["error"] = str(e)


def _job_progress(j: dict) -> dict:
    """Parse run.log for stage progress hints."""
    log_path = Path(j["dir"]) / "run.log"
    if not log_path.exists():
        return {"stage": "queued", "detail": ""}
    try:
        tail = log_path.read_text(errors="ignore")[-4000:]
    except Exception:
        tail = ""
    # detect last $ command line to know stage
    stages = []
    for ln in tail.splitlines():
        if ln.startswith("$ "):
            for s in ("extract.py", "embed.py", "anchors.py", "align_full.py",
                     "rescore_concat.py", "refine.py", "rescore_llm.py", "verify.py"):
                if s in ln:
                    stages.append(s.replace(".py", ""))
                    break
    cur = stages[-1] if stages else "starting"
    # try to pluck a "X/Y" progress line
    detail = ""
    for ln in reversed(tail.splitlines()):
        if "/" in ln and any(k in ln for k in ("embed:", "rescored", "dp:", "segment")):
            detail = ln.strip()
            break
    return {"stage": cur, "detail": detail, "tail": tail[-1500:]}


@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "has_server_key": bool(SERVER_FALLBACK_KEY),
        "max_mb": MAX_UPLOAD_MB,
    })


@app.post("/align", name="start_align")
@limiter.limit(RL_ALIGN)
async def start_align(
    request: Request,
    orig: UploadFile = File(...),
    trans: UploadFile = File(...),
    mode: str = Form("deterministic"),
    api_key: str = Form(""),
    base_url: str = Form(""),
    llm: Optional[str] = Form(None),
    use_server_key: Optional[str] = Form(None),
):
    if not orig.filename.lower().endswith(".epub"):
        raise HTTPException(400, "orig must be .epub")
    if not trans.filename.lower().endswith(".epub"):
        raise HTTPException(400, "trans must be .epub")
    if mode not in ("deterministic", "ai"):
        raise HTTPException(400, "mode must be deterministic or ai")

    chosen_key = ""
    chosen_base_url = base_url.strip()
    if mode == "ai":
        chosen_key = api_key.strip()
        if not chosen_key:
            if use_server_key and SERVER_FALLBACK_KEY:
                chosen_key = SERVER_FALLBACK_KEY
                chosen_base_url = ""  # don't leak user-set base URL onto server key
            else:
                raise HTTPException(400, "AI mode requires an API key (or check the server-key box if available)")

    job_id, jdir = _new_job_dir()
    orig_path = jdir / "orig.epub"
    trans_path = jdir / "trans.epub"
    await _save_upload(orig, orig_path)
    await _save_upload(trans, trans_path)

    JOBS[job_id] = {
        "id": job_id,
        "dir": jdir,
        "orig_path": orig_path,
        "trans_path": trans_path,
        "orig_name": orig.filename,
        "trans_name": trans.filename,
        "mode": mode,
        "status": "queued",
        "created_at": time.time(),
        "use_server_key": bool(mode == "ai" and use_server_key and not api_key.strip()),
    }
    asyncio.create_task(_run_pipeline(job_id, chosen_key, mode=mode, llm=bool(llm and mode == "ai"), base_url=chosen_base_url))
    return RedirectResponse(str(request.url_for("job_view", job_id=job_id)), status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse, name="job_view")
async def job_view(request: Request, job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return templates.TemplateResponse(request, "job.html", {
        "job": j, "progress": _job_progress(j),
    })


@app.get("/job/{job_id}/status", name="job_status")
@limiter.limit(RL_STATUS)
async def job_status(request: Request, job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    p = _job_progress(j)
    return JSONResponse({
        "status": j["status"],
        "stage": p["stage"],
        "detail": p["detail"],
        "tail": p["tail"],
        "elapsed": int(time.time() - j["created_at"]),
    })


def _final_jsonl(j: dict) -> Optional[Path]:
    f = j["dir"] / "book_final.jsonl"
    if f.exists() or f.is_symlink():
        # resolve symlink
        return f.resolve() if f.is_symlink() else f
    return None


@app.get("/job/{job_id}/download/{kind}", name="download")
@limiter.limit(RL_DOWNLOAD)
async def download(request: Request, job_id: str, kind: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    if j["status"] != "done":
        raise HTTPException(409, f"job not done yet (status={j['status']})")
    if kind == "alignment":
        f = _final_jsonl(j)
        if not f:
            raise HTTPException(500, "alignment missing")
        return FileResponse(f, filename="alignment.jsonl",
                            media_type="application/x-ndjson")
    if kind == "sample":
        f = j["dir"] / "book_sample.md"
        if not f.exists():
            raise HTTPException(500, "sample missing")
        return FileResponse(f, filename="sample.md", media_type="text/markdown")
    if kind == "log":
        f = j["dir"] / "run.log"
        return FileResponse(f, filename="run.log", media_type="text/plain")
    raise HTTPException(400, "unknown download kind")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8088"))
    root_path = os.environ.get("PARALIGILO_ROOT_PATH", "")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info",
                root_path=root_path,
                forwarded_allow_ips="*")  # trust nginx's X-Forwarded-*
