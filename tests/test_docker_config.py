"""tests/test_docker_config.py

Automated verification of Docker packaging, .dockerignore security quarantine,
and docker-compose specification for PulmoVision Stage 3A.
"""

from pathlib import Path
import re
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_exists_and_security_contract() -> None:
    """Verify Dockerfile structure, non-root user, and exposed ports."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Base image Python 3.11-slim
    assert "FROM python:3.11-slim" in content, "Dockerfile must use python:3.11-slim base"

    # 2. Non-root user setup
    assert re.search(r"useradd\s+.*pulmo", content), "Dockerfile must create non-root user pulmo"
    assert "USER pulmo" in content, "Dockerfile must switch to USER pulmo"

    # 3. Streamlit port binding
    assert "EXPOSE 8501" in content, "Dockerfile must expose port 8501"

    # 4. Streamlit health check
    assert "HEALTHCHECK" in content, "Dockerfile must define a HEALTHCHECK"
    assert "http://localhost:8501/_stcore/health" in content, "HEALTHCHECK must query /_stcore/health"

    # 5. Model cache configuration
    assert "TORCH_HOME=/app/.cache/torch" in content, "Dockerfile must set TORCH_HOME"


def test_dockerignore_excludes_quarantine_and_large_data() -> None:
    """Ensure strict isolation of patient data and quarantine holdout set."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    patterns = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # Crucial security invariants
    assert "test1.csv" in patterns or any("test1.csv" in p for p in patterns), (
        ".dockerignore MUST explicitly ignore test1.csv quarantine holdout"
    )
    assert any("data/raw" in p for p in patterns), ".dockerignore must exclude raw data"
    assert any(".git" in p for p in patterns), ".dockerignore must exclude .git directory"
    assert any("venv" in p for p in patterns), ".dockerignore must exclude local virtualenvs"


def test_docker_compose_valid_structure() -> None:
    """Verify docker-compose.yml configuration."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    with compose_path.open("r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data, "docker-compose.yml must define services"
    assert "pulmovision" in compose_data["services"], "Service 'pulmovision' must be present"

    pv_service = compose_data["services"]["pulmovision"]
    assert "8501:8501" in pv_service.get("ports", []), "Port 8501 must be mapped"
    assert "healthcheck" in pv_service, "docker-compose service must include healthcheck"
