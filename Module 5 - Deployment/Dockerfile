FROM python:3.12.1-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "./"

RUN uv sync --locked --no-install-project

COPY "predict.py" "model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uv", "run","--no-project", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]