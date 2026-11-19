# Broker (and any CPU-only worker). Small: no CUDA, no torch.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 5555 5556
CMD ["maia-infer", "broker"]
