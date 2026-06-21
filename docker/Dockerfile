FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="FlashVision <dev@flashvision.dev>"
LABEL description="FlashMed: Medical AI for imaging analysis and diagnostics"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY flashmed/ ./flashmed/

RUN pip install --no-cache-dir -e ".[all]"

COPY configs/ ./configs/
COPY examples/ ./examples/

RUN flashmed check

ENTRYPOINT ["flashmed"]
CMD ["version"]
