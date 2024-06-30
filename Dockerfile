FROM python:3.12.3-slim
LABEL maintainer="arun.com"

ENV PYTHONUNBUFFERED 1
WORKDIR /webapp

COPY ./requirements_docker.txt /requirements_docker.txt
COPY ./webapp /webapp
COPY ./models/models.joblib /models/models.joblib

EXPOSE 8000

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /requirements_docker.txt && \
    adduser --disabled-password --no-create-home webapp

ENV PATH="/py/bin:$PATH"

USER webapp