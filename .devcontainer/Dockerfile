ARG VARIANT=3.8-bullseye
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

RUN apt-get update && apt-get -y install cmake

USER vscode
RUN curl -sSL https://install.python-poetry.org | python -
