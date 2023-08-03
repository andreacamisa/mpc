ARG VARIANT=3.8-bullseye
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

USER vscode
RUN curl -sSL https://install.python-poetry.org | python -

# shell aliases to run CI pipeline locally
RUN echo 'alias do-format-check="docker build --target format-check ."' >> ~/.bashrc
RUN echo 'alias do-lint="docker build --target lint ."' >> ~/.bashrc
RUN echo 'alias do-test="docker build --target test ."' >> ~/.bashrc
