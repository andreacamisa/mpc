ARG VARIANT=3.8-bullseye
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

USER vscode
RUN curl -sSL https://install.python-poetry.org | python -

# shell alias to run CI pipeline locally
RUN echo 'alias lint-test="docker build --output type=tar,dest=/dev/null --target lint-test ."' >> ~/.bashrc
