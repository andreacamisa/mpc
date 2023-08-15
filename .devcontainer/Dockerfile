ARG VARIANT=3.8-bullseye
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

USER vscode
RUN curl -sSL https://install.python-poetry.org | python -

# shell aliases to run CI pipeline locally
RUN echo 'alias do-black-check="black $WORKSPACEDIR/src $WORKSPACEDIR/tests --check --diff --config $WORKSPACEDIR/.ci/black.cfg"' >> ~/.bashrc
RUN echo 'alias do-isort-check="isort $WORKSPACEDIR/src $WORKSPACEDIR/tests --check-only --diff --settings-file $WORKSPACEDIR/.ci/isort.cfg"' >> ~/.bashrc
RUN echo 'alias do-flake8-check="flake8 $WORKSPACEDIR/src $WORKSPACEDIR/tests --config $WORKSPACEDIR/.ci/flake8.ini"' >> ~/.bashrc
RUN echo 'alias do-format-check="do-black-check; do-isort-check; do-flake8-check"' >> ~/.bashrc
RUN echo 'alias do-lint="mypy --config-file $WORKSPACEDIR/.ci/mypy.ini --python-version 3.8 $WORKSPACEDIR/src $WORKSPACEDIR/tests"' >> ~/.bashrc
RUN echo 'alias do-test="pytest -vv $WORKSPACEDIR/tests --log-cli-level INFO"' >> ~/.bashrc
