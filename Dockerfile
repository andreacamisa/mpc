FROM python:3.8-bookworm as builder

ARG UID=1000

ENV \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # create virtual environment in /app/.venv instead of poetry cache dir
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    # force creation of virtual environment
    POETRY_VIRTUALENVS_CREATE=1 \
    # poetry cache directory for Docker Buildkit cache mount
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install poetry==1.5.1

RUN useradd -u ${UID} --create-home app
USER app
WORKDIR /home/app

# create virtual environment and install dependencies
RUN touch README.md
# COPY --chown=app pyproject.toml poetry.lock ./
COPY --chown=app pyproject.toml ./
RUN --mount=type=cache,target=${POETRY_CACHE_DIR},uid=${UID} poetry install --no-root

FROM builder AS lint-test

ENV PATH="/home/app/.venv/bin:$PATH"

COPY --chown=app src src
COPY --chown=app tests tests
RUN --mount=type=cache,target=${POETRY_CACHE_DIR},uid=${UID} poetry install --with dev --no-root

# run linting and formatting
COPY --chown=app .ci .ci
RUN black src tests --check --diff --config ./.ci/black.cfg
RUN flake8 src tests --config ./.ci/flake8.ini
RUN mypy --config-file ./.ci/mypy.ini --python-version 3.8 src tests
RUN isort src tests --check-only --diff --settings-file ./.ci/isort.cfg

# run testing
RUN pytest -vv tests --log-cli-level INFO

FROM python:3.8-slim-bookworm as production

RUN useradd --create-home app
USER app
WORKDIR /home/app

ENV VIRTUAL_ENV=/home/app/.venv \
    PATH="${VIRTUAL_ENV}/bin:$PATH"

# copy over dependencies from builder stage
COPY --chown=app --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# install our code
COPY --chown=app src/mpc ./mpc

ENTRYPOINT ["python", "-m", "mpc.main"]
