FROM python:3.9
MAINTAINER Roman Medvedev <medvedev.daff@gmail.com>
COPY . .
RUN python -m venv