FROM python:3.9
LABEL maintainer = "medvedev.daff@gmail.com"

# Copy requirements.txt to the working directory
COPY . .

# Install the Python dependencies on the virtual environment
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && rm requirements.txt