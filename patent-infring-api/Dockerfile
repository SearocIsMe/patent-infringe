# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install Memcached
RUN apt-get update && apt-get install -y memcached && apt-get clean \
    && pip install "fastapi[standard]"

# Set environment variables for Memcached
ENV MEMCACHED_PORT=11211
ENV MEMCACHED_MEMORY=64m
ENV TOKENIZERS_PARALLELISM=false
# Expose the Memcached port
EXPOSE 11211

# Start Memcached in the background
RUN echo "memcached -u memcache -m ${MEMCACHED_MEMORY} -p ${MEMCACHED_PORT} -d" > /usr/local/bin/start_memcached.sh
RUN chmod +x /usr/local/bin/start_memcached.sh

# Set the working directory in the container
WORKDIR /app

# Copy the FastAPI application code to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run both Memcached and FastAPI
CMD ["/bin/sh", "-c", "/usr/local/bin/start_memcached.sh && fastapi run main.py --host 0.0.0.0 --port 8000"]