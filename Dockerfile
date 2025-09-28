# Use an official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "server:server", "--host", "0.0.0.0", "--port", "8080"]
