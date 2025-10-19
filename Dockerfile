# Base image with Python
FROM python:3.9-slim

# Create a working directory
WORKDIR /app

# Copy only requirements first (to leverage caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
# Use 8000 for development, testing, and containers — it’s the safe, common default.
# Reserve 80 for public-facing traffic in production behind a web server or load balancer.
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
