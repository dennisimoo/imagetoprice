# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create listings directory
RUN mkdir -p listings

# Expose port 3000
EXPOSE 3000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application on port 3000
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=3000)"]