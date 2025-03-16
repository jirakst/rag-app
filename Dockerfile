# Use a minimal base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements file (if you have one)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (change as needed)
EXPOSE 8000

# Command to run the application
CMD ["streamlit", "run", "rag.py"]
