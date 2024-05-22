# Use the official Python 3.11 image based on Debian Buster as the base image.
FROM python:3.11-buster

# Update the package list and install Node.js and npm.
RUN apt-get update

# Create a new user named "user" with user ID 1000.
RUN useradd -m -u 1000 user

# Set environment variables for the home directory and local bin path.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's app directory.
WORKDIR $HOME/app

# Upgrade pip to the latest version.
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements.txt file to the working directory and install Python dependencies.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Adjust permissions for the app directory.
RUN chown -R user:user /home/user/app

# Switch back to the "user" account.
USER user

# Copy all remaining application code to the working directory.
COPY --chown=user . $HOME/app

# Create a directory for markdown files and set permissions.
RUN mkdir -p $HOME/app/Files
RUN chmod 777 $HOME/app/Files

# Expose port 8080 for the application
EXPOSE 8080

# Define the command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "src/app:app"]
