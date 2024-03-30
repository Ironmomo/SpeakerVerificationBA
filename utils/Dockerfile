# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /SpeakerVerification

# Clone the repository
RUN git clone https://github.com/Ironmomo/SpeakerVerificationBA.git .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define a volume for the data directory
VOLUME /SpeakerVerification/data