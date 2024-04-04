# Use the official image as a parent image
FROM python:3.10.12

# Update package repositories
RUN apt-get update

# Install packages
RUN apt-get install -y openssh-server

# Copy the public key and set permissions
COPY authorized_key.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# Set permissions for SSH
RUN mkdir /var/run/sshd && \
    echo 'root:docker' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22

# Set the working directory in the container
WORKDIR /SpeakerVerification

# Clone the repository
RUN git clone https://github.com/Ironmomo/SpeakerVerificationBA.git .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define a volume for the data directory
VOLUME /SpeakerVerification/data

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]