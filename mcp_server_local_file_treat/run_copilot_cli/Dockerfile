FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js + npm (Node 20 from NodeSource)
RUN apt-get update \
 && apt-get install -y curl ca-certificates gnupg \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN npm install -g @github/copilot
CMD ["bash"]
