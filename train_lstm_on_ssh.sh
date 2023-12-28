read -p "Enter the SSH command (e.g., ssh -p 21048 root@31.12.82.146 -L 8080:localhost:8080): " ssh_command

commands=(
    "
    apt install python3-venv && \
    cd /workspace/ && \
    git clone https://github.com/lkleinbrodt/auctioneer.git && \
    cd /workspace/auctioneer && \
    git pull && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install -r requirements.txt && \
    python3 src/lstm.py"
)

$ssh_command <<EOF
    # Run commands on the VM
    ${commands[@]}
EOF