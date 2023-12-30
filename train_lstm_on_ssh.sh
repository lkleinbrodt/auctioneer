

read -p "Enter the SSH command (e.g., ssh -p 21048 root@31.12.82.146 -L 8080:localhost:8080): " ssh_command
# Parse the input into variables
IFS=' ' read -ra ssh_parts <<< "$ssh_command"
VM_PORT=""
VM_USER=""
VM_HOST=""
for part in "${ssh_parts[@]}"; do
    if [ "$next_is_port" = true ]; then
        VM_PORT="$part"
        next_is_port=false
    else
        case $part in
            -p) next_is_port=true;;
            *) 
                if [[ $part == *@* ]]; then
                    VM_USER=$(echo "$part" | cut -d "@" -f1)
                    VM_HOST=$(echo "$part" | cut -d "@" -f2)
                fi
                ;;
        esac
    fi
done

commands=(
    "
    apt install python3-venv && \
    cd /workspace/ && \
    if [ ! -d auctioneer ] ; then
        git clone https://github.com/lkleinbrodt/auctioneer.git auctioneer
    fi && \
    cd /workspace/auctioneer && \
    git pull && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip3 install -r requirements.txt
    "
)




$ssh_command <<EOF
    # Run commands on the VM
    ${commands[@]}
EOF

scp -P $VM_PORT .env $VM_USER@$VM_HOST:/workspace/auctioneer/.env

commands=(
    "
    cd /workspace/auctioneer && \
    source venv/bin/activate && \
    python3 src/lstm.py
    "
)

$ssh_command <<EOF
    # Run commands on the VM
    ${commands[@]}
EOF

# #terminate the instance:

# # lol at this but hey it works!
# # output=$(vastai show instances --raw)
# # instance_id=$(python -c "import sys, json; data = json.load(sys.stdin); print(data[0]['id'] if data else '')" <<< "$output")
# # vastai reboot instance $instance_id