#!/bin/bash

# Nginx Reverse Proxy Setup Script for Gradio App
# This script automates the Nginx configuration for your Gradio app

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Nginx Reverse Proxy Setup for Gradio ===${NC}\n"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Get domain name
read -p "Enter your domain name (e.g., yourdomain.com): " DOMAIN
if [ -z "$DOMAIN" ]; then
    echo -e "${RED}Domain name is required!${NC}"
    exit 1
fi

# Validate domain format (basic check)
if [[ ! $DOMAIN =~ ^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$ ]]; then
    echo -e "${YELLOW}Warning: Domain format may be invalid. Continuing anyway...${NC}"
fi

echo -e "\n${GREEN}Domain: ${DOMAIN}${NC}"
echo -e "${GREEN}Gradio app will be accessible at: http://${DOMAIN}${NC}\n"

# Step 1: Install Nginx
echo -e "${YELLOW}[1/5] Installing Nginx...${NC}"
if ! command -v nginx &> /dev/null; then
    apt update
    apt install nginx -y
    echo -e "${GREEN}✓ Nginx installed${NC}"
else
    echo -e "${GREEN}✓ Nginx already installed${NC}"
fi

# Step 2: Create Nginx configuration
echo -e "\n${YELLOW}[2/5] Creating Nginx configuration...${NC}"
CONFIG_FILE="/etc/nginx/sites-available/gradio"

cat > "$CONFIG_FILE" << EOF
server {
    listen 80;
    server_name ${DOMAIN} www.${DOMAIN};

    # Increase timeouts for long-running requests
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support (for Gradio's real-time features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Buffer settings
        proxy_buffering off;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

echo -e "${GREEN}✓ Configuration file created: ${CONFIG_FILE}${NC}"

# Step 3: Enable configuration
echo -e "\n${YELLOW}[3/5] Enabling Nginx configuration...${NC}"
if [ -L /etc/nginx/sites-enabled/gradio ]; then
    rm /etc/nginx/sites-enabled/gradio
fi
ln -s /etc/nginx/sites-available/gradio /etc/nginx/sites-enabled/

# Remove default site (optional)
if [ -L /etc/nginx/sites-enabled/default ]; then
    echo -e "${YELLOW}Removing default Nginx site...${NC}"
    rm /etc/nginx/sites-enabled/default
fi

# Step 4: Test configuration
echo -e "\n${YELLOW}[4/5] Testing Nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✓ Configuration test passed${NC}"
else
    echo -e "${RED}✗ Configuration test failed!${NC}"
    exit 1
fi

# Step 5: Restart Nginx
echo -e "\n${YELLOW}[5/5] Restarting Nginx...${NC}"
systemctl restart nginx
systemctl enable nginx

# Check status
if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✓ Nginx is running${NC}"
else
    echo -e "${RED}✗ Nginx failed to start!${NC}"
    systemctl status nginx
    exit 1
fi

# Summary
echo -e "\n${GREEN}=== Setup Complete! ===${NC}\n"
echo -e "Configuration file: ${CONFIG_FILE}"
echo -e "Enabled site: /etc/nginx/sites-enabled/gradio"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Verify DNS points to this server's IP"
echo -e "2. Test HTTP access: http://${DOMAIN}"
echo -e "3. (Optional) Set up HTTPS with: sudo certbot --nginx -d ${DOMAIN} -d www.${DOMAIN}"
echo -e "\n${YELLOW}Useful commands:${NC}"
echo -e "  Check Nginx status: sudo systemctl status nginx"
echo -e "  View access logs: sudo tail -f /var/log/nginx/access.log"
echo -e "  View error logs: sudo tail -f /var/log/nginx/error.log"
echo -e "  Test config: sudo nginx -t"
echo -e "  Reload config: sudo systemctl reload nginx"
echo -e "\n${GREEN}Done!${NC}\n"

