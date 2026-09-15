#!/bin/bash
# MultiHazard-VLM Phase 1 Docker Setup Script
# For Linux/Mac users or Windows with WSL

echo ""
echo "============================================================"
echo "   MultiHazard-VLM Phase 1 Docker Setup"
echo "============================================================"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "============================================================"
    echo "   $1"
    echo "============================================================"
    echo ""
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "   Please install Docker from https://www.docker.com/products/docker-desktop"
    exit 1
fi

print_section "Step 1: Check Docker & Docker Compose"
docker --version
docker-compose --version

print_section "Step 2: Display Available Commands"
echo "Usage: ./setup_docker.sh [command]"
echo ""
echo "Commands:"
echo "  up              - Start all services in background"
echo "  down            - Stop all services"
echo "  restart         - Restart all services"
echo "  logs            - View logs from all services"
echo "  logs api        - View API logs"
echo "  logs postgres   - View PostgreSQL logs"
echo "  logs redis      - View Redis logs"
echo "  status          - Show service status"
echo "  shell api       - Open shell in API container"
echo "  shell postgres  - Open shell in PostgreSQL container"
echo "  db-init         - Initialize database"
echo "  metrics         - View Prometheus metrics"
echo "  clean           - Remove all containers and volumes"
echo ""

# Get command from argument
COMMAND=${1:-help}

case $COMMAND in
    up)
        print_section "Starting Services"
        docker-compose up -d
        echo "✅ Services started!"
        echo ""
        echo "📍 Access Points:"
        echo "  🌐 API:        http://localhost:8000"
        echo "  📊 Dashboard:  http://localhost:8503"
        echo "  📈 Prometheus: http://localhost:9090"
        echo "  📊 Grafana:    http://localhost:3000 (admin/admin)"
        echo ""
        echo "Run 'docker-compose logs -f' to see logs"
        ;;
    
    down)
        print_section "Stopping Services"
        docker-compose down
        echo "✅ Services stopped!"
        ;;
    
    restart)
        print_section "Restarting Services"
        docker-compose restart
        echo "✅ Services restarted!"
        ;;
    
    logs)
        print_section "Service Logs"
        docker-compose logs -f $2
        ;;
    
    status)
        print_section "Service Status"
        docker-compose ps
        ;;
    
    shell)
        SERVICE=${2:-api}
        print_section "Opening shell in $SERVICE container"
        docker-compose exec $SERVICE /bin/bash
        ;;
    
    db-init)
        print_section "Initializing Database"
        docker-compose exec api python init_database.py
        echo "✅ Database initialized!"
        ;;
    
    metrics)
        print_section "Prometheus Metrics"
        curl -s http://localhost:8000/metrics | head -30
        ;;
    
    clean)
        print_section "Cleaning Up"
        echo "⚠️  This will remove all containers and volumes!"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            echo "✅ Cleanup complete!"
        else
            echo "Cancelled"
        fi
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo "Run: $0 help"
        exit 1
        ;;
esac
