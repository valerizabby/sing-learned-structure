#!/bin/bash
# Запуск UI для генерации музыки
# Использование: ./run_ui.sh [port]

set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Активируем venv если есть
if [ -d "venv" ]; then
  source venv/bin/activate
fi

echo "======================================"
echo "  SingLS Music Generation UI"
echo "  http://localhost:${PORT}"
echo "======================================"

uvicorn ui.server:app --host 0.0.0.0 --port "$PORT" --reload
