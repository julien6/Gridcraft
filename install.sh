#!/usr/bin/env bash

if command -v python3.10 >/dev/null 2>&1; then
	echo "✅ Python 3.10 was found."
else
	echo "❌ Python 3.10 is not installed. Please install it (ex: sudo apt install python3.10 python3.10-venv)."
	exit 1
fi

echo "🛠️  Creating virtual Python 3.10 environment..."
python3.10 -m venv ./venv

echo "Activation de l'environnement virtuel..."
source ./venv/bin/activate

pip install -e .
