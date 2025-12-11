
#!/bin/bash

# Nombre del entorno virtual
ENV_NAME=".env"

#echo "==> Verificando que Python 3.9 esté instalado..."
#if ! command -v python3.9 &> /dev/null; then
#    echo "❌ Python 3.9 no está instalado. Ejecuta: sudo apt install python3.9 python3.9-venv"
#    exit 1
#fi

echo "==> Creando entorno virtual con Python 3.9..."
python3.9 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "==> Actualizando pip..."
python3.9 -m pip install --upgrade pip --break-system-packages

echo "==> Descargando binarios de PyTorch 1.10.2 + cu113..."
wget -q --show-progress https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://download.pytorch.org/whl/cu113/torchaudio-0.10.2-cp39-cp39-linux_x86_64.whl

echo "==> Instalando PyTorch + CUDA 11.3..."
python3.9 -m pip install torch-1.10.2+cu113-cp39-cp39-linux_x86_64.whl --break-system-packages
python3.9 -m pip install torchvision-0.11.3+cu113-cp39-cp39-linux_x86_64.whl --break-system-packages
python3.9 -m pip install torchaudio-0.10.2-cp39-cp39-linux_x86_64.whl --break-system-packages

echo "==> Instalando dependencias adicionales..."
python3.9 -m pip install -r requirements.txt --break-system-packages

echo "==> Verificando instalación de PyTorch y GPU..."
python3.9 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detectada'}')"

echo ""
echo "✅ Entorno virtual '$ENV_NAME' creado y PyTorch 1.10.2+cu113 instalado correctamente."
echo "ℹ️ Para activarlo más tarde, usa:"
echo "   source $ENV_NAME/bin/activate"
