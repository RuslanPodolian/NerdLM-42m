sudo apt-get install python3

python3 -m venv env --system-site-packages

source env/bin/activate

pip install -r requirements.txt

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
