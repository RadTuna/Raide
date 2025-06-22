
pip install pip-tools
conda install -c conda-forge pynini==2.1.5

pip-compile requirements.in
pip install -r requirements.txt

pip install descript-audio-codec --no-deps
pip install descript-audiotools --no-deps
