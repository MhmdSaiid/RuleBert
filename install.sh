conda install -c potassco clingo
sudo apt install gcc
pwd
cd data_generation/lpmln
pwd
python setup.py
cd ..
chmod u+x lpmln/*
chmod u+x lpmln/binSupport/*
cd ../
pip install -r requirements.txt