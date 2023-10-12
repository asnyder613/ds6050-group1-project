default:
	@cat makefile

env:
	python3 -m venv env
	. env/bin/activate; pip install -r requirements.txt

caltechpedestriandataset:
	wget "https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1" -O caltechpedestriandataset.zip
	unzip caltechpedestriandataset.zip -d caltechpedestriandataset
	rm caltechpedestriandataset.zip


