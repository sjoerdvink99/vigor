# Variables
DATA_FOLDER = data/input

# Target to run the data generation script
generate_data:
	@echo "Generating data..."
	@mkdir -p $(DATA_FOLDER)
	@python3 vigor/generator.py --output $(DATA_FOLDER)
	@echo "Data generated and saved in $(DATA_FOLDER)"