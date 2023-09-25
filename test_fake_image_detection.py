# Import necessary modules
from utils import data_loader
from models import fake_image_detection_model
from utils import evaluation_metrics

# Load the trained fake image detection model
model = fake_image_detection_model.load_model()

# Load and preprocess the test dataset
test_images = data_loader.load_and_preprocess_test_data()

# Evaluate the model on the test dataset
test_metrics = evaluation_metrics.evaluate_model(model, test_images)
print("Test Metrics:", test_metrics)
