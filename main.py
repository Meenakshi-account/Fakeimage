# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Import necessary modules
from utils import data_loader
from models import fake_image_detection_model
from utils import evaluation_metrics

def main():
    # Load and preprocess real and fake image datasets
    real_images, fake_images = data_loader.load_and_preprocess_data()

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = data_loader.split_data(real_images, fake_images)

    # Create and train the fake image detection model
    model = fake_image_detection_model.create_model()
    model = fake_image_detection_model.train_model(model, train_data, val_data)

    # Evaluate the model on the test dataset
    test_metrics = evaluation_metrics.evaluate_model(model, test_data)
    print("Test Metrics:", test_metrics)

    # Save the trained model
    fake_image_detection_model.save_model(model)

if __name__ == "__main__":
    main()

