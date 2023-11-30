from model import config
import sys
import os
import torch
import cv2
import numpy

# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    print("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

print(f"**** loading object detector at {model_path}...")
model = torch.load(model_path).to(config.DEVICE)
model.eval()
print(f"**** object detector loaded")

results_labels = dict()
results_bbox = dict()

for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    print(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        filename, start_x, start_y, end_x, end_y, label = row.split(',')
        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        data.append((filename, start_x, start_y, end_x, end_y, label))

    print(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_labels[mode]['all'] = []
    for label_str in config.LABELS:
        results_labels[mode][label_str] = []

    results_bbox[mode] = dict() 
    results_bbox[mode]['all'] = 0
    for label_str in config.LABELS:
        results_bbox[mode][label_str] = 0

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        label_predictions, bbox_prediction = model(image)

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        values = bbox_prediction.cpu().detach().numpy()[0]
        start_x, start_y, end_x, end_y = values[0], values[1], values[2], values[3]
        start_x *= w 
        end_x *= w 
        start_y *= h 
        end_y *= h

        # Compare to gt data
        results_labels[mode]['all'].append(label == gt_label)
        results_labels[mode][gt_label].append(label == gt_label)

        # Not the best metrics
        metrics = (start_x - gt_start_x)**2 + (start_y - gt_start_y)**2 + (end_x - gt_end_x)**2 + (end_y - gt_end_y)**2
        results_bbox[mode]['all'] += metrics
        results_bbox[mode][gt_label] += metrics

        if label != gt_label:
            print(f"\tFailure at {filename}")


# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    print(f'\n*** {mode} set accuracy')
    print(f"\tMean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all']))}")
    print(f"\tMean boundig box metrics for all labels: "
          f"{numpy.mean(numpy.array(results_bbox[mode]['all']))}")
    
    for label_str in config.LABELS:
        print(f'\n\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[mode][label_str]))}')
        print(f'\t\t {numpy.sum(results_labels[mode][label_str])} over '
              f'{len(results_labels[mode][label_str])} samples')
        print(f"\tMean boundig box metrics for label {label_str}: "
          f"{(numpy.array(results_bbox[mode][label_str]))}")

