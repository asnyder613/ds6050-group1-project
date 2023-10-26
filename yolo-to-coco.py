import os
import argparse
from pylabel import importer
import yaml

def main():
    parser = argparse.ArgumentParser(description="Import and analyze a dataset")
    parser.add_argument("--path_to_annotations", type=str, required=True, help="Path to annotations directory")
    parser.add_argument("--path_to_images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--path_to_yolo_yaml", type=str, required=True, help="Path to yolo yaml")
    parser.add_argument("--name", required=True, help="Name of the dataset")
    
    args = parser.parse_args()
    
    path_to_annotations = args.path_to_annotations
    path_to_images = args.path_to_images
    name = args.name

    with open(args.path_to_yolo_yaml, 'r') as file:
        content = file.read()

    data = yaml.load(content, Loader=yaml.FullLoader)

    classes = data['names']
    dataset = importer.ImportYoloV5(
        path=path_to_annotations,
        path_to_images=path_to_images,
        cat_names=classes,
        img_ext="png",
        name=name
    )
    
    print(dataset.df.head(5))
    print(f"Number of images: {dataset.analyze.num_images}")
    print(f"Number of classes: {dataset.analyze.num_classes}")
    print(f"Classes: {dataset.analyze.classes}")
    print(f"Class counts:\n{dataset.analyze.class_counts}")

    dataset.export.ExportToCoco(cat_id_index=1)

if __name__ == "__main__":
    main()