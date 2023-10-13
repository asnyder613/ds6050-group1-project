import os
import argparse
from pylabel import importer

def main():
    parser = argparse.ArgumentParser(description="Import and analyze a dataset")
    parser.add_argument("--path_to_annotations", type=str, required=True, help="Path to annotations directory")
    parser.add_argument("--path_to_images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--name", required=True, help="Name of the dataset")
    
    args = parser.parse_args()
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    path_to_annotations = current_script_path + args.path_to_annotations
    path_to_images = current_script_path + args.path_to_images
    name = args.name
    
    classes = ['person']
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