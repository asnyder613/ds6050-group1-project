from pylabel import importer

path_to_annotations = "/home/ybt7qf/data/datasets/caltechpedestriandataset/labels/train/" # or /val/
path_to_images = "/home/ybt7qf/data/datasets/caltechpedestriandataset/images/train/" # or /val/

classes = ['person']
dataset = importer.ImportYoloV5(path=path_to_annotations,
                                path_to_images=path_to_images, 
                                cat_names=classes,
                                img_ext="png", 
                                name="caltechpedestriandataset_train") # or /val/

print(dataset.df.head(5))

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")

dataset.export.ExportToCoco(cat_id_index=1)