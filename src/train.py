from src.data import DogClfDataset, Letterbox

data_path = 'data/Images'
annotation_path = 'data/lists/train_list.mat'

resizer = Letterbox(512, 512)

ds = DogClfDataset(data_path, annotation_path, resizer=resizer)
ds[0]