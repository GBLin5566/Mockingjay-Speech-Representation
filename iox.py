import pandas

import torch
import torch.utils.data as Data
from tqdm import tqdm

from runner_mockingjay import get_mockingjay_model
from downstream.model import example_classifier
from downstream.solver import get_mockingjay_optimizer
from utility.audio import extract_feature


train_data_csv = pandas.read_csv('/home/gblin/audio_bert/train_data.csv')
ids = train_data_csv.iloc[:, 0]
labels = torch.LongTensor(train_data_csv.iloc[:, 3]).cuda()
class_num = max(labels.tolist()) + 1
example_path = 'mockingjay-500000.ckpt'
solver = get_mockingjay_model(from_path=example_path)

classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=class_num).cuda()
params = list(solver.mockingjay.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

example_inputs = torch.zeros(len(ids), 800, 160).cuda()
for index, id_ in tqdm(enumerate(ids), desc="Transforming wav"):
    spec = torch.from_numpy(extract_feature(f'/home/gblin/audio_bert/data/{id_}.wav')).contiguous().cuda()
    example_inputs[index][:spec.shape[0]] = spec

torch_dataset = Data.TensorDataset(example_inputs, labels)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
for input_, label in loader:
    reps = solver.forward_fine_tune(spec=input_)
    loss = classifier(reps, label)
    print(f"Loss: {loss}")
    loss.backward()
    optimizer.step()
