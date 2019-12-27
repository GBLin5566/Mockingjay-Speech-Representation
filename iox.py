import os
from time import gmtime, strftime

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
test_data_csv = pandas.read_csv('/home/gblin/audio_bert/test_data.csv')
test_ids = test_data_csv.iloc[:, 0]
test_labels = torch.LongTensor(test_data_csv.iloc[:, 3]).cuda()

class_num = max(labels.tolist()) + 1
num_of_epochs = 1000
example_path = 'mockingjay-500000.ckpt'
solver = get_mockingjay_model(from_path=example_path)

classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=class_num).cuda()
params = list(solver.mockingjay.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-5, warmup_proportion=0.7, training_steps=50000)

if os.path.exists('./example_inputs'):
    print("Load example_inputs")
    example_inputs = torch.load('./example_inputs')
else:
    example_inputs = torch.zeros(len(ids), 800, 160).cuda()
    for index, id_ in tqdm(enumerate(ids), desc="Transforming train wav"):
        spec = torch.from_numpy(extract_feature(f'/home/gblin/audio_bert/data/{id_}.wav')).contiguous().cuda()
        if spec.shape[0] < 800:
            example_inputs[index][:spec.shape[0]] = spec
        else:
            example_inputs[index] = spec[:800]
    torch.save(example_inputs, './example_inputs')

if os.path.exists('./test_example_inputs'):
    print("Load test_example_inputs")
    test_example_inputs = torch.load('./test_example_inputs')
else:
    test_example_inputs = torch.zeros(len(test_ids), 800, 160).cuda()
    for index, id_ in tqdm(enumerate(test_ids), desc="Transforming test wav"):
        spec = torch.from_numpy(extract_feature(f'/home/gblin/audio_bert/data/tmp/{id_}.wav')).contiguous().cuda()
        if spec.shape[0] < 800:
            test_example_inputs[index][:spec.shape[0]] = spec
        else:
            test_example_inputs[index] = spec[:800]
    torch.save(test_example_inputs, './test_example_inputs')

torch_dataset = Data.TensorDataset(example_inputs, labels)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
test_torch_dataset = Data.TensorDataset(test_example_inputs, test_labels)
test_loader = Data.DataLoader(
    dataset=test_torch_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)
for epoch in range(num_of_epochs):
    for batch, (input_, label) in enumerate(loader):
        reps = solver.forward_fine_tune(spec=input_)
        loss = classifier(reps, label)
        if batch % 10 == 0:
            print(f"Epoch {epoch} @ Batch {batch}, Loss: {loss}")
        loss.backward()
        optimizer.step()

    test_loss = []
    for input_, label in test_loader:
        reps = solver.forward_fine_tune(spec=input_)
        loss = classifier(reps, label)
        test_loss.append(loss.detach())
    print(f"Epoch {epoch}, Test loss: {sum(test_loss) / len(test_loss)}")


PATH_TO_SAVE_YOUR_MODEL = f'iox_finetuned@{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}.ckpt'
states = {'Classifier': classifier.state_dict(), 'Mockingjay': solver.mockingjay.state_dict()}
torch.save(states, PATH_TO_SAVE_YOUR_MODEL)
