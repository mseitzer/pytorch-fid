import numpy as np
import torch
import torchvision.transforms as TF

from tqdm import tqdm

def calculate_inception_score(data, batch_size, device, num_workers=1, splits=10):
    '''
    Here, decide to use the inceptionv3 on Pytorch
    '''
    from torchvision.models.inception import inception_v3
    import torch.nn.functional as F

    model = inception_v3(pretrained=True, transform_input=False)
    model.to(device)
    model.eval()
    # dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(data,
                                            batch_size = batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=num_workers)
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          drop_last=False,
    #                                          num_workers=num_workers)

    pred_arr = np.empty((len(data), 1000))

    start_idx = 0

    for (batch,_) in tqdm(dataloader):
        batch = batch.to(device)
        
        batch = F.interpolate(batch,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)
        batch = batch*2 - 1

        with torch.no_grad():
            pred = F.softmax(model(batch), dim = 1).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    scores = []
    for i in range(splits):
        part = pred_arr[(i*pred_arr.shape[0]//splits):((i+1)*pred_arr.shape[0]//splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))) # kl divergence       
        kl = np.mean(np.sum(kl, 1)) # can be modify with scipy.stats.entropy
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

if __name__ == '__main__':   
    from torchvision.datasets import CIFAR10
    data = CIFAR10('D:/DATA/CIFAR-10/old-cifar-10-batches-py/torchvision/', train=True, transform=TF.ToTensor(), download=True)
    

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    m, s = calculate_inception_score(data=data,
                                    batch_size=50,
                                    device=device,
                                    num_workers=0,
                                    splits=10)
    print(f'Inception score: {round(m,4)} +- {round(s,4)}')