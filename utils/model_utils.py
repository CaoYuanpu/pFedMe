import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def read_cifa_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 10 # should be muitiple of 10
    NUM_LABELS = 1
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len os user:", user, j,
                    "len data", len(X[user]), num_samples)

    print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
        
    # random.seed(1)
    # np.random.seed(1)
    # NUM_USERS = 1 # should be muitiple of 10
    # NUM_LABELS = 10
    # # Setup directory for train/test data
    # cifa_data_image = []
    # cifa_data_label = []

    # cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    # cifa_data_label.extend(trainset.targets.cpu().detach().numpy())

    # cifa_data_image = np.array(cifa_data_image)
    # cifa_data_label = np.array(cifa_data_label)

    # # Create data structure
    # train_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # # Setup 5 users
    # # for i in trange(5, ncols=120):
    # for i in range(NUM_USERS):
    #     uname = 'f_{0:05d}'.format(i)
    #     train_data['users'].append(uname) 
    #     train_data['user_data'][uname] = {'x': cifa_data_image.tolist(), 'y': cifa_data_label.tolist()}
    #     train_data['num_samples'].append(len(cifa_data_image))

    # #-----------------------------------TEst -------------------------------------#
    # cifa_data_image_test = []
    # cifa_data_label_test = []
    # cifa_data_image_test.extend(testset.data.cpu().detach().numpy())
    # cifa_data_label_test.extend(testset.targets.cpu().detach().numpy())
    # cifa_data_image_test = np.array(cifa_data_image_test)
    # cifa_data_label_test = np.array(cifa_data_label_test)

    # cifa_data = []

    # # Create data structure
    # test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    # for i in range(NUM_USERS):
    #     num_samples = len(cifa_data_image_test)
    #     test_data['users'].append(uname) 
    #     test_data['user_data'][uname] = {'x': cifa_data_image_test.tolist(), 'y': cifa_data_label_test.tolist()}
    #     test_data['num_samples'].append(num_samples)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    if(dataset == "Cifar10"):
        clients, groups, train_data, test_data = read_cifa_data()
        return clients, groups, train_data, test_data

    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

class OrderDataset(Dataset):
    def __init__(self, data,label,transformer=None,device="cuda:0"):
        self.data = data
        self.label = torch.from_numpy(label).squeeze(0).to(device).long()
        self.device =device
        self.transformer = transformer

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.data[idx]
        # image = Image.fromarray(image)
        if self.transformer:
            image = self.transformer(image)
        # return image.to(self.device), self.label[idx].to(self.device)
        return image, self.label[idx]

def dir_data(dataset, num_clients, alpha, data_path, device='cuda:0'):
    if dataset == 'Cifar10':
        channel = 3
        # im_size = (32, 32)
        im_size = 32
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_data_whole = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        test_data_whole = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = train_data_whole.classes

    elif dataset == 'CIFAR100':
        channel = 3
        # im_size = (32, 32)
        im_size = 32
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_data_whole = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        test_data_whole = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = train_data_whole.classes

    order = np.argsort(train_data_whole.targets)
    ordered_train_data = train_data_whole.data[order]
    ordered_train_label = np.array(train_data_whole.targets)[order]
    num_samples_c = int(len(ordered_train_label) / num_classes)

    test_data = test_data_whole.data
    test_label = np.array(test_data_whole.targets)
    num_test_samples_client = int(len(test_label) / num_clients)
    
    print(f'load dataset: [channel] {channel} [im size] {im_size} [num_classes] {num_classes} [num_samples / cls] {num_samples_c} [num_test_samples / clients] {num_test_samples_client}')

    
    # data partition
    alpha_list = [alpha for _ in range(num_clients)]
    total_number = 0

    min_size = 0
    while min_size < 2500:
        client_index_list = [[] for _ in range(num_clients)]
        total_number = 0
        for i in range(num_classes):
            client_port = np.random.dirichlet(alpha=alpha_list)
            number = client_port * num_samples_c
            number = np.int32(number)
            left = num_samples_c - number.sum()
            if left > 0:
                lucky_client = np.random.randint(low=0, high=num_clients)
                number[lucky_client] += left
            total_number += number
            for j in range(num_clients):
                client_index_list[j].extend(list(range(i*num_samples_c+number[:j].sum(),i*num_samples_c+number[:j+1].sum())))
        min_size = min([len(idx_j) for idx_j in client_index_list])

    print()
    print("Num of training samples / client:")
    print(total_number, sum(total_number))
    print("Details of data partition:")
    cls_counts = {}
    for i, client_index in enumerate(client_index_list):
        cls_counts[i] = {}
        for idx in client_index:
            cls_ = idx // num_samples_c
            if cls_ not in cls_counts[i]:
                cls_counts[i][cls_] = 1
            else:
                cls_counts[i][cls_] += 1
    print(cls_counts)
    print()
    
    # init distiller and clients
    trainsets = []
    testsets = []
    for i in range(num_clients):
        client_dataset = OrderDataset(ordered_train_data[client_index_list[i]],ordered_train_label[client_index_list[i]], transform, device=device)
        # client_dataloader = DataLoader(client_dataset, batch_size=500, shuffle=True)
        trainsets.append(client_dataset)

        test_dataset = OrderDataset(test_data[i*num_test_samples_client:(i+1)*num_test_samples_client],test_label[i*num_test_samples_client:(i+1)*num_test_samples_client], transform, device=device)
        # test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        testsets.append(test_dataset)
    return trainsets, testsets

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
        #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
