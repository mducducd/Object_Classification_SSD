#dataloader
#network -> SSD300
#loss -> MultiBoxLoss
#optimizer
#training, validation

from lib import *
from make_datapath import make_datapath_list
from dataset import MyDataset, my_collate_fn
from transform import DataTransform
from extract_inform_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #enable if u have gpu
device = torch.device("cpu")
print("device:", device)
torch.backends.cudnn.benchmark = True

#dataloader
root_path = "./data/VOCdevkit/VOC2012"
train_img_list, train_anno_list, val_img_list, val_anno_list =make_datapath_list(root_path)

classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"]

color_mean = (104, 117, 123)
input_size = 300

train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
val_dataset = MyDataset(val_img_list, train_anno_list, phase="val", transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

batch_size = 16
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

#network
net = SSD(phase="train", cfg=cfg)
vgg_weights = torch.load("./data/weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# He's init
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# multiboxloss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

#optimizer 
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

#training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to gpu
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch+1) % 10 == 0:
                    net.eval()
                    print("---")*10
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                #move to GPU
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                #init optimizer
                optimizer.zero_grad()

                #forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c
                    
                    if phase == "train":
                        loss.backward() # caculate gradient

                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step() #update parameters

                        if (iteration %10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_start - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss, loss.item(), duration))

                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()


        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        loag_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(loag_epoch)

        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_log,csv")

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), "data/weights/ssd300" + str(epoch+1) + ".pth")

num_epochs = 30
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)