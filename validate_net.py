from config import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *

def validate_net(cfg):
    """
    validating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    _, validation_set = return_dataset(cfg)

    params = {'batch_size': cfg.test_batch_size, 'shuffle': True, 'num_workers': 4}

    validation_loader = data.DataLoader(validation_set, **params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    basenet_list = {'volleyball': Basenet_volleyball, 'collective': Basenet_collective}
    gcnnet_list = {'volleyball': GCNnet_volleyball, 'collective': GCNnet_collective}

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.dataset_name]
        model = GCNnet(cfg)

        # Load backbone
        checkpoint = torch.load(cfg.stage2_model_path)

        # original saved file with DataParallel
        state_dict = checkpoint["state_dict"]

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']


    else:
        assert (False)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])

    model = model.to(device=device)
    test_list = {'volleyball': test_volleyball, 'collective': test_collective}
    test = test_list[cfg.dataset_name]

    test_info = test(validation_loader, model, device, epoch, cfg)
    print(test_info)


def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()

    epoch_timer = Timer()
    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_data_test = [b.to(device=device) for b in batch_data_test]
            batch_size = batch_data_test[0].shape[0]
            num_frames = batch_data_test[0].shape[1]

            actions_in = batch_data_test[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data_test[3].reshape((batch_size, num_frames))

            # forward
            actions_scores, activities_scores = model((batch_data_test[0], batch_data_test[1]))

            # Predict actions
            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
            actions_labels = torch.argmax(actions_scores, dim=1)

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)

            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

            # Get accuracy
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return test_info

def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()

    epoch_timer = Timer()
    with torch.no_grad():
        i = 1
        for batch_data in data_loader:
            ground_truth = data_loader.dataset.anns[9][i]
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size, num_frames))
            bboxes_num = batch_data[4].reshape(batch_size, num_frames)

            # forward
            actions_scores, activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))

            actions_in_nopad = []

            if cfg.training_stage == 1:
                actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
                bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
                for bt in range(batch_size * num_frames):
                    N = bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt, :N])
            else:
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,

            if cfg.training_stage == 1:
                activities_in = activities_in.reshape(-1, )
            else:
                activities_in = activities_in[:, 0].reshape(batch_size, )

            actions_loss = F.cross_entropy(actions_scores, actions_in)
            actions_labels = torch.argmax(actions_scores, dim=1)  # ALL_N,
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)  # B,
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

            # Visualize the result
            print("Frame id: ", ground_truth["frame_id"])
            print("Bounding box position: ", ground_truth["bboxes"])
            print("Predict actions: ", actions_labels)
            print("Predict activities: ", activities_labels)

            # to-do: input Frame id, Bounding box position, Predict actions, Predict activities, 
            # output visualized frame-like video with boxes around each person 
            # and a "captioning" (predict actions in words and predict group activites in words)
            # complete this visualize function
            visualize(ground_truth["frame_id"], ground_truth["bboxes"], actions_labels, activities_labels)

            # Get accuracy
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)
            i += 10

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return test_info


def visualize(param, param1, actions_labels, activities_labels):
    pass
