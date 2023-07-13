# System libs
import os
import datetime
import argparse
from packaging.version import Version

# Numerical libs
import numpy as np
import torch

# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from broden_dataset_utils.joint_dataset import broden_dataset
from utils import maskrcnn_colorencode, remove_small_mat

def setup_cuda(args):
    """
    Sets the CUDA device to use.
    """
    torch.cuda.set_device(args.gpu_id)

def build_model(args):
    """
    Builds the model by loading the encoder and decoder and placing them in a segmentation module.
    """
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        nr_classes=args.nr_classes,
        weights=args.weights_decoder,
        use_softmax=True)

    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module.cuda()

    return segmentation_module

def prepare_data_loader(args):
    """
    Prepares the data loader by creating a TestDataset and DataLoader from the input arguments.
    """
    list_test = [{'fpath_img': args.test_img}]
    dataset_val = TestDataset(
        list_test, args, max_sample=args.num_val)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    return loader_val

def initialize_result_matrix(seg_size, args):
    """
    Initialize the result matrix to store predictions.
    """
    pred_ms = {}
    for k in ['object', 'material']:
        pred_ms[k] = torch.zeros(1, args.nr_classes[k], *seg_size)
    pred_ms['part'] = []
    for _, object_label in enumerate(broden_dataset.object_with_part):
                n_part = len(broden_dataset.object_part[object_label])
                pred_ms['part'].append(torch.zeros(1, n_part, *seg_size))
    pred_ms['scene'] = torch.zeros(1, args.nr_classes['scene'])
    return pred_ms

def postprocess_prediction(pred_ms):
    """
    Postprocess predictions.
    """
    pred_ms['scene'] = pred_ms['scene'].squeeze(0)
    for k in ['object', 'material']:
        _, p_max = torch.max(pred_ms[k].cpu(), dim=1)
        pred_ms[k] = p_max.squeeze(0)
    for idx_part, _ in enumerate(broden_dataset.object_with_part):
        _, p_max = torch.max(pred_ms['part'][idx_part].cpu(), dim=1)
        pred_ms['part'][idx_part] = p_max.squeeze(0)
    return pred_ms

def test(segmentation_module, loader, args):
    """
    Test function to generate prediction and visualize results.

    Parameters:
    segmentation_module: the module that will be used for segmentation task.
    -> encoder + decoder
    loader: test data loader.
    args: command line arguments.
    """
    segmentation_module.eval()

    for i, data in enumerate(loader):
        # Process data
        data = data[0]
        seg_size = data['img_ori'].shape[0:2]

        with torch.no_grad():
            pred_ms = initialize_result_matrix(seg_size, args)

            # Store the results
            for img in data['img_data']:
                # Forward pass
                feed_dict = async_copy_to({"img": img}, args.gpu_id)
                pred = segmentation_module(feed_dict, seg_size=seg_size)
                for k in ['scene', 'object', 'material']:
                    pred_ms[k] = pred_ms[k] + pred[k].cpu() / len(args.imgSize)
                for idx_part, _ in enumerate(broden_dataset.object_with_part):
                    pred_ms['part'][idx_part] += pred['part'][idx_part].cpu() / \
                        len(args.imgSize)

            pred_ms = postprocess_prediction(pred_ms)
            pred_ms = as_numpy(pred_ms)

        # Visualize the original and result images
        visualize_result(data, pred_ms, args)

        print('[{}] iter {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i))

def main(args):
    """
    Main function that organizes all the steps for the prediction and visualization of results.
    """
    # Setup CUDA and build the model
    setup_cuda(args)
    segmentation_module = build_model(args)

    # Prepare data loader
    loader_val = prepare_data_loader(args)

    # Test the model and visualize results
    test(segmentation_module, loader_val, args)

    print('Inference done!')

def visualize_result(data, preds, args):
    """
    Visualization of prediction results.

    Parameters:
    data: original image data.
    preds: prediction results including 'object', 'part', 'scene', 'material'.
    args: command line arguments.
    """
    # Setup for color selection for each label
    np.random.seed(233)  # Fix the seed for consistency in color.
    color_list = np.random.rand(1000, 3) * .7 + .3

    # Save the original image
    img = data['img_ori']
    if args.save_ori_img != 0:
        cv2.imwrite(os.path.join(args.result, "original_image.jpg"), img)

    # Save the object prediction result
    object_result = preds['object']
    object_result_colored = maskrcnn_colorencode(
        img, object_result, color_list)
    cv2.imwrite(os.path.join(args.result, "object{}.png".format(args.suffix)),
                object_result_colored)

    # Save the part prediction result
    # A part result is masked by the valid object.
    img_part_pred, part_result = img.copy(), preds['part']
    valid_object = np.zeros_like(object_result)
    present_obj_labels = np.unique(object_result)
    for obj_part_index, object_label in enumerate(broden_dataset.object_with_part):
        object_mask = (object_result == object_label)
        valid_object += object_mask
        part_result_masked = part_result[obj_part_index] * object_mask
        present_part_label = np.unique(part_result_masked)
        if len(present_part_label) == 1:
            continue
        img_part_pred = maskrcnn_colorencode(
            img_part_pred, part_result_masked + object_mask, color_list)
    cv2.imwrite(os.path.join(args.result, "part{}.png".format(args.suffix)), img_part_pred)

    # Save the scene prediction result
    print("scene shape: {}".format(preds['scene'].shape))
    scene_top5 = np.argsort(-preds['scene'])[:5]
    with open(os.path.join(args.result, "scene{}.txt".format(args.suffix)), 'w') as f:
        f.write("scene pred:\n")
        scene_info = ["{}({}) {:.4f}".format(
            l, broden_dataset.names['scene'][l], preds['scene'][l])
            for l in scene_top5]
        f.write("\n".join(scene_info))

    # Save the material prediction result
    material_result = preds['material']
    img_material_result = maskrcnn_colorencode(
        img, remove_small_mat(material_result * (valid_object > 0), object_result), color_list)
    cv2.imwrite(os.path.join(args.result, "material{}.png".format(args.suffix)),
                img_material_result)


if __name__ == '__main__':
    assert Version(torch.__version__) >= Version('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_img', required=True)
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_40',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='./',
                        help='folder to output visualization results')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu_id for evaluation')
    parser.add_argument('--save_ori_img', default=1, type=int,
                        help='save original image')

    args = parser.parse_args()
    print(args)

    nr_classes = broden_dataset.nr.copy()
    nr_classes['part'] = sum(
        [len(parts) for obj, parts in broden_dataset.object_part.items()])
    args.nr_classes = nr_classes

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix + '.pth')
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix + '.pth')

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
