from __future__ import print_function
import json
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial


# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename, curv_filename, pidx_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class ShapeNet(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, split, class_choice, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=False, center='point', point_tuple=1, cache_capacity=1,
                 point_count_std=0.0, sparse_patches=False, neighbor_search_method='r', final_patch_size=175, normal_channel_shapenet=True):

        # initialize parameters
        self.root = root
        self.split = split
        self.class_choice = class_choice
        self.patch_features = patch_features
        self.normal_channel_shapenet = normal_channel_shapenet
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed
        self.neighbor_search_method = neighbor_search_method
        self.include_normals = False
        self.include_curvatures = False
        self.include_neighbor_normals = False
        self.final_patch_size = final_patch_size

        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            elif pfeat == 'neighbor_normals':
                self.include_neighbor_normals = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        self.shape_cache = Cache(cache_capacity, self, ShapeNet.load_shape_by_index)

        # Load categories and filter by class_choice
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if self.class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}

        # Load train/test/val split
        ids_file = None
        if self.split == 'train':
            ids_file = 'shuffled_train_file_list.json'
        elif self.split == 'val':
            ids_file = 'shuffled_val_file_list.json'
        elif self.split == 'test':
            ids_file = 'shuffled_test_file_list.json'
        elif self.split == 'trainval': # Special case for trainval
            pass # Handled below
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', 'test', or 'trainval'.")

        self.shape_names = [] # This will store paths like 'category_folder/model_id'
        self.shape_patch_count = []
        self.patch_radius_absolute = []

        all_model_ids_in_split = set()
        if self.split == 'trainval':
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            all_model_ids_in_split = train_ids.union(val_ids)
        elif ids_file:
            with open(os.path.join(self.root, 'train_test_split', ids_file), 'r') as f:
                all_model_ids_in_split = set([str(d.split('/')[2]) for d in json.load(f)])
        
        for category_synset, category_folder_name in self.cat.items():
            category_path = os.path.join(self.root, category_folder_name)
            if not os.path.isdir(category_path):
                print(f"Warning: Category directory not found {category_path}")
                continue
            
            for model_txt_filename in os.listdir(category_path):
                if not model_txt_filename.endswith('.txt'):
                    continue
                model_id = model_txt_filename[:-4] # Remove .txt

                if model_id not in all_model_ids_in_split:
                    continue

                # This shape_name_prefix is what load_shape_by_index will use (e.g., 'Airplane/10155655850468db78d106ce0a280f87')
                shape_name_prefix_for_list = os.path.join(category_folder_name, model_id)
                
                # Define paths for original .txt and target .npy files
                original_txt_file_path = os.path.join(category_path, model_txt_filename)
                # Base path for .npy files, load_shape() will append .xyz.npy, .normals.npy
                xyz_npy_path = os.path.join(category_path, model_id + '.xyz') 
                normals_npy_path = os.path.join(category_path, model_id + '.normals')

                # Ensure .npy files exist, create if not
                # Check for .xyz.npy (points)
                if not os.path.exists(xyz_npy_path + '.npy'):
                    print(f'Converting {original_txt_file_path} to .xyz.npy')
                    try:
                        data = np.loadtxt(original_txt_file_path).astype(np.float32)
                        points = data[:, 0:3]
                        np.save(xyz_npy_path + '.npy', points)
                    except Exception as e:
                        print(f"Error processing points from {original_txt_file_path}: {e}")
                        continue # Skip this model if points cannot be processed
                
                # Check for .normals.npy if needed
                if self.include_normals and self.normal_channel_shapenet:
                    if not os.path.exists(normals_npy_path + '.npy'):
                        print(f'Converting {original_txt_file_path} to .normals.npy')
                        try:
                            # Reload data if not already loaded or to ensure consistency
                            data = np.loadtxt(original_txt_file_path).astype(np.float32)
                            if data.shape[1] >= 6:
                                normals = data[:, 3:6]
                                np.save(normals_npy_path + '.npy', normals)
                            else:
                                print(f"Warning: Normals requested (normal_channel_shapenet=True) but data in {original_txt_file_path} has < 6 columns. Skipping normal saving for this file.")
                                # We will proceed without normals for this shape if include_normals is true but normal_channel_shapenet couldn't provide them
                        except Exception as e:
                            print(f"Error processing normals from {original_txt_file_path}: {e}")
                            # Decide if we skip the model or proceed without these normals
                
                self.shape_names.append(shape_name_prefix_for_list)

        # Initialize random number generator for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)
            
        # Get basic information for each shape in the dataset
        # This loop now iterates over the self.shape_names populated from ShapeNet structure
        temp_shape_names = []
        temp_shape_patch_count = []
        temp_patch_radius_absolute = []

        for shape_ind_new, shape_name_from_list in enumerate(self.shape_names):
            # The actual loading of .npy files and kdtree construction happens in self.shape_cache.get()
            # which calls load_shape_by_index -> load_shape. load_shape_by_index uses self.shape_names[shape_ind]
            # So, the shape_ind_new here is the correct index for the current self.shape_names list.
            shape = self.shape_cache.get(shape_ind_new) 
            
            if shape is None or shape.pts is None: 
                print(f"Warning: Shape at index {shape_ind_new} ({shape_name_from_list}) could not be loaded properly by cache. Skipping.")
                continue # Skip this shape if it couldn't be loaded
            
            temp_shape_names.append(shape_name_from_list)
            if shape.pidx is None:
                temp_shape_patch_count.append(shape.pts.shape[0])
            else:
                temp_shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            temp_patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])
        
        self.shape_names = temp_shape_names
        self.shape_patch_count = temp_shape_patch_count
        self.patch_radius_absolute = temp_patch_radius_absolute

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3, dtype=torch.float)

        neighbor_normals = torch.zeros(self.final_patch_size * len(self.patch_radius_absolute[shape_ind]), 3,
                                dtype=torch.float)
        patch_pts_valid = []

        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        effective_points_num = np.array([], dtype=int)
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):

            if self.neighbor_search_method == 'r':
                patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))
                patch_scale = rad
            elif self.neighbor_search_method == 'k':
                point_distances, patch_point_inds = shape.kdtree.query(shape.pts[center_point_ind, :], k=self.points_per_patch)
                rad = max(point_distances)
                patch_scale = rad

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            point_count = int(min(self.points_per_patch, len(patch_point_inds)))
            effective_points_num = np.append(effective_points_num, point_count)

            # randomly decrease the number of points to get patches with different point densities
            if self.point_count_std > 0:
                point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
                point_count = min(point_count, len(patch_point_inds))

            # if there are too many neighbors, pick a random subset
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]

            start = s*self.points_per_patch
            end = start+point_count
            scale_ind_range[s, :] = [start, end]

            patch_pts_valid += list(range(start, end))

            # convert points to torch tensors
            patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])


            # center patch (central point at origin - but avoid changing padded zeros)
            if self.center == 'mean':
                patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[start:end, :].mean(0)
            elif self.center == 'point':
                patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
            elif self.center == 'none':
                pass # no centering
            else:
                raise ValueError('Unknown patch centering option: %s' % (self.center))

            # normalize size of patch (scale with 1 / patch radius)
            # if self.neighbor_search_method == 'r':
            patch_pts[start:end, :] = patch_pts[start:end, :] / rad
            # elif self.neighbor_search_method == 'k':
            #     patch_pts[start:end, :] = patch_pts[start:end, :] / torch.max(torch.norm(patch_pts[start:end, :], p=2, dim=1))


        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        if self.include_neighbor_normals:
            neighbor_normals[start:end, :] = torch.from_numpy(shape.normals[patch_point_inds[:175], :])

        if self.include_curvatures:
            patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            # if self.neighbor_search_method == 'r':
            patch_curv = patch_curv * self.patch_radius_absolute[shape_ind][0]
            # elif self.neighbor_search_method == 'k':
            #     patch_curv = patch_curv / torch.max(torch.norm(patch_pts[start:end, :], p=2, dim=1))

        if self.use_pca:

            # compute pca of points in the patch:
            # center the patch around the mean:
            pts_mean = patch_pts[patch_pts_valid, :].mean(0)
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts[patch_pts_valid, :]))
            patch_pts[patch_pts_valid, :] = torch.mm(patch_pts[patch_pts_valid, :], trans)

            cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
            cp_new = torch.matmul(cp_new, trans)

            # re-center on original center point
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - cp_new

            if self.include_normals:
                patch_normal = torch.matmul(patch_normal, trans)

            if self.include_neighbor_normals:
                neighbor_normals = torch.matmul(neighbor_normals, trans)

        else:
            trans = torch.eye(3).float()


        # get point tuples from the current patch
        if self.point_tuple > 1:
            patch_tuples = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple, dtype=torch.float)
            for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
                start = scale_ind_range[s, 0]
                end = scale_ind_range[s, 1]
                point_count = end - start

                tuple_count = point_count**self.point_tuple

                # get linear indices of the tuples
                if tuple_count > self.points_per_patch:
                    patch_tuple_inds = self.rng.choice(tuple_count, self.points_per_patch, replace=False)
                    tuple_count = self.points_per_patch
                else:
                    patch_tuple_inds = np.arange(tuple_count)

                # linear tuple index to index for each tuple element
                patch_tuple_inds = np.unravel_index(patch_tuple_inds, (point_count,)*self.point_tuple)

                for t in range(self.point_tuple):
                    patch_tuples[start:start+tuple_count, t*3:(t+1)*3] = patch_pts[start+patch_tuple_inds[t], :]


            patch_pts = patch_tuples

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal,)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            elif pfeat == 'neighbor_normals':
                patch_feats = patch_feats + (neighbor_normals,)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        return (patch_pts,) + patch_feats + (trans,)  + (patch_scale,) #+ (effective_points_num,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        return load_shape(point_filename, normals_filename, curv_filename, pidx_filename)