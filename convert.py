import pandas as pd
import numpy as np
import argparse, time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='User kNN')
parser.add_argument('--dilation', '-d', type=int, default='1', help='Enter dilation scale')
parser.add_argument('--fillna', '-f', default='-20', help='How to fill nodes without features')
parser.add_argument('--side', '-s', type=str, default='left', help='choose left or right')
parser.add_argument('--subjects', type=int, default=-1, help='number of subjects to be converted')
parser.add_argument('--debug', type=int, default=0, help='Debugging')
args = parser.parse_args()

class Convert:
    def __init__(self, dilation=1, fillna=-20, side='left', subjects=-1, debug=True):
        self.dilation = dilation
        self.fillna = fillna
        self.side = side
        self.subjects = range(subjects) if subjects >= 0 else range(867)
        self.debug = debug

        self.load_data(self.side, self.dilation)


    def load_data(self, side, dilation):

        names = ['vert_index_0', 'x', 'y', 'z', 'is_border', 'vtx_raw', 'vert_index_1']
        self.feats = ['area', 'curv', 'jacobian_white', 'sulc', 'thickness', 'volume']

        NODE_PATH = '../../reference/'
        FEAT_HEAD = '../res/full_data_matrix/oasis1&3_ALL_'
        FEAT_TAIL = '-fsaverage.npy'

        # preset
        if side=='left':
            node_file = NODE_PATH + 'fsaverage-lh_cortex_flat.csv'
            feat_side = 'lh_'

        elif side=='right':
            node_file = NODE_PATH + 'fsaverage-rh_cortex_flat.csv'
            feat_side = 'rh_'

        else:
            pass

        # Raw Nodes
        print("--- Loading Nodes ---")
        self.raw_nodes = pd.read_csv(node_file, header=0, names=names)

        # Features of 149k
        print("--- Loading Features ---")
        idx = self.raw_nodes.vert_index_0.values.astype(int)
        if self.debug:
            self.features = {
                f: FEAT_HEAD + feat_side + f + FEAT_TAIL for f in self.feats
            }
            print(self.features)
        else:
            self.features = {
                f: np.load(FEAT_HEAD + feat_side + f + FEAT_TAIL) for f in self.feats
            }

        # dilated
        print("--- Making Frame ---")
        dilated = pd.DataFrame({
            'idx': self.raw_nodes.vert_index_0.astype(int),
            'x' : self.raw_nodes.x.apply(lambda x: dilation*x).values.astype(int),
            'y' : self.raw_nodes.y.apply(lambda y: dilation*y).values.astype(int),
        })

        # make nodes
        print("--- Find Eigen-Nodes ---")
        nodes = set()
        for x, y in zip(dilated.x, dilated.y):
            nodes.add((x, y))

        # make data of (x, y, degenerate_list)
        print("--- Generate Frame ---")
        deg_idx = dict()
        for i, node in enumerate(nodes):
            x, y = node
            intersect = list(dilated.index[(dilated.x==x) & (dilated.y==y)])
            deg_idx[node] = intersect
        del dilated, nodes

        X, Y = [n[0] for n in deg_idx.keys()], [n[1] for n in deg_idx.keys()]
        lst = list(deg_idx.values())
        self.frame = pd.DataFrame({
            'x': X, 'y': Y, 'idx_lst': lst
        })
        del deg_idx


    def convert(self, sub):
        #print('--- Start Converting ---')
        # Averaging
        averaged = dict()
        x_min, y_min = abs(self.frame.x.min()), abs(self.frame.y.min())
        for lst, (x, y) in zip(self.frame.idx_lst.values, zip(self.frame.x.values, self.frame.y. values)):
            tmp = []
            for feat in self.features.values():
                tmp.append(feat[sub, lst].mean())
            averaged[(x+x_min, y+y_min)] = tmp


        # save
        x_range = self.frame.x.max() - self.frame.x.min() + 1
        y_range = self.frame.y.max() - self.frame.y.min() + 1

        img = np.zeros((x_range, y_range, 6))
        for x in tqdm(range(x_range)):
            for y in range(y_range):
                img[x, y, :] = averaged[(x, y)] if (x, y) in averaged.keys() else np.array([-20 for _ in range(6)])

        PATH = '../res/npy_img/right/right_' + str(sub) + '.npy'
        np.save(PATH, img)
        return img

    def save(self):

        for sub in tqdm(self.subjects):
            start = time.time()
            self.convert(sub)
            print("{}-th done, took {}".format(sub, time.time() - start))



if __name__=='__main__':

    c = Convert(dilation=args.dilation, fillna=args.fillna, side=args.side,
                subjects=args.subjects, debug=args.debug)
    c.save()