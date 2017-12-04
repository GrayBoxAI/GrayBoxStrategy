#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
define scikit-learn style colormap transform for RGBA outputs (label / discrete)
helper code to map arbitrary class label nparray into rgb color that plato read
"""
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import seaborn.apionly as sns
from sklearn import preprocessing

def _fan_out(x, cmap):
    ret = np.asarray(cmap[x])
    # add the alpha value
    ret = np.append(ret, 1)
    return ret.reshape((1, -1))


class RGBALabelTransformer():
    "define scikit-learn style colormap transform for RGBA outputs (discrete labels)"
    def __init__(self, colormap_type):
        self.colormap_type = colormap_type

    def fit(self, data):
        self.labelencoder = preprocessing.LabelEncoder()
        self.labelencoder.fit(data)
        self.classes_ = self.labelencoder.classes_
        self.num_classes_ = len(self.classes_)
        self.colormap = sns.color_palette(self.colormap_type,
                                          n_colors=self.num_classes_
                                         )

    def transform(self, data):
        if len(data.shape) != 1 and not 1 in data.shape:
            raise ValueError("1d array only is accepted")
        labeled_data = self.labelencoder.transform(data)
        ret = np.zeros((len(data), 4))
        for i in range(len(data)):
            ret[i, :] = _fan_out(labeled_data[i], self.colormap)
        return ret

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def palplot(self):
        sample = self.classes_
        sns.palplot(self.transform(sample))
        print("Number of classes:" + str(self.num_classes_))
        print(self.classes_)


class RGBAContinuousTransformer():
    "define scikit-learn style colormap transform for RGBA outputs (continous)"

    def __init__(self, colormap_type, resolution):
        self.colormap_type = colormap_type
        self.resolution = resolution
    
    def fit(self, data):
        self.max_ = np.max(data)
        self.min_ = np.min(data)
        self.range_ = self.max_ - self.min_
        self.seamap = sns.color_palette(self.colormap_type, n_colors=self.resolution)
        self.norm = mpl.colors.Normalize(vmin=self.min_, vmax=self.max_)
        self.colormap = cm.ScalarMappable(norm=self.norm,
                                          cmap=mpl.colors.ListedColormap(self.seamap)
                                         )

    def transform(self, data):
        if len(data.shape) != 1 and not 1 in data.shape:
            raise Exception("1d array only is accepted")
        ret = self.colormap.to_rgba(data)
        ret = ret.reshape((len(data), 4))
        return ret

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def palplot(self):
        step = self.range_ / self.resolution
        sample = np.arange(self.min_, self.max_, step)
        sns.palplot(self.transform(sample))
        print("Sample Min:" + str(self.min_) + \
              ", Sample Max:" + str(self.max_) + \
              ", Resolution:" + str(self.resolution)
             )
