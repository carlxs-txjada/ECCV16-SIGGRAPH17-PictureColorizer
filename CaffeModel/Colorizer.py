import argparse
import matplotlib.pyplot as plt

import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()

from Caffe.demo_release import opt
