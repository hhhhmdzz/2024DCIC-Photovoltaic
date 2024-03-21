import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def plot_importance(features_importance,
                    topk=None,
                    feats_col='feature',
                    impt_col='importance',
                    img_file='feature_importance-lgb-hold_out.png',
                    save_img=False,
                    show_img=True):
    topk = len(features_importance) if topk is None else topk
    plt.figure(figsize=(20, topk // 4), dpi=100)
    p = sns.barplot(y=feats_col,
                    x=impt_col,
                    data=features_importance.iloc[:topk, :],
                    orient='horizontal')
    show_values(p, "h", space=0)
    plt.tight_layout()

    if save_img:
        plt.savefig(img_file)
    if show_img:
        plt.show()
    else:
        plt.close()
