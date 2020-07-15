import itertools

import cv2
import matplotlib.pylab as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from itertools import cycle
cycol = cycle('bgrcmk')

class Utilities:
    @staticmethod
    def batch_image(image: np.ndarray, width, height, batch_size=1, channels=3):
        image = cv2.resize(image, (width, height))
        image = np.moveaxis(image, -1, 0)
        # image = image.transpose((2, 0, 1))
        # image = image.reshape((batch_size, channels, height, width))
        return image

    @staticmethod
    def make_prediction_time_chart(models: []):
        fig = plt.figure(figsize=(20, 20))
        plt.style.use("seaborn")
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        for idx, model in enumerate(models):
            y = np.array(model.stats["predict"])
            x = np.arange(0, len(y), dtype=np.int)
            ax = fig.add_subplot(2, 2, idx + 1)
            ax.set_xlabel("Frames", fontsize=15)
            ax.set_ylabel("Inference Time (ms)", fontsize=15)
            ax.step(x, y)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(model.__class__.__name__, fontsize=18)
        plt.show()

    @staticmethod
    def make_model_loading_time_chart(models: []):
        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize=(5, 5))
        model_names = [model.__class__.__name__ for model in models]
        model_loading_times = [model.stats["load_model"][0] for model in models]
        width = 0.75
        ax.bar(model_names, model_loading_times, width, color="blue")
        ax.set_ylabel('Loading time (ms)')
        ax.set_xlabel('Model Name')
        ax.set_title('Loading time by model')
        for index, data in enumerate(model_loading_times):
            ax.text(x=index, y=data + 1, s=f"{data} ms", fontdict=dict(fontsize=14), ha='center')
        plt.show()

    @staticmethod
    def make_model_performance_chart(models: [], metric_name="cpu_time"):
        fig = plt.figure(figsize=(20, 20))
        plt.style.use("seaborn")
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        for idx, m in enumerate(models):
            layers = []
            model_metrics = m.perf_counts
            for layer_name, layer_info in model_metrics.items():
                if layer_info["status"] == "EXECUTED" \
                        and layer_info["real_time"] > 0 \
                        and layer_info["cpu_time"] > 0:
                    layers.append(layer_info)
            # group layers by type
            layers = sorted(layers, key=lambda l: l["layer_type"])

            labels = []
            sizes = []
            for layer_group_type, layer_group in itertools.groupby(layers, key=lambda l: l["layer_type"]):
                labels.append(layer_group_type)
                sizes.append(np.sum([l[metric_name] for l in layer_group]))
            # make chart
            explode = [0] * len(labels)
            explode[np.argmax(sizes)] = 0.1
            ax = fig.add_subplot(2, 2, idx + 1)
            ax.set_title(m.__class__.__name__, fontsize=18)
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
            ax.axis('equal')
        plt.show()


    @staticmethod
    def make_model_perform_chart_with_annot(models: [], metric_name="cpu_time"):
        fig = plt.figure(figsize=(20, 20))
        plt.style.use("seaborn")
        #fig.subplots_adjust(hspace=0.3, wspace=0.2)
        for idx, m in enumerate(models):
            layers = []
            model_metrics = m.perf_counts
            for layer_name, layer_info in model_metrics.items():
                if layer_info["status"] == "EXECUTED" \
                        and layer_info["real_time"] > 0 \
                        and layer_info["cpu_time"] > 0:
                    layers.append(layer_info)
            # group layers by type
            layers = sorted(layers, key=lambda l: l["layer_type"])
            labels = []
            values = []
            counts = []
            for layer_group_type, layer_group in itertools.groupby(layers, key=lambda l: l["layer_type"]):
                layer_list = list(layer_group)
                labels.append(layer_group_type)
                values.append(np.sum([l[metric_name] for l in layer_list]))
                counts.append(len(layer_list))
            values = np.array(values)
            cmap = plt.cm.get_cmap(plt.cm.terrain, 143)
            n = 30
            colors = [cmap(n*i) for i in range(len(labels))]
            explode = [0] * len(labels)
            explode[np.argmax(values)] = 0.1
            ax = fig.add_subplot(2, 2, idx + 1)
            patches, texts = ax.pie(values,labels=counts, explode=explode, colors=colors, shadow=True, startangle=90, radius=1.2)
            # sort legend
            patches, labels, dummy = zip(*sorted(zip(patches, labels, values), key=lambda x: x[2],reverse=True))
            percents = 100. * values / values.sum()
            labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, percents)]
            ax.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=12)
            ax.set_title(m.__class__.__name__, fontsize=18, pad=20)
        plt.show()
