import os
from Support import Support
from Segmenter import Segmenter
from Compare import compare
from random import shuffle


class BatchSegmenter:
    def __init__(self, map_path: str, label_path: str, test_path: str, gt_path: str, batch_size: int, support_size: int,
                 invert_label: bool = True):
        self.segmented_batches, self.support_batches, self.test_filenames = BatchSegmenter.create_and_segment_batches(
            map_path, label_path, test_path, gt_path,
            invert_label,
            batch_size,
            support_size)

    @staticmethod
    def create_and_segment_batches(map_path: str, label_path: str, test_path: str, gt_path: str, invert_label: bool,
                                   batch_size: int,
                                   support_size: int):
        segmented_batches = []
        support_batches = []
        filenames = os.listdir(map_path)
        for batch in range(batch_size):
            shuffle(filenames)
            support_batch = filenames[:support_size]
            support_batches.append(support_batch)
            support = Support(map_path, label_path, support_batch, invert_label=invert_label)
            print(f"Batch n°{batch+1} - Number of sub-images): {support.maps.shape[1]}")
            segmenter = Segmenter(test_path, support)
            enhanced_images = compare(gt_path, test_path, segmenter)
            segmented_batches.append(enhanced_images)
        return segmented_batches, support_batches, os.listdir(test_path)
