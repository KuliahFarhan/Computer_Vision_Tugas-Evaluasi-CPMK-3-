import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / '01_instance_segmentation_kaggle.ipynb').read_text(encoding='utf-8'))


def cells(tag):
    return [''.join(c['source']) for c in NOTEBOOK['cells']
            if c.get('metadata', {}).get('tags') == [tag]]


def execute(tag, namespace):
    for source in cells(tag):
        exec('\n'.join(source.splitlines()[1:]), namespace)


class RfDetrPredictionTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(np=np, needs_model=False, ACTIVE_IDS=list(range(1, 75)),
                       INDEX_TO_ID={i: i + 1 for i in range(74)},
                       SCORE_FLOOR=0.001, MAX_DETECTIONS=100)
        execute('rf_predictor', self.ns)

    def predict(self, labels, scores=None, masks=True):
        if scores is None:
            scores = [0.8] * len(labels)
        result = SimpleNamespace(class_id=np.asarray(labels), confidence=np.asarray(scores),
                                 mask=np.ones((len(labels), 2, 2), dtype=bool) if masks else None)
        self.ns['rf_model'] = SimpleNamespace(predict=lambda image, threshold: result)
        return self.ns['predict_rfdetr'](None)

    def test_background_74_does_not_crash_or_shift_foreground_labels(self):
        result = self.predict([74, 0, 73], [0.99, 0.8, 0.7])
        self.assertEqual([r[0] for r in result], [1, 74])
        self.assertEqual(result[0][2].dtype, np.uint8)

    def test_background_is_filtered_before_detection_limit(self):
        self.ns['MAX_DETECTIONS'] = 2
        result = self.predict([74, 74, 0, 73], [0.99, 0.98, 0.9, 0.8])
        self.assertEqual([r[0] for r in result], [1, 74])

    def test_hard3_uses_its_own_background_slot(self):
        self.ns.update(ACTIVE_IDS=[4, 12, 74], INDEX_TO_ID={0: 4, 1: 12, 2: 74})
        result = self.predict([3, 2, 0], [0.99, 0.8, 0.7])
        self.assertEqual([r[0] for r in result], [74, 4])

    def test_unknown_ids_are_not_silently_discarded(self):
        for label in [-1, 75]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'Class ID'):
                self.predict([label])

    def test_non_finite_scores_are_rejected(self):
        for score in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(score=score), self.assertRaises(FloatingPointError):
                self.predict([0], [score])

    def test_no_masks_or_foreground_produces_empty_predictions(self):
        self.assertEqual(self.predict([0], masks=False), [])
        self.assertEqual(self.predict([74]), [])
        self.assertEqual(self.predict([]), [])

    def test_mismatched_output_shapes_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Dimensi'):
            self.predict([0, 1], [0.7])


class RunConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(Path=Path)
        execute('config', self.ns)
        self.ns.update(DATASET_ID='dataset', ACTIVE_IDS=[1, 2], WORK=Path('/tmp/cpmk3-test'),
                       WORKER_SOURCE_ID='source', run_name='run-test', STAGED_DATASET='',
                       fingerprint=lambda data: hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())
        execute('sequence_config', self.ns)

    def test_recipe_changes_reach_worker_and_change_signature(self):
        for key, value in [('YOLO_MASK_RATIO', 4), ('YOLO_MOSAIC', 0.3),
                           ('MASKRCNN_LR', 0.0025), ('RFDETR_AMP_DTYPE', None),
                           ('RFDETR_COMPUTE_VAL_LOSS', False), ('YOLO_VARIANT', 'n'),
                           ('MASKRCNN_SCHEDULER', 'step'), ('RFDETR_VARIANT', 'nano')]:
            with self.subTest(key=key):
                original = self.ns[key]
                _, baseline, _ = self.ns['make_job_config']('rfdetr')
                self.ns[key] = value
                config, modified, _ = self.ns['make_job_config']('rfdetr')
                self.assertEqual(config[key], value)
                self.assertNotEqual(baseline, modified)
                self.ns[key] = original

    def test_resume_rejects_changed_training_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / 'last.pth'
            checkpoint.write_bytes(b'regression fixture')
            config = dict(model='maskrcnn', scenario='all_classes', dataset_id='dataset',
                          category_ids=[1, 2], smoke_test=False, resolution={'maskrcnn': [480, 640]},
                          training_options={'maskrcnn': {'lr': 0.005}})
            (folder / 'run_config.json').write_text(json.dumps(config))
            self.ns.update(RESUME_CHECKPOINT=str(checkpoint), MODE='train', MODEL='maskrcnn', INFERENCE_CHECKPOINT='',
                           RUN_DIR=folder, RUN_CONFIG={**config, 'training_options': {'maskrcnn': {'lr': 0.0025}}},
                           read_json=lambda path: json.loads(Path(path).read_text()))
            with self.assertRaisesRegex(AssertionError, 'Recipe berubah'):
                execute('checkpoint_guard', self.ns)

    def test_legacy_checkpoint_can_be_evaluated_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, destination = folder / 'input', folder / 'output'
            source.mkdir()
            destination.mkdir()
            checkpoint = source / 'best.pth'
            checkpoint.write_bytes(b'regression fixture')
            config = dict(model='rfdetr', scenario='all_classes', dataset_id='dataset',
                          category_ids=[1, 2], smoke_test=False, resolution={'rfdetr': 312}, run_label='run01')
            (source / 'run_config.json').write_text(json.dumps(config))
            self.ns.update(RESUME_CHECKPOINT='', MODE='predict', MODEL='rfdetr', INFERENCE_CHECKPOINT=str(checkpoint),
                           RUN_DIR=destination, RUN_CONFIG={**config, 'resolution': {'rfdetr': 384}}, run_name='run01_evalfix',
                           read_json=lambda path: json.loads(Path(path).read_text()),
                           write_json=lambda path, data: Path(path).write_text(json.dumps(data)))
            execute('checkpoint_guard', self.ns)
            execute('inference_settings', self.ns)
            self.assertEqual(self.ns['CHECKPOINT'], checkpoint)
            self.assertEqual(self.ns['RFDETR_SIZE'], 312)
            self.assertEqual(json.loads((destination / 'run_config.json').read_text())['source_run_label'], 'run01')
            self.assertEqual(self.ns['RUN_CONFIG']['run_label'], 'run01_evalfix')
            self.assertEqual(json.loads((source / 'run_config.json').read_text())['run_label'], 'run01')

    def test_local_staging_path_does_not_invalidate_drive_resume(self):
        config, before, _ = self.ns['make_job_config']('rfdetr')
        self.ns['STAGED_DATASET'] = '/content/cpmk3_workspace/raw_new_session'
        staged, after, _ = self.ns['make_job_config']('rfdetr')
        self.assertNotEqual(config['STAGED_DATASET'], staged['STAGED_DATASET'])
        self.assertEqual(before, after)

    def test_existing_runner_signature_survives_the_cuda_rng_resume_fix(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.ns.update(WORK=Path(temporary), LEGACY_RNG_SOURCE_ID='old-worker',
                           restore_job=lambda signature, folder: None,
                           read_json=lambda path: json.loads(path.read_text()),
                           write_json=lambda path, value: path.write_text(json.dumps(value)))
            execute('sequence_resume', self.ns)
            for model in ('yolo11', 'maskrcnn'):
                _, signature, folder = self.ns['make_job_config'](model, source_id='old-worker')
                folder.mkdir(parents=True)
                (folder / 'job_signature.json').write_text(json.dumps({'signature': signature}))
                if model == 'yolo11':
                    (folder / 'completion.json').write_text(json.dumps({'signature': signature}))
                    for name in self.ns['required_job_files'](model):
                        (folder / name).write_bytes(b'output')
                else:
                    (folder / 'last.pth').write_bytes(b'checkpoint')
                config, returned_signature, _, skip = self.ns['prepare_job'](model)
                self.assertEqual(returned_signature, signature)
                self.assertEqual(skip, model == 'yolo11')
                if model == 'maskrcnn':
                    self.assertEqual(config['RESUME_CHECKPOINT'], str(folder / 'last.pth'))

            self.ns['SEED'] += 1
            with self.assertRaisesRegex(AssertionError, 'Konfigurasi berubah'):
                self.ns['prepare_job']('maskrcnn')


class MaskRcnnResumeTests(unittest.TestCase):
    def test_last_checkpoint_restores_cuda_rng_from_cpu_tensors(self):
        class CudaState:
            def cpu(self):
                return np.array([42, 7], dtype=np.uint8)

        def set_rng(states):
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0].dtype, np.uint8)
            np.testing.assert_array_equal(states[0], [42, 7])
            restored.append(True)

        restored = []
        state = {k: {} for k in ('model', 'optimizer', 'scheduler', 'scaler')}
        state.update(epoch=13, best_ap=0.14, history=[{'epoch': 13}], python_rng=None,
                     numpy_rng=None, torch_rng=CudaState(), cuda_rng=[CudaState()])
        loader = object()
        loadable = SimpleNamespace(load_state_dict=lambda data: None, parameters=lambda: [])
        torch = SimpleNamespace(
            utils=SimpleNamespace(data=SimpleNamespace(DataLoader=lambda *args, **kwargs: loader)),
            optim=SimpleNamespace(SGD=lambda *args, **kwargs: loadable,
                                  lr_scheduler=SimpleNamespace(ReduceLROnPlateau=lambda *args, **kwargs: loadable)),
            amp=SimpleNamespace(GradScaler=lambda *args, **kwargs: loadable),
            load=lambda *args, **kwargs: state, set_rng_state=lambda value: None,
            cuda=SimpleNamespace(set_rng_state_all=set_rng),
        )
        namespace = dict(torch=torch, random=SimpleNamespace(setstate=lambda value: None),
                         np=SimpleNamespace(random=SimpleNamespace(set_state=lambda value: None)),
                         IndoorCocoDataset=lambda *args, **kwargs: None, TRAIN_ROOT=Path('/dataset'),
                         MASKRCNN_BATCH=2, MASKRCNN_LR=0.005, MASKRCNN_SCHEDULER='plateau',
                         DEVICE='cuda', RESUME_CHECKPOINT='last.pth', mask_model=loadable)
        execute('mask_optimizer', namespace)
        _, _, _, _, start, _, history = namespace['setup_mask_training']()
        self.assertEqual(restored, [True])
        self.assertEqual(start + 1, 14)
        self.assertEqual(history, [{'epoch': 13}])


class DatasetStagingTests(unittest.TestCase):
    def test_colab_writes_artifacts_to_drive_and_prepares_locally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'content/drive/MyDrive/Computer_Vision_Tugas_CPMK3'
            project.mkdir(parents=True)

            def sandbox_path(value):
                path = Path(value)
                return root / str(path).lstrip('/') if path.is_absolute() else path

            ns = dict(Path=sandbox_path)
            execute('config', ns)
            ns['RUN_LABEL'] = 'trial'
            execute('paths', ns)
            self.assertEqual(ns['WORK'], project / 'outputs')
            self.assertEqual(ns['REPORTS'], project / 'outputs/reports/all_classes/trial')
            self.assertEqual(ns['PREPARED'], root / 'content/cpmk3_workspace/prepared')
            dataset = sandbox_path(ns['RAW_DATASET'])
            self.assertEqual(dataset, project / 'dataset')
            for source in (dataset, root / 'kaggle/input/another-dataset'):
                for split in ('train', 'valid', 'test'):
                    folder = source / split
                    folder.mkdir(parents=True)
                    (folder / '_annotations.coco.json').write_text(json.dumps({'source': str(source)}))
            ns.update(Path=sandbox_path, read_json=lambda path: json.loads(path.read_text()))
            execute('discover', ns)
            self.assertEqual(ns['RAW'], dataset)
            self.assertEqual(ns['raw_coco']['train']['source'], str(dataset))
            (dataset / 'valid/_annotations.coco.json').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'valid/_annotations.coco.json'):
                execute('discover', ns)

    def test_staging_uses_only_referenced_files_from_the_three_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'drive', root / 'runtime'
            data = {}
            for split in ('train', 'valid', 'test'):
                folder = source / split
                folder.mkdir(parents=True)
                (folder / 'photo.jpg').write_bytes(split.encode())
                (folder / 'not_in_annotations.jpg').write_bytes(b'unused')
                data[split] = {'images': [{'file_name': 'photo.jpg'}]}
            ns = dict(Path=Path, shutil=shutil, IS_COLAB=False, RAW=source,
                      write_json=lambda path, value: path.write_text(json.dumps(value)))
            execute('stage_dataset', ns)
            ns['stage_coco'](source, data, destination)
            for split, annotation in data.items():
                folder = destination / split
                self.assertEqual((folder / 'photo.jpg').read_bytes(), split.encode())
                self.assertEqual(json.loads((folder / '_annotations.coco.json').read_text()), annotation)
                self.assertFalse((folder / 'not_in_annotations.jpg').exists())


class WorkerLoggingTests(unittest.TestCase):
    def test_rapid_metrics_and_traceback_context_survive_throttling(self):
        ns = dict(os=os, sys=sys, subprocess=subprocess, time=time,
                  interrupt_worker=lambda process: process.terminate())
        execute('sequence_process', ns)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            script = folder / 'worker.py'
            expected = ('startup', 'Overall Metrics', 'AP50: 0.31',
                        'Traceback (most recent call last):', '  frame at predict_rfdetr', 'KeyError: 74')
            script.write_text('for line in ' + repr(expected) + ':\n    print(line, flush=True)\n')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = ns['execute_worker'](sys.executable, script, folder / 'config.json', folder)
            self.assertEqual(code, 0)
            disk_log = (folder / 'console_log.txt').read_text()
            for line in expected:
                self.assertIn(line, output.getvalue())
                self.assertIn(line, disk_log)


if __name__ == '__main__':
    unittest.main()
