import unittest

import pytest

from transformers import pipeline
from transformers.testing_utils import require_tf, require_torch, slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


EXPECTED_FILL_MASK_RESULT = [
    [
        {"sequence": "<s>My name is John</s>", "score": 0.00782308354973793, "token": 610, "token_str": "ĠJohn"},
        {"sequence": "<s>My name is Chris</s>", "score": 0.007475061342120171, "token": 1573, "token_str": "ĠChris"},
    ],
    [
        {"sequence": "<s>The largest city in France is Paris</s>", "score": 0.3185044229030609, "token": 2201},
        {"sequence": "<s>The largest city in France is Lyon</s>", "score": 0.21112334728240967, "token": 12790},
    ],
]

EXPECTED_FILL_MASK_TARGET_RESULT = [
    [
        {
            "sequence": "<s>My name is Patrick</s>",
            "score": 0.004992353264242411,
            "token": 3499,
            "token_str": "ĠPatrick",
        },
        {
            "sequence": "<s>My name is Clara</s>",
            "score": 0.00019297805556561798,
            "token": 13606,
            "token_str": "ĠClara",
        },
    ]
]


class FillMaskPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "fill-mask"
    pipeline_loading_kwargs = {"top_k": 2}
    small_models = ["sshleifer/tiny-distilroberta-base"]  # Models tested without the @slow decorator
    large_models = ["distilroberta-base"]  # Models tested with the @slow decorator
    mandatory_keys = {"sequence", "score", "token"}
    valid_inputs = [
        "My name is <mask>",
        "The largest city in France is <mask>",
    ]
    invalid_inputs = [
        "This is <mask> <mask>"  # More than 1 mask_token in the input is not supported
        "This is"  # No mask_token is not supported
    ]
    expected_check_keys = ["sequence"]

    @require_torch
    def test_torch_topk_deprecation(self):
        # At pipeline initialization only it was not enabled at pipeline
        # call site before
        with pytest.warns(FutureWarning, match=r".*use `top_k`.*"):
            pipeline(task="fill-mask", model=self.small_models[0], topk=1)

    @require_torch
    def test_torch_fill_mask(self):
        valid_inputs = "My name is <mask>"
        nlp = pipeline(task="fill-mask", model=self.small_models[0])
        outputs = nlp(valid_inputs)
        self.assertIsInstance(outputs, list)

        # This passes
        outputs = nlp(valid_inputs, targets=[" Patrick", " Clara"])
        self.assertIsInstance(outputs, list)

        # This used to fail with `cannot mix args and kwargs`
        outputs = nlp(valid_inputs, something=False)
        self.assertIsInstance(outputs, list)

    @require_torch
    def test_torch_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in self.small_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="pt")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_tf
    def test_tf_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in self.small_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_torch
    @slow
    def test_torch_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in self.large_models:
            nlp = pipeline(
                task="fill-mask",
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                top_k=2,
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=EXPECTED_FILL_MASK_RESULT,
                expected_check_keys=["sequence"],
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs[:1],
                mandatory_keys,
                expected_multi_result=EXPECTED_FILL_MASK_TARGET_RESULT,
                expected_check_keys=["sequence"],
                targets=valid_targets,
            )

    @require_tf
    @slow
    def test_tf_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in self.large_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf", topk=2)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=EXPECTED_FILL_MASK_RESULT,
                expected_check_keys=["sequence"],
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs[:1],
                mandatory_keys,
                expected_multi_result=EXPECTED_FILL_MASK_TARGET_RESULT,
                expected_check_keys=["sequence"],
                targets=valid_targets,
            )