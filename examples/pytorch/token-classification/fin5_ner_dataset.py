# borrowed from https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py
import os

import datasets


logger = datasets.logging.get_logger(__name__)

_TRAINING_FILE="https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/train/FIN5.txt"
_TEST_FILE="https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/test/FIN3.txt"


class Fin5Config(datasets.BuilderConfig):
    """BuilderConfig for Fin5"""

    def __init__(self, **kwargs):
        """BuilderConfig for Fin5.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Fin5Config, self).__init__(**kwargs)


class Fin5(datasets.GeneratorBasedBuilder):
    """Fin5 dataset."""

    BUILDER_CONFIGS = [
        Fin5Config(name="fin5", version=datasets.Version("1.0.0"), description="Fin5 dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                '"',
                                "''",
                                "#",
                                "$",
                                "(",
                                ")",
                                ",",
                                ".",
                                ":",
                                "``",
                                "-NONE-",
                                "CC",
                                "CD",
                                "DT",
                                "EX",
                                "FW",
                                "IN",
                                "JJ",
                                "JJR",
                                "JJS",
                                "LS",
                                "MD",
                                "NN",
                                "NNP",
                                "NNPS",
                                "NNS",
                                "NN|SYM",
                                "PDT",
                                "POS",
                                "PRP",
                                "PRP$",
                                "RB",
                                "RBR",
                                "RBS",
                                "RP",
                                "SYM",
                                "TO",
                                "UH",
                                "VB",
                                "VBD",
                                "VBG",
                                "VBN",
                                "VBP",
                                "VBZ",
                                "WDT",
                                "WP",
                                "WP$",
                                "WRB",
                            ]
                        )
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                                "B-MISC",
                                "I-MISC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = {
            "train": dl_manager.download(_TRAINING_FILE),
            "dev": dl_manager.download(_TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            pos_tags = []
            chunk_tags = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                        ner_tags = []
                else:
                    # fin5 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1])
                    ner_tags.append(splits[3].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
            }
