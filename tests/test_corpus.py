import unittest
from pathlib import Path

from benchmarker.data.reader import Corpus, qa_strategies
from benchmarker.data.reader.common import DataInstance
from benchmarker.data.document import Doc2d


class TestCorpus(unittest.TestCase):
    def test_docvqa(self) -> None:
        data_path = Path("examples/docvqa")
        docvqa_ocr = "microsoft_cv"

        # `train_strategy` means how to read the train dataset
        # The variable values is set in the dataset, if this parameter is set as `first_item`, the first item in the dataset will be used.
        # In detail, see `benchmarker/data/reader/qa_strategies.py`.
        corpus = Corpus(
            unescape_prefix=False,
            unescape_values=True,
            use_prefix=True,
            prefix_separator=":",
            values_separator="|",
            single_property=True,
            use_none_answers=False,
            case_augmentation=False,
            lowercase_expected=False,
            lowercase_input=False,
            train_strategy=getattr(qa_strategies, "first_item"),
            dev_strategy=getattr(qa_strategies, "concat"),
            test_strategy=getattr(qa_strategies, "concat"),
            augment_tokens_from_file="",
        )
        # set train/dev/test dataset
        # Note: This method have not read the data yet! Only set the dataset.
        corpus.read_benchmark_challenge(directory=data_path, ocr=docvqa_ocr)

        expected_data = [
            DataInstance
            (
                identifier="xnbl0037_1",
                input_prefix="what is the date mentioned in this letter? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="what is the date mentioned in this letter?",
                output="1/8/93",
            ),
            DataInstance
            (
                identifier="xnbl0037_1",
                input_prefix="what is the contact person name mentioned in letter? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="what is the contact person name mentioned in letter?",
                output="P. Carter",
            ),
        ]

        # Return Iterator[DataInstance]
        train_subset = getattr(corpus, "train")
        actual_data: list[DataInstance] = [data_instance for data_instance in train_subset]

        self.assertEqual(len(expected_data), len(actual_data))

        for actual, expected in zip(actual_data, expected_data):
            self.assertEqual(actual.identifier, expected.identifier)
            self.assertEqual(actual.input_prefix, expected.input_prefix)
            self.assertEqual(actual.output_prefix, expected.output_prefix)
            self.assertEqual(actual.output, expected.output)

    def test_pwc(self) -> None:
        data_path = Path("examples/AxCell")
        docvqa_ocr = "tesseract"

        # `train_strategy` means how to read the train dataset
        # The variable values is set in the dataset, if this parameter is set as `concat`, the all items in the dataset will be used with `values_separator`.
        # In detail, see `benchmarker/data/reader/qa_strategies.py`.
        corpus = Corpus(
            unescape_prefix=False,
            unescape_values=True,
            use_prefix=True,
            prefix_separator=":",
            values_separator="|",
            single_property=True,
            use_none_answers=False,
            case_augmentation=False,
            lowercase_expected=False,
            lowercase_input=False,
            train_strategy=getattr(qa_strategies, "concat"),
            dev_strategy=getattr(qa_strategies, "concat"),
            test_strategy=getattr(qa_strategies, "concat"),
            augment_tokens_from_file="",
        )
        # set train/dev/test dataset
        # Note: This method have not read the data yet! Only set the dataset.
        corpus.read_benchmark_challenge(directory=data_path, ocr=docvqa_ocr)

        # As for PWC, the template is used at the prefix.
        # https://github.com/due-benchmark/baselines/blob/2378c02238a04432c7e1401cbe471d57aaf26ff4/benchmarker/data/reader/benchmark_dataset.py#L75
        expected_data = [
            DataInstance
            (
                identifier="1703.10295v3",
                input_prefix="What are the leaderboard_entry values for the task column? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What are the leaderboard_entry values for the task column?",
                output="Object Detection | Object Detection | Object Detection | Object Detection | Object Detection | Object Detection | Object Detection | Object Detection",
            ),
            DataInstance
            (
                identifier="1703.10295v3",
                input_prefix="What are the leaderboard_entry values for the dataset column? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What are the leaderboard_entry values for the dataset column?",
                output="COCO test-dev | COCO test-dev | COCO test-dev | COCO test-dev | COCO test-dev | COCO test-dev | PASCAL VOC 2007 test | PASCAL VOC 2012 test",
            ),
            DataInstance
            (
                identifier="1703.10295v3",
                input_prefix="What are the leaderboard_entry values for the metric column? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What are the leaderboard_entry values for the metric column?",
                output="box AP | AP50 | AP75 | APS | APM | APL | MAP | MAP",
            ),
            DataInstance
            (
                identifier="1703.10295v3",
                input_prefix="What are the leaderboard_entry values for the model column? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What are the leaderboard_entry values for the model column?",
                output="DeNet-101 (wide) | DeNet-101 (wide) | DeNet-101 (wide) | DeNet-101 (wide) | DeNet-101 (wide) | DeNet-101 (wide) | DeNet-101 (skip) | DeNet-101",
            ),
            DataInstance
            (
                identifier="1703.10295v3",
                input_prefix="What are the leaderboard_entry values for the value column? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What are the leaderboard_entry values for the value column?",
                output="0.338 | 0.534 | 0.361 | 0.123 | 0.361 | 0.508 | 0.771 | 0.739",
            ),
        ]

        # Return Iterator[DataInstance]
        train_subset = getattr(corpus, "train")
        actual_data: list[DataInstance] = [data_instance for data_instance in train_subset]

        self.assertEqual(len(expected_data), len(actual_data))

        for actual, expected in zip(actual_data, expected_data):
            self.assertEqual(actual.identifier, expected.identifier)
            self.assertEqual(actual.input_prefix, expected.input_prefix)
            self.assertEqual(actual.output_prefix, expected.output_prefix)
            self.assertEqual(actual.output, expected.output)

    def test_deepform(self) -> None:
        data_path = Path("examples/DeepForm")
        docvqa_ocr = "microsoft_cv"
        # docvqa_ocr = "tesseract"  # As for deepform, the OCR is done by Microsoft CV and tesseract.

        # `train_strategy` means how to read the train dataset
        # The variable values is set in the dataset, if this parameter is set as `first_item`, the first item in the dataset will be used.
        # In detail, see `benchmarker/data/reader/qa_strategies.py`.
        corpus = Corpus(
            unescape_prefix=False,
            unescape_values=True,
            use_prefix=True,
            prefix_separator=":",
            values_separator="|",
            single_property=True,
            use_none_answers=False,
            case_augmentation=False,
            lowercase_expected=False,
            lowercase_input=False,
            train_strategy=getattr(qa_strategies, "first_item"),
            dev_strategy=getattr(qa_strategies, "concat"),
            test_strategy=getattr(qa_strategies, "concat"),
            augment_tokens_from_file="",
        )
        # set train/dev/test dataset
        # Note: This method have not read the data yet! Only set the dataset.
        corpus.read_benchmark_challenge(directory=data_path, ocr=docvqa_ocr)

        expected_data = [
            DataInstance
            (
                identifier="3515690b-0081-10b9-1077-26ea74749d49",
                input_prefix="gross_amount : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="gross_amount",
                output="63705.00",
            ),
            DataInstance
            (
                identifier="3515690b-0081-10b9-1077-26ea74749d49",
                input_prefix="advertiser : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="advertiser",
                output="MIKEBLOOMBERG2020INC-D",
            ),
            DataInstance
            (
                identifier="3515690b-0081-10b9-1077-26ea74749d49",
                input_prefix="contract_num : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="contract_num",
                output="1339936",
            ),
            DataInstance
            (
                identifier="3515690b-0081-10b9-1077-26ea74749d49",
                input_prefix="flight_from : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="flight_from",
                output="02/01/20",
            ),
            DataInstance
            (
                identifier="3515690b-0081-10b9-1077-26ea74749d49",
                input_prefix="flight_to : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="flight_to",
                output="02/12/20",
            ),
        ]

        # Return Iterator[DataInstance]
        train_subset = getattr(corpus, "train")
        actual_data: list[DataInstance] = [data_instance for data_instance in train_subset]

        self.assertEqual(len(expected_data), len(actual_data))

        for actual, expected in zip(actual_data, expected_data):
            self.assertEqual(actual.identifier, expected.identifier)
            self.assertEqual(actual.input_prefix, expected.input_prefix)
            self.assertEqual(actual.output_prefix, expected.output_prefix)
            self.assertEqual(actual.output, expected.output)

    def test_infographics(self) -> None:
        data_path = Path("examples/infographics_vqa")
        docvqa_ocr = "microsoft_cv"

        # `train_strategy` means how to read the train dataset
        # The variable values is set in the dataset, if this parameter is set as `first_item`, the first item in the dataset will be used.
        # In detail, see `benchmarker/data/reader/qa_strategies.py`.
        corpus = Corpus(
            unescape_prefix=False,
            unescape_values=True,
            use_prefix=True,
            prefix_separator=":",
            values_separator="|",
            single_property=True,
            use_none_answers=False,
            case_augmentation=False,
            lowercase_expected=False,
            lowercase_input=False,
            train_strategy=getattr(qa_strategies, "first_item"),
            dev_strategy=getattr(qa_strategies, "concat"),
            test_strategy=getattr(qa_strategies, "concat"),
            augment_tokens_from_file="",
        )
        # set train/dev/test dataset
        # Note: This method have not read the data yet! Only set the dataset.
        corpus.read_benchmark_challenge(directory=data_path, ocr=docvqa_ocr)

        expected_data = [
            DataInstance
            (
                identifier="20471",
                input_prefix="Which type of fonts offer better readability in printed works? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which type of fonts offer better readability in printed works?",
                output="serif fonts",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Which fonts are suited for the web? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which fonts are suited for the web?",
                output="sans serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Which medium has the highest resolution? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which medium has the highest resolution?",
                output="print",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Based on this, how many dots are there in an inch, on a computer screen : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Based on this, how many dots are there in an inch, on a computer screen",
                output="100 dpi",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Based on this how many dots are there in an inch, in a printed medium? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Based on this how many dots are there in an inch, in a printed medium?",
                output="1000",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="What is sans serif fonts reply to serif font, when the serif font says \"you are unreadable in print\"? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What is sans serif fonts reply to serif font, when the serif font says \"you are unreadable in print\"?",
                output="you are unreadable on screen!",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Which fonts survive reproduction and smearing : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which fonts survive reproduction and smearing",
                output="sans serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Which font has simpler letter shpes? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which font has simpler letter shpes?",
                output="sans serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="In which type of font is the word \"serif\" written at the top? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="In which type of font is the word \"serif\" written at the top? : ",
                output="serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="What is the name of the serif font introduced in 1932? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="What is the name of the serif font introduced in 1932? : ",
                output="times roman",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Grotesque is an example of serif or sans? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Grotesque is an example of serif or sans?",
                output="sans",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="For headings on web  which type of font is mostly used? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="For headings on web  which type of font is mostly used?",
                output="sans serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="For body text on web which type of font is mostly used? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="For body text on web which type of font is mostly used?",
                output="sans serif",
            ),
            DataInstance
            (
                identifier="20471",
                input_prefix="Which is the most popular serif font used for web? : ",
                document_2d=Doc2d(tokens=["fake tokens"], seg_data={"tokens": "fake bbox data"}),
                output_prefix="Which is the most popular serif font used for web?",
                output="georgia",
            ),
        ]

        # Return Iterator[DataInstance]
        train_subset = getattr(corpus, "train")
        actual_data: list[DataInstance] = [data_instance for data_instance in train_subset]

        for actual in actual_data:
            print(actual)
        exit()

        self.assertEqual(len(expected_data), len(actual_data))

        for actual, expected in zip(actual_data, expected_data):
            self.assertEqual(actual.identifier, expected.identifier)
            self.assertEqual(actual.input_prefix, expected.input_prefix)
            self.assertEqual(actual.output_prefix, expected.output_prefix)
            self.assertEqual(actual.output, expected.output)

if __name__ == "__main__":
    unittest.main()
