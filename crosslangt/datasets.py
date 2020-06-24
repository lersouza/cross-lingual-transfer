import os

from lxml import etree


TSV_MNLI_HEADER_FIELDS = [
    'index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse',
    'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
    'sentence1', 'sentence2', 'label1',	'label2', 'label3', 'label4',
    'label5', 'gold_label'
]

MNLI_CLASSES = ['neutral', 'entailment', 'contradiction']


def convert_assin_to_tsv(assin_base_path: str,
                         assin_train_file: str = 'assin2-train.xml',
                         assin_eval_file: str = 'assin2-eval.xml',
                         assin_test_file: str = 'assin2-test.xml'):
    """
    Converts the ASSIN Dataset files (XML-Based)
    into TSV files, compatible with MNLI TSV Format.
    """
    prepare_assin_tsv(
        os.path.join(assin_base_path, assin_train_file),
        os.path.join(assin_base_path, f'{assin_train_file}.tsv'),
        MNLI_CLASSES)

    prepare_assin_tsv(
        os.path.join(assin_base_path, assin_train_file),
        os.path.join(assin_base_path, f'{assin_eval_file}.tsv'),
        MNLI_CLASSES)

    prepare_assin_tsv(
        os.path.join(assin_base_path, assin_train_file),
        os.path.join(assin_base_path, f'{assin_test_file}.tsv'),
        MNLI_CLASSES)


def prepare_assin_tsv(xml_path, target_path, MNLI_CLASSES):
    with open(xml_path, 'r') as assin_xml:
        xml_file_contents = assin_xml.read()

    xml_file_contents = xml_file_contents.encode('utf-8')
    root = etree.fromstring(xml_file_contents)

    with open(target_path, 'w+') as assin_tsv:
        index_pos, pairid_pos = TSV_MNLI_HEADER_FIELDS.index('index'), \
                                TSV_MNLI_HEADER_FIELDS.index('pairID')

        s1_pos, s2_pos = \
            TSV_MNLI_HEADER_FIELDS.index('sentence1'), \
            TSV_MNLI_HEADER_FIELDS.index('sentence2')

        labels_start_pos = TSV_MNLI_HEADER_FIELDS.index('label1')

        assin_tsv.write('\t'.join(TSV_MNLI_HEADER_FIELDS))
        assin_tsv.write('\n')

        for i, pair in enumerate(root):
            entailment = pair.attrib['entailment'].lower()
            pairID = pair.attrib['id']
            sentence1 = sentence2 = ''

            # Map None to Neutral
            entailment = 'neutral' if entailment == 'none' else entailment

            if entailment not in MNLI_CLASSES:
                continue  # Ignoring disaligned labels

            for child in pair:
                if child.tag == 't':
                    sentence1 = child.text
                else:
                    sentence2 = child.text

            example = [''] * len(TSV_MNLI_HEADER_FIELDS)
            example[index_pos] = str(i)
            example[pairid_pos] = pairID
            example[s1_pos] = sentence1
            example[s2_pos] = sentence2
            example[labels_start_pos:] = [entailment] * 6

            assin_tsv.write('\t'.join(example))
            assin_tsv.write('\n')


def convert_milkqa_to_json(milkqa_path: str):
    """
    Converts MilkQA Dataset files into JSON format,
    compatible with SQuAD v1.1
    """
    pass
