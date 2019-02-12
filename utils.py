import tokenization
import re

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def read_examples(lst_strs):
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(guid=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1

def read_tokenized_examples(lst_strs):
    unique_id = 0
    lst_strs = [[tokenization.convert_to_unicode(w) for w in s] for s in lst_strs]
    for ss in lst_strs:
        text_a = ss
        text_b = None
        try:
            j = ss.index('|||')
            text_a = ss[:j]
            text_b = ss[(j + 1):]
        except ValueError:
            pass
        yield InputExample(guid=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1


def convert_lst_to_features(lst_str, seq_length, tokenizer, is_tokenized=False, mask_cls_sep=False):
    """Loads a data file into a list of `InputBatch`s."""

    examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

    _tokenize = lambda x: tokenizer.mark_unk_tokens(x) if is_tokenized else tokenizer.tokenize(x)

    for (ex_index, example) in enumerate(examples):
        tokens_a = _tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = _tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        input_type_ids = [0] * len(tokens)
        input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            input_type_ids += [1] * (len(tokens_b) + 1)
            input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_len = seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        input_type_ids += [0] * pad_len

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        # print('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
        # print('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
        # print('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
        # print('input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))
        yield InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=input_type_ids)