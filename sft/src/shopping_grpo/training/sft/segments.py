"""Segment only supervised assistant tokens in the actual Qwen chat rendering."""

import re


SEGMENT_NAMES = {0: 'context', 1: 'reason', 2: 'action', 3: 'option', 4: 'final'}
MODE_WEIGHTS = {
    'uniform': (0.0, 1.0, 1.0, 1.0, 1.0),
    'smartad': (0.0, 1.0, 1.5, 1.5, 2.0),
    'option-weighted': (0.0, 1.0, 1.5, 2.0, 2.0),
}
_CALL = re.compile(r'<tool_call>\s*<function=([^>]+)>.*?</function>\s*</tool_call>', re.DOTALL)


def build_segment_weights(text, input_ids, labels, assistant_ranges, tokenizer, mode):
    """Use assistant token ranges first, then match their authoritative tool calls.

    A token overlapping a call boundary belongs to that action. Assistant EOS and
    other supervised tokens outside calls remain reason tokens (weight 1).
    Observation text can never become supervision, even if it contains XML.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != list(input_ids):
        raise ValueError('Offset tokenization differs from the supervised rendering')
    offsets = encoded['offset_mapping']
    segments = [0 if label == -100 else 1 for label in labels]
    for start, end, message in assistant_ranges:
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        calls = list(_CALL.finditer(text, char_start, char_end))
        names = [call['function']['name'] for call in message.get('tool_calls') or []]
        if [call.group(1) for call in calls] != names:
            raise ValueError('Rendered Qwen calls do not match the assistant message')
        for call in calls:
            segment = {'select_option': 3, 'buy_now': 4}.get(call.group(1), 2)
            for index in range(start, end):
                left, right = offsets[index]
                if labels[index] != -100 and left < call.end() and right > call.start():
                    segments[index] = max(segments[index], segment)
    return [MODE_WEIGHTS[mode][segment] for segment in segments], segments


def supervision_counts(examples):
    counts = {name: 0 for name in SEGMENT_NAMES.values()}
    for example in examples:
        for segment in example.get('segment_ids', []):
            counts[SEGMENT_NAMES[segment]] += 1
    return counts
