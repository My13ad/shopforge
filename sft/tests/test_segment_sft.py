"""Check segment boundaries and the actual weighted CE gradients."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F

from shopping_grpo.training.sft.dataset import build_supervised_example
from shopping_grpo.training.sft.loss import weighted_linear_ce, segment_trainer_class
from scripts.train_lora_sft import _collate, _torch_dataset


class XmlTokenizer:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = ''
        for message in messages:
            text += '<|im_start|>' + message['role'] + '\n' + (message.get('content') or '')
            for call in message.get('tool_calls') or []:
                function = call['function']
                text += '<tool_call>\n<function=' + function['name'] + '>\n'
                for key, value in function['arguments'].items():
                    text += f'<parameter={key}>\n{value}\n</parameter>\n'
                text += '</function>\n</tool_call>'
            text += '<|im_end|>\n'
        return text + '<|im_start|>assistant\n' if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': [ord(char) for char in text]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index + 1) for index in range(len(text))]
        return result


def action(name, arguments=None, content=''):
    return {'role': 'assistant', 'content': content, 'tool_calls': [
        {'function': {'name': name, 'arguments': arguments or {}}},
    ]}


def test_observation_xml_is_never_supervised_and_eos_is_learned():
    messages = [
        {'role': 'user', 'content': '用户需求'},
        action('select_option', {'value': '红色'}, content='先选红色。'),
        {'role': 'tool', 'content': '<tool_call><function=buy_now></function></tool_call>'},
        action('buy_now'),
    ]
    original = deepcopy(messages)
    example = build_supervised_example(messages, [], XmlTokenizer(), max_length=2000,
                                       loss_mode='option-weighted')
    assert messages == original
    text = ''.join(chr(token) for token in example['input_ids'])
    tool_start = text.index('<|im_start|>tool')
    tool_end = text.index('<|im_end|>', tool_start) + len('<|im_end|>')
    assert all(example['labels'][i] == -100 and example['loss_weights'][i] == 0
               for i in range(tool_start, tool_end))
    for phrase, weight in [('先选红色。', 1), ('<function=select_option>', 2),
                           ('<parameter=value>', 2), ('<function=buy_now>', 2)]:
        index = text.rindex(phrase) if phrase == '<function=buy_now>' else text.index(phrase)
        assert example['loss_weights'][index] == weight
    final_eos = text.rindex('<|im_end|>')
    assert example['labels'][final_eos] != -100
    assert example['loss_weights'][final_eos] == 1


def test_paper_and_option_extension_differ_only_on_option_tokens():
    messages = [action('search_products', {'query': '鞋'}, content='搜索。'),
                {'role': 'tool', 'content': '候选'}, action('select_option', {'value': 'S'}),
                {'role': 'tool', 'content': '已选择'}, action('buy_now')]
    first = build_supervised_example(messages, [], XmlTokenizer(), loss_mode='smartad')
    second = build_supervised_example(messages, [], XmlTokenizer(), loss_mode='option-weighted')
    assert first['labels'] == second['labels'] and first['segment_ids'] == second['segment_ids']
    for left, right, segment in zip(first['loss_weights'], second['loss_weights'], first['segment_ids']):
        assert (left, right) == (1.5, 2) if segment == 3 else left == right


def test_padding_has_zero_weight_and_context_segment():
    examples = [build_supervised_example([action('buy_now')], [], XmlTokenizer(), loss_mode='smartad'),
                build_supervised_example([{'role': 'user', 'content': 'long user text'}, action('buy_now')],
                                         [], XmlTokenizer(), loss_mode='smartad')]
    dataset = _torch_dataset(examples, torch)
    batch = _collate([dataset[0], dataset[1]], 0, torch)
    padding = batch['attention_mask'] == 0
    assert padding.any()
    assert (batch['loss_weights'][padding] == 0).all()
    assert (batch['segment_ids'][padding] == 0).all()
    assert (batch['labels'][padding] == -100).all()


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('frozen_head', [False, True])
def test_loss_and_all_gradients_match_dense_reference(uniform, frozen_head):
    torch.manual_seed(123)
    hidden = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(4, 11, dtype=torch.float64)
    reference_head = deepcopy(head)
    if frozen_head:
        head.requires_grad_(False)
        reference_head.requires_grad_(False)
    labels = torch.tensor([[-100, 2, -100, 5, 3], [-100, 1, 4, -100, -100]])
    weights = labels.ne(-100).double()
    if not uniform:
        weights[0, 1], weights[0, 3], weights[1, 2] = 2, 1.5, 3
    loss, _, _ = weighted_linear_ce(hidden, head, labels, weights, chunk_size=2)
    logits = reference_head(reference_hidden[:, :-1])
    losses = F.cross_entropy(logits.reshape(-1, 11), labels[:, 1:].reshape(-1),
                            ignore_index=-100, reduction='none').reshape(2, 4)
    expected = (losses * weights[:, 1:]).sum() / weights[:, 1:].sum()
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    assert (hidden.grad[:, -1] == 0).all()
    if not frozen_head:
        torch.testing.assert_close(head.weight.grad, reference_head.weight.grad)
        torch.testing.assert_close(head.bias.grad, reference_head.bias.grad)


def test_accumulation_denominator_matches_a_single_padded_batch():
    torch.manual_seed(42)
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(3, 7)
    head.requires_grad_(False)
    labels = torch.tensor([[-100, 2, 4, 5], [-100, 1, -100, -100]])
    weights = torch.tensor([[0., 1.5, 2., 1.], [0., 2., 0., 0.]])
    denominator = weights[:, 1:].sum()
    full, _, _ = weighted_linear_ce(hidden, head, labels, weights)
    accumulated = sum(weighted_linear_ce(other[i:i+1], head, labels[i:i+1], weights[i:i+1],
                                         denominator=denominator)[0] for i in range(2))
    full.backward()
    accumulated.backward()
    torch.testing.assert_close(full, accumulated)
    torch.testing.assert_close(hidden.grad, other.grad)


def test_invalid_mask_weights_are_rejected():
    head = torch.nn.Linear(2, 3)
    with pytest.raises(ValueError, match='zero loss weight'):
        weighted_linear_ce(torch.randn(1, 2, 2), head, torch.tensor([[-100, 1]]),
                           torch.ones(1, 2))


def test_fp32_hidden_states_with_bf16_projection_match_native_projection():
    torch.manual_seed(42)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(8, 17, dtype=torch.bfloat16).requires_grad_(False)
    labels = torch.tensor([[-100, 2, 4, 5]])
    weights = torch.tensor([[0., 1., 1.5, 2.]])
    actual, _, _ = weighted_linear_ce(hidden, head, labels, weights)
    logits = head(reference[:, :-1].to(torch.bfloat16)).float()
    ce = F.cross_entropy(logits.reshape(-1, 17), labels[:, 1:].reshape(-1), reduction='none')
    expected = (ce * weights[:, 1:].reshape(-1)).sum() / weights.sum()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, reference.grad)


class MinimalTrainer:
    def __init__(self):
        self.args = SimpleNamespace(world_size=1, n_gpu=0)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 3)
        self.head = torch.nn.Linear(3, 13)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **kwargs):
        assert kwargs['logits_to_keep'] == 1 and kwargs['use_cache'] is False
        return SimpleNamespace(hidden_states=(self.embedding(input_ids),))


def test_validation_reports_uniform_nll_and_accumulation_counts_weights():
    torch.manual_seed(42)
    trainer = segment_trainer_class(MinimalTrainer, use_liger=False)()
    inputs = {'input_ids': torch.tensor([[2, 3, 5, 6]]),
              'labels': torch.tensor([[-100, 3, 5, 6]]),
              'loss_weights': torch.tensor([[0., 1., 1.5, 2.]]),
              'segment_ids': torch.tensor([[0, 1, 2, 4]])}
    assert trainer._get_num_items_in_batch([inputs, inputs], 'cpu').item() == 9
    model = TinyLM().eval()
    actual = trainer.compute_loss(model, inputs)
    logits = model.head(model.embedding(inputs['input_ids']))[:, :-1]
    expected = F.cross_entropy(logits.reshape(-1, 13), inputs['labels'][:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)
    model.train()
    weighted = trainer.compute_loss(model, inputs)
    assert not torch.isclose(actual, weighted)
