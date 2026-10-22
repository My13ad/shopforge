"""Check the model-output boundary before using the real shopping environment."""

import json

import pytest
from types import SimpleNamespace

from shopping_grpo.evaluation.local_inference import chat_eos_token_ids, parse_qwen_completion


def test_search_keeps_plain_string_values_and_comparison_symbols():
    reply = parse_qwen_completion(
        '<tool_call>\n<function=search_products>\n'
        '<parameter=query>\n鞋 <100元 & 透气\n</parameter>\n</function>\n</tool_call>'
    )
    assert reply['content'] == ''
    assert reply['tool_calls'][0]['function']['name'] == 'search_products'
    assert json.loads(reply['tool_calls'][0]['function']['arguments']) == {'query': '鞋 <100元 & 透气'}


def test_no_argument_purchase_is_empty_object():
    reply = parse_qwen_completion('<tool_call><function=buy_now></function></tool_call>')
    assert json.loads(reply['tool_calls'][0]['function']['arguments']) == {}


def test_complete_function_without_outer_wrapper_is_recognized():
    reply = parse_qwen_completion('<function=select_option><parameter=value>紫色</parameter></function>')
    assert json.loads(reply['tool_calls'][0]['function']['arguments']) == {'value': '紫色'}


def test_multiple_calls_remain_visible_for_the_existing_serial_guard():
    reply = parse_qwen_completion(
        '<tool_call><function=view_description></function>'
        '<function=buy_now></function></tool_call>'
    )
    assert [call['function']['name'] for call in reply['tool_calls']] == ['view_description', 'buy_now']


def test_truncated_function_is_not_completed_or_executed():
    text = '<tool_call><function=search_products><parameter=query>运动鞋'
    reply = parse_qwen_completion(text)
    assert not reply['tool_calls']
    assert reply['content'] == text
    assert reply['generation_metadata']['incomplete_function']


def test_unknown_function_and_extra_arguments_are_left_for_schema_guard():
    reply = parse_qwen_completion(
        '<tool_call><function=unknown><parameter=asin>00001</parameter>'
        '<parameter=other>  keep spaces  </parameter></function></tool_call>'
    )
    assert reply['tool_calls'][0]['function']['name'] == 'unknown'
    assert json.loads(reply['tool_calls'][0]['function']['arguments']) == {
        'asin': '00001', 'other': '  keep spaces  ',
    }


def test_reasoning_prefix_is_preserved_in_assistant_content():
    reply = parse_qwen_completion(
        '先搜索明确的品牌。\n<tool_call>\n<function=search_products>'
        '<parameter=query>品牌 鞋</parameter></function></tool_call>'
    )
    assert reply['content'] == '先搜索明确的品牌。'


def test_base_tokenizer_stops_at_chat_boundary_as_well_as_document_end():
    tokenizer = SimpleNamespace(eos_token_id=248044, unk_token_id=None,
                                convert_tokens_to_ids=lambda token: 248046)
    assert chat_eos_token_ids(tokenizer) == [248044, 248046]


def test_missing_chat_boundary_is_rejected_before_evaluation():
    tokenizer = SimpleNamespace(eos_token_id=248044, unk_token_id=None,
                                convert_tokens_to_ids=lambda token: None)
    with pytest.raises(ValueError, match='missing <\\|im_end\\|>'):
        chat_eos_token_ids(tokenizer)
