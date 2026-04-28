from word_rag.entity_extractor import extract_entities, extract_dax_numbers


def test_extract_entities_from_fd_text():
    text = "DAX-7406 использует WMS_IsOversizedItemIM и SalesLine"
    entities = extract_entities(text)
    keys = {(e['type'], e['value']) for e in entities}
    assert ('dax_task', 'DAX-7406') in keys
    assert ('ax_object', 'WMS_IsOversizedItemIM') in keys
    assert ('ax_object', 'SalesLine') in keys


def test_extract_dax_numbers_unique():
    text = "DAX-7406 и dax-7406 и DAX-11367"
    assert extract_dax_numbers(text) == ['DAX-11367', 'DAX-7406']
