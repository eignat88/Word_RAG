from word_rag.filtering import should_skip_chunk


def test_skip_bad_text_variants():
    assert should_skip_chunk("Нет.", "Отчеты и выходные формы", min_chars=100)
    assert should_skip_chunk("-", "Алгоритмы", min_chars=100)


def test_skip_too_short_text():
    assert should_skip_chunk("короткий текст", "Алгоритмы", min_chars=100)


def test_do_not_skip_meaningful_algorithm_text():
    text = "Алгоритм проверки oversized item выполняется по признакам номенклатуры и габаритов с проверкой типа IM" * 2
    assert not should_skip_chunk(text, "Алгоритмы", min_chars=100)
