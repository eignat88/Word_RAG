from word_rag.docx_parser import extract_fd_number


def test_extract_fd_number():
    assert extract_fd_number("spec_DAX-7407_v2.docx") == "DAX-7407"
    assert extract_fd_number("without_fd.docx") is None
