from src.chunking import chunk_text


def test_chunk_text_splits_paragraphs():

    text = (
        "ERA5 is ECMWF's fifth-generation atmospheric reanalysis.\n\n"
        "ERA5 model-level parameters are archived in GRIB2 format."
    )

    chunks = chunk_text(text)

    assert len(chunks) == 2

    assert chunks[0] == (
        "ERA5 is ECMWF's fifth-generation atmospheric reanalysis."
    )

    assert chunks[1] == (
        "ERA5 model-level parameters are archived in GRIB2 format."
    )
