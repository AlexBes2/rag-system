from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


def get_token_encoder(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoder = get_token_encoder(encoding_name)
    return len(encoder.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    encoder = get_token_encoder(encoding_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoder.encode(x)),
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_document_with_metadata(
    text: str,
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[dict]:
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )

    result = []
    for index, chunk in enumerate(chunks):
        result.append(
            {
                "chunk_id": index,
                "source": source_name,
                "text": chunk,
                "tokens": count_tokens(chunk, encoding_name=encoding_name),
            }
        )

    return result