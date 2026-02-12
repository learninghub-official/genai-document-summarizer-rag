from transformers import pipeline

def summarize_compliance_text(text):
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    summary = summarizer(
        text,
        max_length=180,
        min_length=80,
        do_sample=False
    )

    return summary[0]["summary_text"]


if __name__ == "__main__":
    sample_text = """
    All financial institutions must comply with regulatory requirements.
    Failure to comply may result in penalties and reputational damage.
    Regular audits and reporting are mandatory.
    """

    output = summarize_compliance_text(sample_text)
    print(output)
