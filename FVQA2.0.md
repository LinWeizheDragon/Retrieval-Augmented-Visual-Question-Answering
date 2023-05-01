# FVQA 2.0 Data Download

This is the dataset download page for our investigation published at EACL 2023 Findings:

[FVQA 2.0: Introducing Adversarial Samples into Fact-based Visual Question Answering](https://aclanthology.org/2023.findings-eacl.11) (Lin et al., Findings 2023)

The dataset used in this paper can be downloaded from [here](https://drive.google.com/drive/folders/1dud0hIDMwGiprLS1RgzWbIGJM7pSICyz?usp=share_link)

There are two files under the shared folder. The one with "\_remove_person" removes all questions that have "person" as answers. This is because knowledge related to "person" normally do not form useful questions. The final VQA performance may improve slightly, but the conclusions in the paper still hold.

If our work help your research, please kindly cite the paper:

```
@inproceedings{lin-etal-2023-fvqa,
    title = "{FVQA} 2.0: Introducing Adversarial Samples into Fact-based Visual Question Answering",
    author = "Lin, Weizhe  and
      Wang, Zhilin  and
      Byrne, Bill",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.11",
    pages = "149--157",
    abstract = "The widely used Fact-based Visual Question Answering (FVQA) dataset contains visually-grounded questions that require information retrieval using common sense knowledge graphs to answer. It has been observed that the original dataset is highly imbalanced and concentrated on a small portion of its associated knowledge graph. We introduce FVQA 2.0 which contains adversarial variants of test questions to address this imbalance. We show that systems trained with the original FVQA train sets can be vulnerable to adversarial samples and we demonstrate an augmentation scheme to reduce this vulnerability without human annotations.",
}
```
