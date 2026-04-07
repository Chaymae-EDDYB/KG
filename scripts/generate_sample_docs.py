"""Generate sample document files for batch ingestion testing."""
from __future__ import annotations
from pathlib import Path

DOCS = [
    {
        "id": "marie-curie-001",
        "text": (
            "Marie Curie was born in Warsaw, Poland in 1867. She moved to Paris, France "
            "to study at the Sorbonne University. Together with her husband Pierre Curie, "
            "she discovered Polonium and Radium. Marie Curie won the Nobel Prize in Physics "
            "in 1903 and the Nobel Prize in Chemistry in 1911, making her the first person "
            "to win Nobel Prizes in two different sciences. She founded the Curie Institute "
            "in Paris and worked extensively on radioactivity research."
        ),
    },
    {
        "id": "nikola-tesla-001",
        "text": (
            "Nikola Tesla was born in Smiljan, Austria in 1856. He studied at the "
            "Technical University of Graz and later worked for the Continental Edison "
            "Company in Paris, France. Tesla moved to the United States in 1884 and "
            "developed the alternating current electrical system. He worked at his "
            "laboratory in New York City and held over 300 patents. His rivalry with "
            "Thomas Edison became known as the War of Currents. Tesla died in New York "
            "in 1943."
        ),
    },
    {
        "id": "alan-turing-001",
        "text": (
            "Alan Turing was born in London, England in 1912. He studied mathematics "
            "at Cambridge University and later worked at the Government Code and "
            "Cypher School at Bletchley Park during World War II. Turing developed "
            "the Turing Machine concept and the Turing Test for artificial intelligence. "
            "He worked at the National Physical Laboratory and the University of "
            "Manchester on early computer designs."
        ),
    },
    {
        "id": "isaac-newton-001",
        "text": (
            "Isaac Newton was born in Woolsthorpe, England in 1643. He studied at "
            "Trinity College, Cambridge University. Newton developed the laws of motion "
            "and universal gravitation. He published Principia Mathematica in 1687. "
            "Newton served as Warden and then Master of the Royal Mint in London. "
            "He was elected President of the Royal Society in 1703."
        ),
    },
    {
        "id": "darwin-001",
        "text": (
            "Charles Darwin was born in Shrewsbury, England in 1809. He studied at "
            "Edinburgh University and Cambridge University. Darwin embarked on a five-year "
            "voyage aboard HMS Beagle, visiting the Galapagos Islands. He developed the "
            "theory of evolution by natural selection and published On the Origin of Species "
            "in 1859. Darwin lived at Down House in Kent, England."
        ),
    },
    {
        "id": "ada-lovelace-001",
        "text": (
            "Ada Lovelace was born in London, England in 1815. She was the daughter of "
            "Lord Byron. Ada worked with Charles Babbage on the Analytical Engine and "
            "is considered the first computer programmer. She wrote the first algorithm "
            "intended for machine processing. Ada Lovelace studied mathematics under "
            "Augustus De Morgan at University College London."
        ),
    },
    {
        "id": "galileo-001",
        "text": (
            "Galileo Galilei was born in Pisa, Italy in 1564. He studied at the "
            "University of Pisa and later taught at the University of Padua. Galileo "
            "improved the telescope and made key astronomical observations including "
            "the moons of Jupiter. He supported the Copernican heliocentric model and "
            "was tried by the Roman Inquisition. His work Dialogue Concerning the Two "
            "Chief World Systems was published in 1632."
        ),
    },
    {
        "id": "rosalind-franklin-001",
        "text": (
            "Rosalind Franklin was born in London, England in 1920. She studied at "
            "Cambridge University and conducted X-ray crystallography research at "
            "King's College London. Franklin's Photo 51 was crucial to understanding "
            "the structure of DNA. She also worked at Birkbeck College on the tobacco "
            "mosaic virus. James Watson and Francis Crick used her data to build their "
            "DNA model."
        ),
    },
    {
        "id": "hawking-001",
        "text": (
            "Stephen Hawking was born in Oxford, England in 1942. He studied at "
            "University College Oxford and then at Cambridge University. Hawking "
            "became Lucasian Professor of Mathematics at Cambridge. He developed "
            "theories about black holes and Hawking Radiation. His book A Brief "
            "History of Time became a bestseller. Hawking worked with Roger Penrose "
            "on singularity theorems."
        ),
    },
    {
        "id": "feynman-001",
        "text": (
            "Richard Feynman was born in New York City in 1918. He studied at "
            "Massachusetts Institute of Technology and Princeton University. Feynman "
            "worked on the Manhattan Project at Los Alamos Laboratory. He developed "
            "Feynman Diagrams for quantum electrodynamics. Feynman won the Nobel Prize "
            "in Physics in 1965 and taught at the California Institute of Technology "
            "for most of his career."
        ),
    },
]


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for doc in DOCS:
        path = raw_dir / ("%s.txt" % doc["id"])
        path.write_text(doc["text"])
        print("  Created %s" % path)

    print("\nGenerated %d sample documents in %s/" % (len(DOCS), raw_dir))


if __name__ == "__main__":
    main()
