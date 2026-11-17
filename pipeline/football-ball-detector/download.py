"""Fetch pinned public detector artifacts into ignored local storage."""
import urllib.request

import bootstrap as B

REVISION = "305936007fe7d19ea528d73d08ccd7e70d088adf"
FILES = [
    ("Model/weights/best.pt", "football_best.pt"),
    ("Model/args.yaml", "training_args.yaml"),
]


def main():
    B.LOCAL.mkdir(parents=True, exist_ok=True)
    for remote, name in FILES:
        path = B.LOCAL / name
        if not path.exists():
            url = f"https://huggingface.co/Adit-jain/soccana/resolve/{REVISION}/{remote}"
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as handle:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    assert total < 16 * 1024 * 1024
                    handle.write(chunk)
            temporary.replace(path)
        print(name, flush=True)


if __name__ == "__main__":
    main()
