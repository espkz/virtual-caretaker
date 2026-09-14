"""Create ignored local configuration. Never modifies an existing .env."""
import argparse
import json
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-legacy-key", action="store_true", help="Copy an sk- API key from the local openai.json file without printing it")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print(".env already exists; left unchanged.")
        return
    text = (root / ".env.example").read_text(encoding="utf-8")
    text = text.replace("replace-with-a-long-random-secret", secrets.token_urlsafe(48))
    if args.import_legacy_key:
        values = json.loads((root / "openai.json").read_text(encoding="utf-8"))
        key = next((value for value in values.values() if isinstance(value, str) and value.startswith("sk-")), None)
        if not key or any(c in key for c in "\r\n"):
            raise SystemExit("No usable API key found in openai.json.")
        text = text.replace("OPENAI_API_KEY=", "OPENAI_API_KEY=" + key)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print("Created local .env. Configure OPENAI_API_KEY there if it was not imported. Keep this file private.")


if __name__ == "__main__":
    main()
