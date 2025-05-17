
# Emoji Translation Dataset & Scripts

This project provides scripts and resources for translating English or Spanish sentences into emoji sequences using Google Gemini (Generative AI) models.

## Requirements

- Python 3.8+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Set your Google API key as an environment variable or in a `.env` file:
  ```sh
  export GOOGLE_GENAI_API_KEY=your-api-key-here
  # or create a .env file with:
  # GOOGLE_GENAI_API_KEY=your-api-key-here
  ```

## Main Script: `emoji_translator.py`

Translates sentences from a text file to emoji sequences using Gemini.

### Usage

```sh
python emoji_translator.py -i INPUT_FILE -o OUTPUT_FILE -l LANGUAGE [options]
```

#### Required arguments:
- `-i`, `--input-file`   : Path to the input text file (tab-separated columns, one line per sentence pair)
- `-o`, `--output-file`  : Path to the output JSON file
- `-l`, `--input-language`: `english` (first column) or `spanish` (second column)

#### Optional arguments:
- `-c`, `--concurrency`   : Max concurrent API requests (default: 3)
- `-b`, `--batch-size`    : Sentences per API call (default: 10)
- `-e`, `--max-emojis`    : Max emojis per sentence (default: 5)
- `-t`, `--timeout`       : API request timeout in seconds (default: 60)
- `-n`, `--max-sentences` : Max sentences to process (default: all)
- `--min-words`           : Minimum words per sentence (default: 6)
- `--max-words`           : Maximum words per sentence (default: 8)

### Example

```sh
python emoji_translator.py -i eng2emoji.txt -o emoji_translations.json -l english
```

## Input File Format

- Each line should have at least one sentence. If both English and Spanish are present, they should be tab-separated:
  ```
  Hello, how are you?	Hola, ¿cómo estás?
  I love pizza!	¡Me encanta la pizza!
  ```

## Output

- The output is a JSON file mapping each input sentence to its emoji translation (or null if translation failed).

## Logging

- Logs are written to `emoji_translator.log` and also shown in the console.

## License

See [LICENSE](LICENSE).
