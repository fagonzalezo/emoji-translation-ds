import asyncio
import json
import os
import logging
import argparse # Added for command-line arguments
from typing import List, Dict, Optional
from dotenv import load_dotenv # Load environment variables from .env file

import google.generativeai as genai
# import aiofiles # Removed for synchronous file operations

# --- Configuration Constants (Defaults for CLI args) ---
DEFAULT_INPUT_FILE_PATH = "input_sentences.txt"
DEFAULT_OUTPUT_FILE_PATH = "emoji_translations_batched.json"
DEFAULT_MAX_CONCURRENT_REQUESTS = 3
DEFAULT_SENTENCES_PER_BATCH = 10
DEFAULT_MAX_EMOJIS_PER_SENTENCE = 5
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_SENTENCES_TO_PROCESS = -1 # -1 means no limit / process all
DEFAULT_MIN_WORDS_PER_SENTENCE = 6 # New: -1 means no minimum word count limit
DEFAULT_MAX_WORDS_PER_SENTENCE = 8 # New: -1 means no maximum word count limit


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emoji_translator.log"), # Log to a file
        logging.StreamHandler() # Log to console
    ]
)
logger = logging.getLogger(__name__)

# --- API Key Setup ---
# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY) # type: ignore
else:
    logger.error("GOOGLE_API_KEY not found. Please set it as an environment variable or in a .env file.")
    # Script will exit early if no API key in main guard

# --- Core Translation Function for Batches ---
async def translate_sentences_batch_to_emojis(
    batch_of_sentences: List[str],
    model: genai.GenerativeModel, # type: ignore
    semaphore: asyncio.Semaphore,
    max_emojis: int,
    request_timeout: int
) -> List[Optional[str]]:
    """
    Translates a batch of English sentences to sequences of emojis using the Gemini API.

    Args:
        batch_of_sentences: A list of English sentences to translate.
        model: The initialized Gemini GenerativeModel.
        semaphore: An asyncio.Semaphore to limit concurrent API calls.
        max_emojis: The maximum number of emojis to request per sentence.
        request_timeout: Timeout for the API request.

    Returns:
        A list of emoji strings (or None for failed translations),
        corresponding to each sentence in the batch.
    """
    if not batch_of_sentences:
        return []

    prompt = (
        f"You will be given a JSON array of English sentences.\n"
        f"For each sentence in the array, generate a concise sequence of 1 to {max_emojis} relevant emojis "
        f"that best represents its core idea or sentiment.\n"
        f"Return your response as a single JSON array of strings. Each string in the output array "
        f"should be the emoji sequence for the corresponding sentence in the input array, maintaining the original order.\n"
        f"If you cannot generate emojis for a specific sentence, return `null` (JSON null, not the string 'null') for that position in the output array.\n\n"
        f"Example Input:\n"
        f"[\n"
        f"  \"I am very happy today!\",\n"
        f"  \"This is a sad story.\",\n"
        f"  \"Let's go for a walk in the park.\"\n"
        f"]\n\n"
        f"Example Output (ensure this is valid JSON, with null for failures):\n"
        f"[\n"
        f"  \"😄🎉☀️\",\n"
        f"  \"😢💔📖\",\n"
        f"  null\n"
        f"]\n\n"
        f"Input Sentences (JSON Array):\n"
        f"{json.dumps(batch_of_sentences, ensure_ascii=False)}"
    )

    logger.debug(f"Attempting to translate batch of {len(batch_of_sentences)} sentences. First sentence: '{batch_of_sentences[0]}'")

    async with semaphore: # Acquire a spot from the semaphore
        try:
            response = await model.generate_content_async(
                prompt,
                request_options={'timeout': request_timeout}
            )

            if response and response.text:
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                
                try:
                    emoji_results = json.loads(raw_text)
                    if isinstance(emoji_results, list) and len(emoji_results) == len(batch_of_sentences):
                        validated_results = []
                        for i, res in enumerate(emoji_results):
                            if res is None:
                                validated_results.append(None)
                                logger.info(f"Sentence '{batch_of_sentences[i]}' explicitly marked as untranslatable by API.")
                            elif isinstance(res, str) and any(char > '\u2300' for char in res):
                                validated_results.append(res)
                            elif isinstance(res, str):
                                logger.warning(f"Response for sentence '{batch_of_sentences[i]}' in batch does not appear to be emojis: '{res}'. Treating as failed.")
                                validated_results.append(None)
                            else:
                                logger.warning(f"Unexpected type for sentence '{batch_of_sentences[i]}' in batch response: {type(res)}. Original: '{res}'. Treating as failed.")
                                validated_results.append(None)
                        logger.info(f"Successfully translated batch of {len(validated_results)} length. First result: '{validated_results[0] if validated_results else 'N/A'}'")
                        return validated_results
                    else:
                        logger.warning(
                            f"Mismatched response for batch. Expected {len(batch_of_sentences)} items, got {len(emoji_results) if isinstance(emoji_results, list) else 'not a list'}. Raw: {raw_text}"
                        )
                        return [None] * len(batch_of_sentences)
                except json.JSONDecodeError as je:
                    logger.error(f"JSONDecodeError parsing API response for batch. Error: {je}. Raw response: '{raw_text}'")
                    return [None] * len(batch_of_sentences)
            else:
                logger.warning(f"No valid text response for batch starting with: {batch_of_sentences[0]}")
                return [None] * len(batch_of_sentences)
        except Exception as e:
            logger.error(f"Error translating batch starting with '{batch_of_sentences[0]}': {e}")
            if "API key not valid" in str(e):
                logger.error("Critical: Gemini API key is invalid. Please check your GOOGLE_API_KEY.")
            elif "DeadlineExceeded" in str(e) or "timeout" in str(e).lower():
                logger.error(f"Timeout during API call for batch starting with: {batch_of_sentences[0]}")
            return [None] * len(batch_of_sentences)

# --- Main Processing Function ---
async def process_sentences_from_file(
    input_filepath: str,
    output_filepath: str,
    concurrency_limit: int,
    sentences_per_batch_count: int,
    max_emojis_count: int,
    request_timeout_val: int,
    max_sentences_to_process: int,
    min_words_per_sentence: int, # New argument
    max_words_per_sentence: int,  # New argument
    input_language: str = 'english'
) -> None:
    """
    Reads sentences from a file, translates them in batches, and writes results to a JSON file.
    Uses parameters passed from CLI or defaults.
    Uses synchronous file I/O.
    Limits processing based on max_sentences_to_process, min_words, and max_words.
    """
    if not API_KEY: 
        logger.error("Cannot proceed without a valid GOOGLE_API_KEY.")
        return

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # type: ignore
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return

    all_sentences: List[str] = []
    skipped_by_word_count = 0
    total_lines_read = 0

    try:
        with open(input_filepath, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                total_lines_read += 1

                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # Select sentence based on language parameter
                parts = stripped_line.split('\t')
                if input_language == 'english':
                    sentence = parts[0] if len(parts) > 0 else ''
                elif input_language == 'spanish':
                    sentence = parts[1] if len(parts) > 1 else ''
                else:
                    logger.error(f"Invalid input_language: {input_language}. Must be 'english' or 'spanish'.")
                    return

                if not sentence:
                    continue

                # Word count filtering
                word_count = len(sentence.split())
                if min_words_per_sentence > 0 and word_count < min_words_per_sentence:
                    skipped_by_word_count += 1
                    continue
                if max_words_per_sentence > 0 and word_count > max_words_per_sentence:
                    skipped_by_word_count += 1
                    continue

                all_sentences.append(sentence)

                # Apply max_sentences_to_process limit (counts sentences kept after filtering)
                if max_sentences_to_process > 0 and len(all_sentences) >= max_sentences_to_process:
                    logger.info(f"Reached maximum sentences to process: {max_sentences_to_process}.")
                    break

        logger.info(f"Read {total_lines_read} lines from '{input_filepath}'.")
        if skipped_by_word_count > 0:
            logger.info(f"Skipped {skipped_by_word_count} sentences due to word count constraints (min: {min_words_per_sentence}, max: {max_words_per_sentence}).")
        logger.info(f"Collected {len(all_sentences)} sentences for processing.")
        if max_sentences_to_process > 0 and total_lines_read > max_sentences_to_process :
             logger.info(f"(Input reading was limited to {max_sentences_to_process} lines by --max-sentences)")


    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filepath}")
        return
    except Exception as e:
        logger.error(f"Error reading input file '{input_filepath}': {e}")
        return

    if not all_sentences:
        logger.info("No sentences to process after filtering.")
        return


    # Check for duplicate sentences
    unique_sentences = set(all_sentences)
    num_duplicates = len(all_sentences) - len(unique_sentences)
    if num_duplicates > 0:
        logger.warning(f"Detected {num_duplicates} duplicated input sentences out of {len(all_sentences)}. Only the first occurrence will be used in the output.")
        print(all_sentences)
    else:
        logger.info("No duplicated input sentences detected.")

    sentence_batches: List[List[str]] = [
        all_sentences[i:i + sentences_per_batch_count]
        for i in range(0, len(all_sentences), sentences_per_batch_count)
    ]
    logger.info(f"Created {len(sentence_batches)} batches with up to {sentences_per_batch_count} sentences each.")

    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = [
        translate_sentences_batch_to_emojis(
            batch, model, semaphore, max_emojis_count, request_timeout_val
        )
        for batch in sentence_batches
    ]

    logger.info(f"Starting translation for {len(sentence_batches)} batches ({len(all_sentences)} sentences) with concurrency limit {concurrency_limit}...")
    
    results_from_batches: List[List[Optional[str]]] = await asyncio.gather(*tasks)
    
    all_emoji_results: List[Optional[str]] = []
    for i, batch_result in enumerate(results_from_batches):
        all_emoji_results.extend(batch_result)
        current_batch_start_index = i * sentences_per_batch_count
        processed_count_in_script = min(current_batch_start_index + len(batch_result), len(all_sentences))
        logger.info(f"Processed batch {i+1}/{len(sentence_batches)}. Total sentences processed so far: {processed_count_in_script}/{len(all_sentences)}")


    final_output_data: Dict[str, Optional[str]] = {}
    successful_translations = 0
    failed_translations = 0
    
    if len(all_emoji_results) < len(all_sentences):
        logger.warning(f"Result count ({len(all_emoji_results)}) is less than processed sentence count ({len(all_sentences)}). This might indicate partial batch failures. Padding with None.")
        all_emoji_results.extend([None] * (len(all_sentences) - len(all_emoji_results)))
    elif len(all_emoji_results) > len(all_sentences):
        logger.warning(f"Result count ({len(all_emoji_results)}) is greater than processed sentence count ({len(all_sentences)}). Truncating results.")
        all_emoji_results = all_emoji_results[:len(all_sentences)]


    for original_sentence, emoji_translation in zip(all_sentences, all_emoji_results):
        final_output_data[original_sentence] = emoji_translation
        if emoji_translation:
            successful_translations += 1
        else:
            failed_translations += 1
    
    logger.info(f"Translation complete. Total sentences processed: {len(all_sentences)}. Successful: {successful_translations}, Failed: {failed_translations}.")
    logger.info(f"Final output data size: {len(final_output_data)} items.")
    try:
        with open(output_filepath, mode="w", encoding="utf-8") as f:
            f.write(json.dumps(final_output_data, indent=4, ensure_ascii=False))
        logger.info(f"Successfully wrote translations to '{output_filepath}'.")
    except Exception as e:
        logger.error(f"Error writing output file '{output_filepath}': {e}")

# --- Main Execution ---
async def main(args):
    """Main async function to run the script with parsed arguments."""
    if not API_KEY:
        print("Critical Error: GOOGLE_API_KEY environment variable not set. Exiting.")
        logger.critical("GOOGLE_API_KEY not set. Aborting main execution.")
        return

    logger.info(f"Script started with arguments: {args}")

    await process_sentences_from_file(
        input_filepath=args.input_file,
        output_filepath=args.output_file,
        concurrency_limit=args.concurrency,
        sentences_per_batch_count=args.batch_size,
        max_emojis_count=args.max_emojis,
        request_timeout_val=args.timeout,
        max_sentences_to_process=args.max_sentences,
        min_words_per_sentence=args.min_words, # Pass new argument
        max_words_per_sentence=args.max_words,  # Pass new argument
        input_language=args.input_language
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate sentences to emojis using Gemini API.")
    parser.add_argument(
        "-i", "--input-file",
        type=str,
        required=True,
        help="Path to the input text file (one sentence per line). (Required)"
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        required=True,
        help="Path to the output JSON file. (Required)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help=f"Maximum number of concurrent batch API requests. Default: {DEFAULT_MAX_CONCURRENT_REQUESTS}"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=DEFAULT_SENTENCES_PER_BATCH,
        help=f"Number of sentences to send in one API call (batch size). Default: {DEFAULT_SENTENCES_PER_BATCH}"
    )
    parser.add_argument(
        "-e", "--max-emojis",
        type=int,
        default=DEFAULT_MAX_EMOJIS_PER_SENTENCE,
        help=f"Maximum number of emojis to request per sentence. Default: {DEFAULT_MAX_EMOJIS_PER_SENTENCE}"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"Timeout for API requests in seconds. Default: {DEFAULT_REQUEST_TIMEOUT}"
    )
    parser.add_argument(
        "-n", "--max-sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES_TO_PROCESS,
        help=f"Maximum number of sentences to read from the input file. Default: {DEFAULT_MAX_SENTENCES_TO_PROCESS} (process all)."
    )
    parser.add_argument(
        "--min-words", # New argument
        type=int,
        default=DEFAULT_MIN_WORDS_PER_SENTENCE,
        help=f"Minimum number of words a sentence must have to be processed. Default: {DEFAULT_MIN_WORDS_PER_SENTENCE} (no limit)."
    )
    parser.add_argument(
        "--max-words", # New argument
        type=int,
        default=DEFAULT_MAX_WORDS_PER_SENTENCE,
        help=f"Maximum number of words a sentence can have to be processed. Default: {DEFAULT_MAX_WORDS_PER_SENTENCE} (no limit)."
    )
    parser.add_argument(
        "-l", "--input-language",
        type=str,
        choices=["english", "spanish"],
        default="english",
        help="Language of the input sentences: 'english' (first column) or 'spanish' (second column). Default: english."
    )



    parsed_args = parser.parse_args()

    if not API_KEY: 
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it and try again. You can create a .env file with GOOGLE_API_KEY='YOUR_KEY'")
        logger.critical("GOOGLE_API_KEY not set at script startup. Exiting.")
    else:
        logger.info("Starting emoji translation script (batched, CLI args, sync I/O)...")
        asyncio.run(main(parsed_args))
        logger.info("Emoji translation script (batched, CLI args, sync I/O) finished.")
