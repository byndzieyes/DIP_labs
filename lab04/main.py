from pathlib import Path
from uuid import uuid4

from gtts import gTTS
from playsound import playsound


AUDIO_DIR = Path(__file__).resolve().parent / "audio"
RESPONSES = {
    "hello": "Hello! How can I help you?",
    "how are you": "I'm doing well, thank you.",
    "what can you do": "I respond to text commands and speak my answers.",
    "goodbye": "Goodbye! See you soon.",
}
EXIT_COMMAND = "goodbye"
UNKNOWN_RESPONSE = "I don't know that command. Please try again."


def say_message(message: str) -> Path:
    """Save the spoken response as an MP3 file and play it."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"response_{uuid4().hex}.mp3"
    gTTS(text=message, lang="en").save(str(audio_path))

    print(f"Assistant: {message}")
    print(f"Audio file: {audio_path}")
    try:
        playsound(str(audio_path))
    except Exception as exc:
        print(f"Could not play the audio: {exc}")
        print("You can open the saved file manually.")
    return audio_path


def do_this_command(command: str) -> bool:
    """Handle a command; return False when the assistant should stop."""
    normalized = " ".join(command.casefold().split())
    response = RESPONSES.get(normalized, UNKNOWN_RESPONSE)
    say_message(response)
    return normalized != EXIT_COMMAND


def main() -> None:
    print("Available commands: " + ", ".join(RESPONSES))
    while True:
        try:
            command = input("Enter a command: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant stopped.")
            break

        try:
            should_continue = do_this_command(command)
        except Exception as exc:
            print(f"Could not create audio: {exc}")
            print("Check your internet connection and try again.")
            continue

        if not should_continue:
            break


if __name__ == "__main__":
    main()
