"""Скрипт для обнаружения всех недостающих атрибутов в vocab_remi.pkl."""
import pickle
from huggingface_hub import hf_hub_download

vocab_path = hf_hub_download(repo_id="amaai-lab/text2midi", filename="vocab_remi.pkl")
with open(vocab_path, "rb") as f:
    r_tok = pickle.load(f)

# Известные патчи
PATCHES = {
    "use_velocities":              True,
    "use_note_duration_programs":  [],    # must be iterable
    "default_note_duration":       0.5,   # fraction of a beat (eighth note)
}
for k, v in PATCHES.items():
    if not hasattr(r_tok.config, k):
        setattr(r_tok.config, k, v)
        print(f"  patched: {k} = {v}")

# Итеративно обнаруживаем и патчим все AttributeError
tokens = list(range(1, 50))
for attempt in range(15):
    try:
        result = r_tok.decode(tokens)
        print(f"\ndecode OK after {attempt} extra patches")
        print(f"result type: {type(result)}")
        print(f"has dump_midi: {hasattr(result, 'dump_midi')}")
        break
    except AttributeError as e:
        msg = str(e)
        # Extract attribute name from error like "'TokenizerConfig' object has no attribute 'foo'"
        attr = msg.split("'")[-2]
        print(f"  patched: {attr} = None")
        setattr(r_tok.config, attr, None)
    except Exception as e:
        import traceback
        print(f"\nNon-AttributeError: {type(e).__name__}: {e}")
        traceback.print_exc()
        break
else:
    print("Could not resolve all AttributeErrors after 15 attempts")

