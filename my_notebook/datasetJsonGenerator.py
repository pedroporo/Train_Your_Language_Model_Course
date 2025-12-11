import sys
import re
import pandas as pd
from pathlib import Path
import numpy as np
sys.path.append('..')
from minbpe.v2 import RegexTokenizer
import time






class CombinedTextGenerator:
    def read_whatsapp_chat(self,file_path: str) -> pd.DataFrame:
        encryption_message = "Los mensajes y las llamadas están cifrados de extremo a extremo. Solo las personas en este chat pueden leerlos, escucharlos o compartirlos. Obtén más información."
        media_pattern = "<Multimedia omitido>"
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        edited_message = "<Se editó este mensaje.>"
        deleted_message = "Eliminaste este mensaje."
        null_message = "null"
        created_group_message = "creó el grupo"
        added_you_to_group_message = "Se te añadió al grupo."
        added_someone_to_group_message = "añadió a"
        removed_someone_to_group_message = "eliminó a"
        tagging_pattern = r'@[\w]+'

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

       
        filtered_lines = []
        for line in lines:
            if (
                encryption_message not in line and
                deleted_message not in line and
                null_message != line.split(" ")[-1] and
                media_pattern not in line and
                created_group_message not in line and
                added_you_to_group_message not in line and
                added_someone_to_group_message not in line and
                removed_someone_to_group_message not in line and
                not re.search(email_pattern, line) and
                not re.search(url_pattern, line)
            ):
                line = line.replace(edited_message, "").strip()
                line = re.sub(tagging_pattern, "", line).strip()
                filtered_lines.append(line)

      
        content = '\n'.join(filtered_lines)
      
        content = content.replace('\u202f', ' ')
        # Remove square brackets if they surround the timestamp (only for iOS)
        content = re.sub(
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm])\]',
            r'\1',
            content
        )
        # Remove LRM and RLM characters (Left-to-Right Mark and Right-to-Left Mark)
        content = content.replace('\u200E', '').replace('\u200F', '')

        # Updated regex pattern to match both iOS and Android WhatsApp exports.
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s?(?:-|\~)?\s?(.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}|$)'
        messages = re.findall(pattern, content, re.DOTALL)
        df = pd.DataFrame(messages, columns=['timestamp', 'sender', 'message'])

        timestamps = []
        for timestamp in df['timestamp']:
            try:
                timestamp = pd.to_datetime(
                    timestamp, format='mixed', errors='coerce')
            except Exception as e:
                print(f"Error parsing timestamp '{timestamp}': {e}")
                timestamp = pd.NaT
            timestamps.append(timestamp)

        df['timestamp'] = timestamps
        return df
    
    def generateText(self):
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--input', '-i', default=str(Path("../data/private")),
                            help='Directorio de entrada que contiene archivos .txt (se recorre recursivamente).')
        args, _ = parser.parse_known_args()

        input_path = Path(args.input)
        all_chats = {}
        # Usar rglob para búsqueda recursiva y soportar rutas con espacios correctamente (Path maneja bien esto)
        if input_path.is_file() and input_path.suffix.lower() == '.txt':
            files = [input_path]
        else:
            #print("Hola")
            #files = list(input_path.rglob('*.txt'))
            files = list(input_path.glob('*.txt'))
        print(files)
        for file in files:
            try:
                # clave única: ruta relativa desde el directorio de entrada para evitar colisiones
                try:
                    key = str(file.relative_to(input_path))
                except Exception:
                    key = str(file)
                print(f"Procesando: {file}")
                df = self.read_whatsapp_chat(str(file))
                # Si el parser no encontró mensajes (p. ej. archivos tipo libro), hacer fallback a texto bruto
                if df is None or df.empty or 'message' not in df.columns:
                    try:
                        with open(file, 'r', encoding='utf-8') as fh:
                            raw = fh.read()
                        df = pd.DataFrame({'message': [raw]})
                    except Exception as e:
                        print(f"No se pudo leer como texto bruto {file}: {e}")
                        df = pd.DataFrame({'message': []})
                all_chats[key] = df
            except Exception as e:
                print(f"Error procesando {file}: {e}")

        text_sequence = ""
        for key, df in all_chats.items():
            if 'message' in df.columns:
                text_sequence += " ".join(df['message'].dropna().astype(str).values)

        # Asegurarse de que la carpeta de salida exista
        out_path = Path("../output")
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "combined_text4.txt", "w", encoding="utf-8") as f:
            f.write(text_sequence)

class AtlasSetGenerator:
    def __init__(self):
        self.tokenizer=RegexTokenizer()
        self.tokenizer.load(model_file="../output/tokenizer/pedro_nuevo_tokenizer3.model")
        self.encoded_text_sequence = []
        self.batch_size = 100_000_000
        # por defecto leerá el archivo generado en ../output/combined_text2.txt
        self.file_path = "../output/combined_text4.txt"

    def generateAtlas(self):
        with open(self.file_path, "r") as f:
            while True:
                chunk = f.read(self.batch_size)
                if not chunk:
                    break
                batch_tokens = self.tokenizer.encode(chunk)
                self.encoded_text_sequence.extend(batch_tokens)
                print(f"Processed {len(self.encoded_text_sequence)} tokens so far.")

        print(f"Total tokens: {len(self.encoded_text_sequence)}")
        output_path = "../output/encoded_data/encoded_atlaset_p_nuevo4.npy"
        np.save(output_path, np.array(self.encoded_text_sequence, dtype=np.int64))
        del self.encoded_text_sequence



if __name__ == "__main__":
    ctg=CombinedTextGenerator()
    ctg.generateText()
    time.sleep(3)
    asg=AtlasSetGenerator()
    asg.generateAtlas()
