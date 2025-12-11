
if isinstance(text, str) and os.path.isfile(text):
            # Si es un archivo, usamos generadores para leer por chunks
            def text_chunk_generator():
                with open(text, 'r', encoding='utf-8') as f:
                    for line in f:
                        for chunk in re.findall(self.compiled_pattern, line):
                            yield chunk
            text_chunks = text_chunk_generator()
        else:
            # Si es texto directo, procesamos normalmente
            text_chunks = re.findall(self.compiled_pattern, text)

        # Convertir chunks a IDs usando generadores para mejor manejo de memoria
        def ids_generator():
            for chunk in text_chunks:
                yield list(chunk.encode("utf-8"))
        ids = list(ids_generator())  # Convertimos a lista para procesamiento paralelo