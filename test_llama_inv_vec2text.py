import vec2text

device = "cuda"
inversion_model=vec2text.models.InversionFromLogitsEmbModel.from_pretrained("jxm/t5-base__llama-7b__one-million-instructions__emb").to(device)
corrector_model = vec2text.models.CorrectorEncoderFromLogitsModel.from_pretrained("jxm/t5-base___llama-7b___one-million-instructions__correct")
corrector = vec2text.load_corrector(inversion_model,corrector_model)
