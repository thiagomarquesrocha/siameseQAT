def compile_model(model):
    model.compile(optimizer='adam', loss=model.get_loss(), metrics=model.get_metrics())