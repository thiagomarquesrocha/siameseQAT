def compile_model(model):
    model.get_model().compile(optimizer='adam', loss=model.get_loss(), metrics=model.get_metrics())
    return model.get_model()