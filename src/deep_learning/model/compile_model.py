from keras.models import Model

def compile_model(model):
    model.get_model().compile(optimizer='adam', loss=model.get_loss(), metrics=model.get_metrics())
    return model.get_model()

def get_bug_encoder(model, loss, name='bug_encoder'):
    bug_enconder = model.get_layer('concatenated_bug_embed')
    output = bug_enconder.output
    inputs = model.inputs[:-1] # Remove label input (1, )
    bug_enconder = Model(inputs = inputs, outputs = output, name = name)
    bug_enconder.compile(optimizer='adam', loss=loss)
    return bug_enconder