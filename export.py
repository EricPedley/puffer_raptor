from datetime import datetime
from torch import nn

def export_weights(policy, filepath: str, wandb_url: str = None, run_id: str = None):
    net = policy.net

    i = 0
    forward_fn = """void nn_forward(const float* input, float* output) {
    const float* fc0_output = input;"""

    layers = [layer for layer in net if isinstance(layer, nn.Linear)]+[policy.action_mean]
    float_to_str = lambda x: str(float(x))
    with open(filepath, "w+") as file:
        file.write("/*\n")
        file.write(f" * Neural Network Weights\n")
        file.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if run_id:
            file.write(f" * Run ID: {run_id}\n")
        if wandb_url:
            file.write(f" * Wandb: {wandb_url}\n")
        file.write(" */\n\n")
        file.write("#include \"neural_network.h\"\n")
        file.write("#include \"nn_helpers.h\"\n\n")
        for i, layer in enumerate(layers):
            weights_layer = layer.weight.data.cpu().numpy()
            biases_layer = layer.bias.data.cpu().numpy()
            file.write(f"const float weights_fc{i}[] = {{\n")
            file.write(
                ",\n".join([", ".join(map(float_to_str, row)) for row in weights_layer])
            )
            file.write("\n};\n\n")

            file.write(f"const float biases_fc{i}[] = {{\n")
            file.write(", ".join(map(float_to_str, biases_layer)))
            file.write("\n};\n\n")
        
            if i == len(layers) - 1:
                forward_fn += f'\n    float* fc{i+1}_output = output;'
            else:
                forward_fn += f'\n    float fc{i+1}_output[{layer.out_features}];'
            forward_fn += \
                f'\n    nn_linear(weights_fc{i}, biases_fc{i}, fc{i}_output, {layer.in_features}, {layer.out_features}, fc{i+1}_output);' \
                f'\n    nn_elu(fc{i+1}_output, {layer.out_features});'
            
        forward_fn += "\n}"
        file.write(forward_fn)