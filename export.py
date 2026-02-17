from datetime import datetime
from torch import nn

def export_weights(policy, filepath: str, wandb_url: str = None, run_id: str = None):
    float_to_str = lambda x: str(float(x))

    # Collect all linear layers in order: encoder, decoder, action_mean
    encoder_layers = [layer for layer in policy.encoder if isinstance(layer, nn.Linear)]
    decoder_layers = [layer for layer in policy.decoder if isinstance(layer, nn.Linear)]
    linear_layers = encoder_layers + decoder_layers + [policy.action_mean]

    # LSTM weights
    lstm = policy.lstm
    input_size = lstm.input_size
    hidden_size = lstm.hidden_size

    with open(filepath, "w+") as file:
        file.write("/*\n")
        file.write(f" * Neural Network Weights (LSTM)\n")
        file.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if run_id:
            file.write(f" * Run ID: {run_id}\n")
        if wandb_url:
            file.write(f" * Wandb: {wandb_url}\n")
        file.write(" */\n\n")
        file.write("#include \"neural_network.h\"\n")
        file.write("#include \"nn_helpers.h\"\n\n")

        # Write linear layer weights
        for i, layer in enumerate(linear_layers):
            weights = layer.weight.data.cpu().numpy()
            biases = layer.bias.data.cpu().numpy()
            file.write(f"const float weights_fc{i}[] = {{\n")
            file.write(",\n".join([", ".join(map(float_to_str, row)) for row in weights]))
            file.write("\n};\n\n")
            file.write(f"const float biases_fc{i}[] = {{\n")
            file.write(", ".join(map(float_to_str, biases)))
            file.write("\n};\n\n")

        # Write LSTM weights (layer 0, single layer)
        # PyTorch LSTM stores: weight_ih_l0 (4*H, input), weight_hh_l0 (4*H, H), bias_ih_l0 (4*H), bias_hh_l0 (4*H)
        for name in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
            param = getattr(lstm, name).data.cpu().numpy()
            file.write(f"const float lstm_{name}[] = {{\n")
            if param.ndim == 2:
                file.write(",\n".join([", ".join(map(float_to_str, row)) for row in param]))
            else:
                file.write(", ".join(map(float_to_str, param)))
            file.write("\n};\n\n")

        # Write forward function
        file.write(f"#define LSTM_HIDDEN_SIZE {hidden_size}\n")
        file.write(f"#define LSTM_INPUT_SIZE {input_size}\n\n")

        # Build forward function
        forward_fn = "void nn_forward(const float* input, float* output, float* lstm_h, float* lstm_c) {\n"
        forward_fn += "    const float* fc0_output = input;\n"

        fc_idx = 0
        # Encoder layers
        for i, layer in enumerate(encoder_layers):
            forward_fn += f"    float fc{fc_idx+1}_output[{layer.out_features}];\n"
            forward_fn += f"    nn_linear(weights_fc{fc_idx}, biases_fc{fc_idx}, fc{fc_idx}_output, {layer.in_features}, {layer.out_features}, fc{fc_idx+1}_output);\n"
            forward_fn += f"    nn_elu(fc{fc_idx+1}_output, {layer.out_features});\n"
            fc_idx += 1

        # LSTM step
        forward_fn += f"    nn_lstm_step(lstm_weight_ih_l0, lstm_weight_hh_l0, lstm_bias_ih_l0, lstm_bias_hh_l0, fc{fc_idx}_output, lstm_h, lstm_c, {input_size}, {hidden_size});\n"

        # Decoder layers
        prev_output = "lstm_h"
        for i, layer in enumerate(decoder_layers):
            forward_fn += f"    float fc{fc_idx+1}_output[{layer.out_features}];\n"
            forward_fn += f"    nn_linear(weights_fc{fc_idx}, biases_fc{fc_idx}, {prev_output}, {layer.in_features}, {layer.out_features}, fc{fc_idx+1}_output);\n"
            forward_fn += f"    nn_elu(fc{fc_idx+1}_output, {layer.out_features});\n"
            prev_output = f"fc{fc_idx+1}_output"
            fc_idx += 1

        # Action mean (final layer, output goes to output buffer, no activation)
        action_layer = policy.action_mean
        forward_fn += f"    nn_linear(weights_fc{fc_idx}, biases_fc{fc_idx}, {prev_output}, {action_layer.in_features}, {action_layer.out_features}, output);\n"

        forward_fn += "}"
        file.write(forward_fn)
