import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from torch import Tensor
from functools import reduce
import pickle

def infer_basic_model(model: nn.Module,
                      batch: torch.Tensor,
                      opts: Optional[dict] = None,
                      device: torch.device = None) -> torch.Tensor:
    """Generate prediction for models that take an input and generate the output in one forward pass
    Args: 
        model: The model to use for inference
        input: The input to the model
        opts: Options to pass to the model as a dictionary, can be empty here
    """
    x, y = batch
    target = y[0]
    # Need additional logic around gradient tracking for modules associated with the
    #   transformer because it seems behavior can change depending on the no_grad() context
    if opts is None:
        track_gradients = False  # Default option
    elif (opts is not None):
        if 'track_gradients' in opts:
            track_gradients = opts['track_gradients']
        else:
            track_gradients = False
    if track_gradients:
        output = model(x)
    else:
        with torch.no_grad():
            output = model(x)
    # Also save x[1] which is the set of SMILES strings
    #   Note that even for a batch size of 1, the batch smiles element
    #   returned by a dataloader is a tuple of the form (str,) which converts
    #   correctly to [str] when using list(). It does not cause the string
    #   to break apart into a list of characters.
    return [(
        target.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
        list(x[1])
    )]


def get_top_k_sample_batched(k_val: int | float,
                             character_probabilities: Tensor) -> Tensor:
    """
    Generates the next character using top-k sampling scheme.

    In top-k sampling, the probability mass is redistributed among the
    top-k next tokens, where k is a hyperparameter. Once redistributed, 
    the next token is sampled from the top-k tokens.
    """
    top_values, top_indices = torch.topk(
        character_probabilities, k_val, sorted=True)
    # Take the sum of the top probabilities and renormalize
    tot_probs = top_values / torch.sum(top_values, dim=-1).reshape(-1, 1)
    # Sample from the top k probabilities. This represents a multinomial distribution
    try:
        assert (torch.allclose(torch.sum(tot_probs, dim=-1), torch.tensor(1.0)))
    except:
        print("Probabilities did not pass allclose check!")
        print(f"Sum of probs is {torch.sum(tot_probs)}")
    selected_index = torch.multinomial(tot_probs, 1)
    # For gather to work, both tensors have to have the same number of dimensions:
    if len(top_indices.shape) != len(selected_index.shape):
        top_indices = top_indices.reshape(selected_index.shape[0], -1)
    output = torch.gather(top_indices, -1, selected_index)
    return output


def transformer_forward_inference(model: nn.Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  track_grads: bool = True) -> torch.Tensor:
    """Forward pass in inference for a transformer model
    Args:
        model: Transformer model that takes a tuple of arguments (x, None), (y, None)
        x: The input to the model, i.e. context
        y: The shifted target to the model, i.e. the start tokens
        track_grads: Whether gradients are tracked during inference pass. Default is True
            because Transformer behavior can change w.r.t no_grad() context
    """
    if track_grads:
        return model((x, None), (y, None))
    else:
        with torch.no_grad():
            return model((x, None), (y, None))


def encoder_forward_inference(model: nn.Module,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              track_grads: bool = True) -> torch.Tensor:
    """Forward pass in inference for an encoder model
    Args:
        model: Encoder model that takes a single input
        x: The input to the model, the context consisting of the start tokens
        y: The shifted target to the model also consisting of only start tokens,
            not used here. Included for interface consistency with 
            transformer_forward_inference()
        track_grads: Whether gradients are tracked during inference pass. Default is True 
            because Transformer module behaviors can change w.r.t no_grad() context
    """
    if track_grads:
        return model((x, None))
    else:
        with torch.no_grad():
            return model((x, None))
        
def output_base(targets,
                curr_batch_predictions,
                effective_bsize,
                num_pred_per_tgt,
                smiles,
                decode):
    assert (targets.shape[0] == len(
        curr_batch_predictions[0]) == effective_bsize)
    generated_predictions = []
    for i in range(effective_bsize):
        # TODO: Think, is this the best way to represent output predictions for each batch as
        #   tuples of (target, [pred1, pred2, ...])?
        generated_predictions.append((
            smiles[i] if decode else targets[i].detach().cpu().numpy(),
            list(curr_batch_predictions[j][i]
                 for j in range(num_pred_per_tgt)),
            smiles[i]
        ))
    return generated_predictions

def output_era(effective_bsize,
               curr_batch_predictions,
               num_pred_per_tgt,
               pad_token: int = -100):
    #REMOVE THIS LATER
    assert(pad_token == 324)
    generated_tokens = []
    for i in range(effective_bsize):
        generated_tokens.append(
                list(curr_batch_predictions[j][i]
                 for j in range(num_pred_per_tgt))
        )
    generated_tokens = list(reduce(lambda x, y : x + y, generated_tokens))
    return pad_ragged_seq(generated_tokens, pad_token=pad_token)

def output_era_prompted(prompts_and_generations,
                        pad_token: int = -100):
    assert(pad_token == 324)
    generated_tokens = []
    # import pdb; pdb.set_trace()
    for prompt, generation in prompts_and_generations:
        assert(np.allclose(prompt, gen[:len(prompt)]) for gen in generation)
        #Remove the last token if it's a stop token, and remove the prompt too
        corrected_generations = []
        for gen in generation:
            # if gen[-1] == pad_token + 2:
            #     corrected_generations.append(gen[len(prompt):-1])
            # else:
            #Retain stop and start tokens in the sequence, -1 because 
            #   prompt includes second start
            corrected_generations.append(gen[len(prompt)-1:])
        generated_tokens.append(corrected_generations)
    generated_tokens = list(reduce(lambda x, y : x + y, generated_tokens))
    return pad_ragged_seq(generated_tokens, pad_token=pad_token)

def output_era_retain_prompt(prompts_and_generations,
                             pad_token: int = -100):
    assert pad_token == 324
    generated_sequences = []
    masks = []
    for prompt, generation in prompts_and_generations:
        assert(np.array_equal(prompt, gen[:len(prompt)]) for gen in generation)
        for gen in generation:
            generated_sequences.append(gen)
            curr_mask = np.ones(len(gen))
            curr_mask[:len(prompt)] = 0
            masks.append(curr_mask)
    padded_generated_seqs = pad_ragged_seq(generated_sequences, pad_token=pad_token)
    padded_masks = pad_ragged_seq(masks, pad_token=0)
    return padded_generated_seqs, padded_masks

def pad_ragged_seq(ragged_seqs: list[np.ndarray], pad_token: int) -> np.ndarray:
    """Pads a list of ragged sequences to the length of the longest sequence
    Args:
        ragged_seqs: A list of numpy arrays, each of which is a ragged sequence
    Returns:
        A numpy array with the ragged sequences padded to the length of the longest sequence
    """
    max_len = max(len(seq) for seq in ragged_seqs)
    padded_seqs = np.ones((len(ragged_seqs), max_len), dtype=np.int64) * pad_token
    for i, seq in enumerate(ragged_seqs):
        padded_seqs[i, :len(seq)] = seq
    return padded_seqs

def infer_SMILES_generator(model: nn.Module,
                           batch: torch.Tensor,
                           opts: dict,
                           device: torch.device = None) -> list[tuple[str, list[str]]] | list[tuple[np.ndarray, list[np.ndarray]]]:
    """Generates a prediction for the input using sampling over a transformer model
    Args:
        model: The model to use for inference
        batch: The input to the model
        opts: Options to pass to the model as a dictionary
        device: The device to use for inference

    The opts dictionary should contain the following additional arguments:
        'num_pred_per_tgt' (int): The number of predictions to generate for each input
        'sample_val' (int or float): The sampling value for the model, e.g. the number of values to use for
            top-k sampling
        'stop_token' (int): The stop token to use for the model
        'start_token' (int): The start token to use for the model
        'track_gradients' (bool): Whether to track gradients during inference. This is because the transformer
            is known to misbehave if gradient tracking is disabled in certain cases
        'alphabet' (str): Path to a file containing the alphabet for the model to use in decoding
        'decode' (bool): Whether to decode the output indices of the model against the provided alphabet
    """
    num_pred_per_tgt = opts['num_pred_per_tgt']
    sample_val = opts['sample_val']
    stop_token = opts['tgt_stop_token']
    start_token = opts['tgt_start_token']
    track_gradients = opts['track_gradients']
    alphabet = opts['alphabet']
    if isinstance(alphabet, str):
        alphabet = np.load(alphabet, allow_pickle=True)
    pad_token = len(alphabet)
    #Add additional tokens
    alphabet = np.append(alphabet, ['<PAD>', '<START>', '<STOP>'])
    decode = opts['decode']
    model_type = opts['model_type']
    run_mode = opts['run_mode']
    token_limit = opts['token_limit']

    x, y = batch
    curr_batch_predictions = []
    effective_bsize = x[0].shape[0]
    if run_mode == 'generation':
        if model_type == 'Transformer':
            targets = y[1]
        elif model_type == 'Encoder':
            targets = y[0]
    elif run_mode == 'era':
        targets = None
    smiles = x[1]

    for _ in range(num_pred_per_tgt):

        # TODO: Consider changing what gets fed in via dataloader to allow the inference start of the transformer
        #   to be something other than the start token.

        if model_type == 'Encoder':
            working_y = x[0].clone().to(device)
            working_x = torch.tensor(
                [start_token] * effective_bsize).reshape(effective_bsize, 1).to(device)
        elif model_type == 'Transformer':
            working_x = x[0].clone().to(device)
            working_y = torch.tensor(
                [start_token] * effective_bsize).reshape(effective_bsize, 1).to(device)
        # import pdb; pdb.set_trace()
        completed_structures = [None] * effective_bsize
        index_mapping = torch.arange(
            effective_bsize, device=device, dtype=torch.long)
        all_structures_completed = False
        iter_counter = 0

        while not all_structures_completed:
            if (iter_counter % 10 == 0):
                print(f"On iteration {iter_counter}")

            if model_type == 'Encoder':
                next_pos = encoder_forward_inference(
                    model, working_y, working_x, track_gradients)
            elif model_type == 'Transformer':
                next_pos = transformer_forward_inference(
                    model, working_x, working_y, track_gradients)
            else:
                raise ValueError(
                    "Model must be either 'Encoder' or 'Transformer' right now")

            next_val = next_pos[:, -1, :]
            char_probs = torch.nn.functional.softmax(next_val, dim=-1)
            selected_indices = get_top_k_sample_batched(sample_val, char_probs)
            concatenated_results = torch.cat(
                (working_y, selected_indices), dim=-1)
            stop_token_mask = concatenated_results[:, -1] == stop_token
            comp_structs = concatenated_results[stop_token_mask]
            comp_inds = index_mapping[stop_token_mask]
            for i, ind in enumerate(comp_inds):
                completed_structures[ind] = comp_structs[i].detach(
                ).cpu().numpy()
            working_y = concatenated_results[~stop_token_mask]
            working_x = working_x[~stop_token_mask]
            index_mapping = index_mapping[~stop_token_mask]
            if working_y.shape[-1] > token_limit:
                working_y = torch.cat((working_y,
                                       torch.tensor([stop_token] * working_y.shape[0]).reshape(-1, 1).to(device)),
                                      dim=-1)
                for j, ind in enumerate(index_mapping):
                    completed_structures[ind] = working_y[j].detach(
                    ).cpu().numpy()
                all_structures_completed = True
            if len(working_y) == 0:
                all_structures_completed = True
            iter_counter += 1
        for elem in completed_structures:
            assert (elem is not None)
        if decode:
            generated_smiles = [''.join(
                np.array(alphabet)[elem[1:-1].astype(int)]) for elem in completed_structures]
            curr_batch_predictions.append(generated_smiles)
        else:
            curr_batch_predictions.append(completed_structures)

    if run_mode == 'generation':
        generated_output = output_base(targets,
                                    curr_batch_predictions,
                                    effective_bsize,
                                    num_pred_per_tgt,
                                    smiles,
                                    decode)
    elif run_mode == 'era':
        generated_output = output_era(effective_bsize,
                                    curr_batch_predictions,
                                    num_pred_per_tgt,
                                    pad_token)
    
    return generated_output


def prompted_generation(model: nn.Module,
                        batch: torch.Tensor,
                        opts: dict,
                        device: torch.device = None) -> list[tuple[str, list[str]]] | list[tuple[np.ndarray, list[np.ndarray]]]:
    """Prompted generation where input is no longer start token but some arbitrary sequence"""

    num_pred_per_tgt = opts['num_pred_per_tgt']
    sample_val = opts['sample_val']
    stop_token = opts['tgt_stop_token']
    track_gradients = opts['track_gradients']
    alphabet = opts['alphabet']
    if isinstance(alphabet, str):
        alphabet = np.load(alphabet, allow_pickle=True)
    pad_token = len(alphabet)
    #Add additional tokens
    alphabet = np.append(alphabet, ['<PAD>', '<START>', '<STOP>'])
    decode = opts['decode']
    token_limit = opts['token_limit']
    run_mode = opts['run_mode']

    x, y = batch #Here, x contains a series of prompting sequences, we iterate over one by one
    prompts_and_generations = []
    all_prompts = x[0]
    for seq in all_prompts:
        #Strip padding
        seq = seq[seq != pad_token]
        assert len(seq) > 0
        # import pdb; pdb.set_trace()
        #if len(seq) == 0:
        completed_structures = []
        for _ in range(num_pred_per_tgt):
            working_x = seq.reshape(1, -1).to(device)
            working_y = None
            iter_counter = 0
            complete = False

            while not complete:
                if ((iter_counter > 0) and (iter_counter % 100 == 0)):
                    print(f"On iteration {iter_counter}")

                next_pos = encoder_forward_inference(
                        model, working_x, working_y, track_gradients)
                next_val = next_pos[:, -1, :]
                char_probs = torch.nn.functional.softmax(next_val, dim=-1)
                selected_indices = get_top_k_sample_batched(sample_val, char_probs)
                concatenated_results = torch.cat(
                    (working_x, selected_indices), dim=-1)
                stop_token_mask = concatenated_results[:, -1] == stop_token
                comp_structs = concatenated_results[stop_token_mask]
                working_x = concatenated_results[~stop_token_mask]
                
                if comp_structs.shape[0] > 0:
                    completed_structures.append(comp_structs[0].detach().cpu().numpy())
                    complete = True
                if working_x.shape[-1] > token_limit:
                    working_x = torch.cat((working_x,
                                        torch.tensor([stop_token] * working_x.shape[0]).reshape(-1, 1).to(device)),
                                        dim=-1)
                    completed_structures.append(working_x[0].detach().cpu().numpy())
                    complete = True
                iter_counter += 1
                # if iter_counter % 1000 == 0:
                #     import pdb; pdb.set_trace()
        if decode:
            generated_smiles = [''.join(
                np.array(alphabet)[elem.astype(int)]) for elem in completed_structures
            ]
            start_sequence = ''.join(np.array(alphabet)[seq.detach().cpu().numpy().astype(int)])
            prompts_and_generations.append((start_sequence, generated_smiles))
        else:
            prompts_and_generations.append((seq.detach().cpu().numpy(), completed_structures))
    if run_mode == 'generation':
        return prompts_and_generations
    elif run_mode == 'era':
        return output_era_prompted(prompts_and_generations, pad_token)
    elif run_mode == 'era_retain_prompts':
        return output_era_retain_prompt(prompts_and_generations, pad_token)
