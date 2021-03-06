"""Here are some functions to train recurrent neural network with ctc loss
"""

import torch
import os

from tqdm import tqdm
from tensorboardX import SummaryWriter as summary_writer

from utils.voc import PAD_token, SOS_token

def maskNLLLoss(inp, target, mask):
    ntotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, ntotal.item()


def binary_mask(tensor, padding_value=PAD_token):
    """Generates a binary matrix that has zeroes where the sequences
    have been padded also called a 'mask'
    
    Arguments:
        tensor {Tensor} -- A padded tensor
    
    Keyword Arguments:
        padding_value {float} -- The value used for padding (default: {PAD_token})
    
    Returns:
        ByteTensor -- A binary matrix that 'masks' padding
    """
    return torch.ByteTensor(tensor != PAD_token)


def ctc_iter(input_sequences, inputlens, target_sequences, targetlens,
               model, optimiser, batch_size, num_targets, clip, device, ctc_loss):
    optimiser.zero_grad()
    input_sequences = input_sequences.to(device)
    inputlens = inputlens.to(device)
    target_sequences = target_sequences.to(device)
    targetlens = targetlens.to(device)

    if hasattr(model, "pack_unpack"):
        logprobs = model(input_sequences, inputlens)
    else:
        logprobs = model(input_sequences)
    # Switch order of batch and sequence dimensions
    loss = ctc_loss(logprobs.transpose(0,1), target_sequences, inputlens, targetlens)

    loss.backward()
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    _ = optimiser.step()

    return loss.item()

def train_ctc(modelname, corpusname, dataset, voc,
              model, optimiser, save_dir, epochs=20, 
              batch_size=8, print_every=1, save_every=100, clip=0.0,
              device='cpu', start_epoch=1):
    assert voc._mode == 'ctc', "Wrong vocabulary mode - should be 'ctc', but is {}".format(voc._mode)
    # init 
    print("Initialising...")
    print_loss = 0
    total_iterations = epochs * len(dataset)
    iter = 0
    model.train()
    ctc = torch.nn.CTCLoss()

    model.to(device)
    ctc.to(device)

    avg_train_loss = 0.0

    # Where to save the checkpoints and logs
    outdir = os.path.join(save_dir, modelname, corpusname, "{}l_{}hsz_{}drop".format(model.nlayers,
                                                                                     model.hsz,
                                                                                     model.dropout))
    os.makedirs(outdir, exist_ok=True)
    # TensorboardX 
    logx = summary_writer(outdir, comment='CTCdev')

    print("training...")
    with tqdm(total=total_iterations) as pbar:
        for epoch in range(start_epoch, epochs + 1):
            for xlens, ylens, xs, ys in dataset:
                loss = ctc_iter(xs, xlens, ys, ylens, model, optimiser,
                                    batch_size, len(voc), clip, device, ctc_loss=ctc)

                print_loss += loss

                # report progress
                if iter % print_every == 0:
                    print_loss_avg = print_loss / print_every
                    avg_train_loss += print_loss_avg
                    pbar.set_description("Epoch {}; Average loss: {:.4f}".format(
                                         epoch, print_loss_avg))
                    print_loss = 0 # reset ?
                if bool(save_every):
                    if iter % save_every == 0:
                        torch.save({
                            'epoch':epoch,
                            'mdl':model.state_dict(),
                            'opt':optimiser.state_dict(),
                            'loss': loss,
                            'voc': voc.__dict__,
                        }, os.path.join(outdir, "{}-{}.tar".format(epoch, 'checkpoint')))
                iter += 1
                pbar.update(1)

            # Ending each epoch, log model params and other info
            values = {"Avg Train Loss": avg_train_loss}
            logx.add_scalars("scalars", values, global_step=epoch)
            for tag, val in model.named_parameters():
                tag = tag.replace('.', '/')
                logx.add_histogram(tag, val.detach().numpy(), iter)
                logx.add_histogram(tag + '/grad', val.grad, iter)
    print("To visualise training, run: tensorboard --logdir={} --host localhost --port 8088".format(outdir))
        


def attention_iter(input_sequences, inputlens, target_sequences, targetlens,
               encoder, decoder, encoder_optimiser, decoder_optimiser, batch_size,
               teacher_forcing_ratio, clip, device):
    # Optimisers
    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    # Init decoder input and vars
    dec_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    max_targetlen = torch.max(targetlens)
    mask = binary_mask(target_sequences)

    input_sequences = input_sequences.to(device)
    target_sequences = target_sequences.to(device)
    targetlens = targetlens.to(device)
    max_targetlen = max_targetlen.to(device)
    inputlens = inputlens.to(device)
    mask = mask.to(device)
    dec_input = dec_input.to(device)

    # some vars
    loss = 0.0
    print_losses = []
    ntotals = 0.0

    # Pass entire sequence through decoder
    enc_out, enc_hidden_state = encoder(input_sequences, inputlens)
    
    # print(enc_hidden_state.size())
    enc_out = enc_out.transpose(0,1)
    target_sequences = target_sequences.transpose(0,1).long()
    mask = mask.transpose(0,1)

    # First decoder state is the last hidden _encoder_ state
    dec_hidden_state = enc_hidden_state[:decoder.nlayers]
    # dec_input = torch.ones(1, batch_size, dec_hidden_state.size()[-1]) * SOS_token

    # Use teacher forcing? True if inequality holds, else False
    teacher_forcing = torch.rand(1, 1).item() < teacher_forcing_ratio

    if teacher_forcing:
        # Teacher forcing makes the input to the next time step equal to the target 
        # in the current time step rather than the most likely prediction
        for step in range(max_targetlen):
            # 1 frame at a time
            dec_out, dec_hidden_state = decoder(dec_input, dec_hidden_state, enc_out)        
            # We need to make sure the dims fit and cast dtype as torch.long so we can embed
            dec_input = target_sequences[step,:].unsqueeze(0).long()
            # mask out the padding when we calculate loss
            maskloss, ntotal = maskNLLLoss(dec_out, target_sequences[step], mask[step])
            # we need to backprop loss for the entire sequence
            loss += maskloss
            # Use .item() so we don't pass gradients past sequences
            print_losses.append(maskloss.item() * ntotal)
            ntotals += ntotal
    else:
        for step in range(max_targetlen):
            # print('1best', step)
            dec_out, dec_hidden_state = decoder(dec_input, dec_hidden_state, enc_out)
            # Select best prediction
            _, onebest = dec_out.topk(1)
            # Reshape output to become the input vector from the previous time step
            dec_input = torch.Tensor([onebest[i][0] for i in range(batch_size)]).long()
            maskloss, ntotal = maskNLLLoss(dec_out, target_sequences[step], mask[step])
            loss += maskloss
            print_losses.append(maskloss.item() * ntotal)
            ntotals += ntotal


    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimiser.step()
    decoder_optimiser.step()

    return sum(print_losses) / ntotals


def train_attention(modelname, corpusname, dataset, voc,
              encoder, decoder, enc_optimiser, dec_optimiser, save_dir, epochs=20, 
              batch_size=8, print_every=1, save_every=100, clip=0.1,
              device='cpu', start_epoch=1, teacher_forcing_ratio=0.5):

    assert voc._mode == 'enc-dec', "Wrong vocabulary mode - should be 'enc-dec', but is {}".format(voc._mode)
    # init 
    print("Initialising...")
    print_loss = 0
    total_iterations = epochs * len(dataset)
    iter = 0
    encoder.train()
    decoder.train()

    encoder.to(device)
    decoder.to(device)


    avg_train_loss = 0.0

    # Where to save the checkpoints and logs
    # encoder and decoder has the same dropout for now
    outdir = os.path.join(save_dir, modelname, corpusname, 
                          "{0}-{1}l_{2}-{3}hsz_{4}drop".format(encoder.nlayers, decoder.nlayers,
                                                           encoder.hsz, decoder.hsz, encoder.dropout))
    os.makedirs(outdir, exist_ok=True)
    # TensorboardX 
    logx = summary_writer(outdir, comment='ATTNdev')


    print("training...")
    with tqdm(total=total_iterations) as pbar:
        for epoch in range(start_epoch, epochs + 1):
            for xlens, ylens, xs, ys in dataset:
                loss = attention_iter(xs, xlens, ys, ylens, encoder, decoder, enc_optimiser, dec_optimiser,
                                        batch_size, teacher_forcing_ratio, clip, device)
                # Just used for reporting        
                print_loss += loss
                logx.add_scalar('Train/Loss/Batch', loss.item(), global_step=iter)
                # report progress
                if iter % print_every == 0:
                    print_loss_avg = print_loss / print_every
                    pbar.set_description("Epoch {}; Average loss: {:.4f}".format(
                                         epoch, print_loss_avg))
                    avg_train_loss += print_loss_avg
                    print_loss = 0 # reset ?
                if bool(save_every):
                    if iter % save_every == 0:
                        torch.save({
                            'epoch':epoch,
                            'enc':encoder.state_dict(),
                            'enc_opt':enc_optimiser.state_dict(),
                            'dec':decoder.state_dict(),
                            'dec_opt':dec_optimiser.state_dict(),
                            'loss': loss,
                            'voc': voc.__dict__,
                        }, os.path.join(outdir, "{}-{}.tar".format(epoch, 'checkpoint')))
                iter += 1
                pbar.update(1)

            # Ending each epoch, log model params and other info
            values = {"Train/Loss/Avg": avg_train_loss}
            logx.add_scalars("scalars", values, global_step=epoch)
            for model in (('ENC', encoder), ('DEC', decoder)):
                for tag, val in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logx.add_histogram(tag, val.detach().numpy(), iter)
                    logx.add_histogram(tag + '/grad', val.grad, iter)

    print("To visualise training, run: tensorboard --logdir={} --host localhost --port 8088".format(outdir))


if __name__ == "__main__":
    import sys
    import torch
    import torch.utils.data as tud
    from random import sample
    import utils.data as data
    import utils.audio as audio
    from utils.dataset import AudioDataset, Collate
    from models.gru import CTCgrup
    from utils.voc import generate_char_voc

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    translist = [open(x).read().strip() for x in reflist]
    # Sample randomly because an4 is already sorted
    idxs = sample(range(0,len(audiolist)), 40)
    testlist = [audiolist[x] for x in idxs]
    testref = [translist[x] for x in idxs]
    sortedlist = audio.audiosort(testlist, list_of_references=testref)
    testlist, testref = zip(*sortedlist)
 
    voc = generate_char_voc(testref, "LE TEST")
    partition, labels = data.format_data(testlist, testref, voc)

    trainset = AudioDataset(partition['train'], labels, audio.mfcc)
    ctc_batch_fn = Collate(-1)
    bsz = 8
    params = {'batch_size': bsz,
              'shuffle':False,
              'num_workers':2,
              'collate_fn':ctc_batch_fn,
              'drop_last': True}

    allloss = 0
    traingenerator = tud.DataLoader(trainset, **params, )
    model = CTCgrup(40, len(voc), 120, 2)
    model.to(device)
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=0.01)

    print("Test train_ctc() (and train_iter_ctc())")
    try:
        train_ctc("traintest", voc.name, traingenerator, voc,
                  model,optimiser, "an4test", epochs=50, batch_size=bsz,
                  print_every=2, save_every=100, clip=0.0, device=device)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.exit()
