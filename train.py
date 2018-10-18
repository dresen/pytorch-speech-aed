"""Here are some functions to train recurrent neural network with ctc loss
"""


import torch
import os


def train_iter_ctc(input_sequences, inputlens, target_sequences, targetlens,
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
    logprobs = logprobs.reshape(-1, batch_size, num_targets + 1)
    loss = ctc_loss(logprobs, target_sequences, inputlens, targetlens)

    loss.backward()
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    _ = optimiser.step()

    return loss.item()

def train_ctc(modelname, corpusname, dataset, sym2int, int2sym,
              model, optimiser, save_dir, epochs=20, 
              batch_size=8, print_every=1, save_every=100, clip=0.0,
              device='cpu', start_epoch=1):
    # init 
    print("Initialising...")
    print_loss = 0
    total_iterations = epochs * len(dataset)
    iter = 0
    model.train()
    ctc = torch.nn.CTCLoss()
    print("training...")
    for epoch in range(start_epoch, epochs + 1):
        for xlens, ylens, xs, ys in dataset:
            loss = train_iter_ctc(xs, xlens, ys, ylens, model, optimiser,
                                  batch_size, len(sym2int), clip, device, ctc_loss=ctc)

            print_loss += loss

            # report progress
            if iter % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Epoch {}; % complete: {:.1f}%; Average loss: {:.4f}".format(
                      epoch, iter/total_iterations*100, print_loss_avg))
                print_loss = 0 # reset ?
            if bool(save_every):
                if iter % save_every == 0:
                    outdir = os.path.join(save_dir, modelname, corpusname, 
                                        "{}l_{}hsz_{}drop".format(model.nlayers, model.hsz, model.dropout))
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    torch.save({
                        'epoch':epoch,
                        'mdl':model.state_dict(),
                        'opt':optimiser.state_dict(),
                        'loss': loss,
                        'sym2int': sym2int,
                        'int2sym': int2sym,
                    }, os.path.join(outdir, "{}-{}.tar".format(epoch, 'checkpoint')))
            iter += 1
        

if __name__ == "__main__":
    import sys
    import torch
    import torch.utils.data as tud
    from random import sample
    import utils.data as data
    import utils.text as text
    import utils.audio as audio
    from utils.dataset import AudioDataset, Collate
    from ctc_lstm import CTClstm

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
 
    sym2int, int2sym = text.make_char_int_maps(testref, offset=1)
    partition, labels = data.format_data(testlist, testref, sym2int, text.labels_from_string)

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
    model = CTClstm(40, len(sym2int), 120, 2)
    model.to(device)
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=0.01)

    print("Test train_ctc() (and train_iter_ctc())")
    try:
        train_ctc("traintest", "AN4", traingenerator, sym2int, int2sym,
                  model,optimiser, "an4test", epochs=50, batch_size=bsz,
                  print_every=2, save_every=100, clip=0.0, device=device)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.exit()
