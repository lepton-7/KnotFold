import os
import sys
import torch
import click
import tempfile
import subprocess
import numpy as np
import torch.nn.functional as F

from model.main_model import MainModel as Model

from pathlib import Path
from Bio.SeqIO import parse
from datetime import datetime
from time import perf_counter_ns
from dataclasses import dataclass
from Bio.SeqRecord import SeqRecord

old_print = vars(__builtins__)["print"]


def print(*args, sep=" ", **kwargs):
    combined_arg = sep.join(map(str, args))
    [
        old_print(datetime.now(), "|", x, sep=" ", **kwargs)
        for x in combined_arg.split("\n")
    ]


@dataclass
class Perf:
    slen: int
    id: str
    inf_seq: int | None = None
    inf_ref: int | None = None
    min_cost_flow: int | None = None

    def __str__(self) -> str:
        assert self.inf_seq is not None
        assert self.inf_ref is not None
        assert self.min_cost_flow is not None
        return f"""Runtimes for {self.id} of length {self.slen}:
    Sequence inference: {self.inf_seq/1e6:,.3f} ms
    Reference inference: {self.inf_ref/1e6:,.3f} ms
    Structure flow algorithm: {self.min_cost_flow/1e6:,.3f} ms"""


def load_model(chk_path):
    model = Model().eval()
    chk = torch.load(chk_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in chk["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model


def inference(seq: str, weight, cuda):
    model = load_model(os.path.join(os.path.dirname(__file__), weight))
    seq = "".join([_ if _ in "ACGU" else "N" for _ in seq])
    vocab = np.full(128, -1, dtype=np.int16)
    vocab[np.array("NAUCG", "c").view(np.uint8)] = np.arange(len("NAUCG"))
    seq = vocab[np.array(seq, "c").view(np.uint8)]  # type: ignore
    data = {"seq": torch.from_numpy(seq[None]).long()}  # type: ignore
    if cuda:
        model = model.cuda()
        data = {k: v.cuda() for k, v in data.items()}
    with torch.no_grad():
        output = model(data, inference_only=True)
    logits = output["contact_logits"]
    prob = torch.softmax(logits, dim=-1)[0, :, :, 1]
    prob = (prob + prob.transpose(-1, -2)) / 2
    prob = prob.cpu().numpy()
    return prob


def predict(seq: str, cuda, t: Perf, out_sub: Path):
    here = os.path.dirname(__file__)
    id = t.id

    with tempfile.TemporaryDirectory() as d:
        fgs = [[] for _ in range(5)]
        ptime = perf_counter_ns()
        for i in range(5):
            weight = "weights/prior_" + str(i) + ".pth"
            fgs[i] = inference(seq, weight, cuda)
        ptime = perf_counter_ns() - ptime
        t.inf_seq = ptime

        fg = np.mean(np.array(fgs, dtype=np.float64), axis=0)
        ptime = perf_counter_ns()
        bg = inference(seq, "weights/reference.pth", cuda)
        ptime = perf_counter_ns() - ptime
        t.inf_ref = ptime

        bppath = out_sub.joinpath(f"{id}_prior.mat")
        refpath = out_sub.joinpath(f"{id}_reference.mat")
        with open(bppath, "w") as fp:
            for i in range(fg.shape[0]):
                for j in range(fg.shape[0]):
                    fp.write("%.10f" % fg[i][j])
                    fp.write("\t")
                fp.write("\n")
            print(f"Saved bp probabilities to {bppath}")

        with open(refpath, "w") as fp:
            for i in range(bg.shape[0]):
                for j in range(bg.shape[0]):
                    fp.write("%.10f" % bg[i][j])
                    fp.write("\t")
                fp.write("\n")
            print(f"Saved reference probabilities to {refpath}")

        mincostflowcmd = f"{here}/KnotFold_mincostflow {bppath} {refpath}"
        ptime = perf_counter_ns()
        p = subprocess.run(mincostflowcmd, shell=True, capture_output=True)
        ptime = perf_counter_ns() - ptime
        t.min_cost_flow = ptime

        assert p.returncode == 0
        pairs = []
        for line in p.stdout.decode().split("\n"):
            if len(line) == 0:
                continue
            l, r = line.split()
            pairs.append((int(l), int(r)))
    return pairs, t


def write_bpseq(seq, pairs, outfile):
    bp = [-1 for _ in seq]
    for l, r in pairs:
        bp[l - 1] = r - 1
        bp[r - 1] = l - 1
    with open(outfile, "w") as fp:
        for i, k in enumerate(seq):
            fp.write("%d %s %d\n" % (i + 1, k, bp[i] + 1))


@click.command()
@click.option("-i", "--fasta", help="Input sequence file (fasta format)", required=True)
@click.option(
    "-o",
    "--outdir",
    help="Output dictionary. A subdirectory will be made for sequences from the input sequence file",
    default="./",
)
@click.option("--cuda", is_flag=True, default=True)
def main(fasta, outdir, cuda):
    fasta = Path(fasta)
    out_sub = Path(outdir).joinpath(f"{fasta.stem}")
    out_sub.mkdir(exist_ok=True, parents=True)

    task = open(fasta, "r").read().split("\n")

    for rec in parse(fasta, "fasta"):
        rec: SeqRecord
        name, seq = str(rec.id), str(rec.seq)
        pairs, t = predict(seq, cuda, t=Perf(len(seq), name), out_sub=out_sub)
        print(t)
        out_file_path = out_sub.joinpath(f"{name}.bpseq")
        write_bpseq(seq, pairs, out_file_path)


if __name__ == "__main__":
    main()
