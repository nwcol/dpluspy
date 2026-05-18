"""
For manipulating and representing tskit tree sequences created using msprime.
"""

import numpy as np
import tskit


def write_vcf():
    """
    Write a synthetic VCF or GVCF file representing one or more samples from
    a tskit tree sequence.
    """
    # TODO write this function.

    ref_seq = ref.replace("\n", "").replace(">0", "")
    hap_0_seq = hap_0.replace("\n", "").replace(">0", "")
    hap_1_seq = hap_1.replace("\n", "").replace(">0", "")

    with open("data/true_genotypes.vcf", "w") as fout:
        fout.write("##DUMMY\n")
        header = "\t".join([
            "CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "FILTER",
            "INFO",
            "FORMAT",
            "SAMPLE"]) + "\n"
        fout.write(header)
        for i in range(L):
            ref_allele = ref_seq[i]
            allele0 = hap_0_seq[i]
            allele1 = hap_1_seq[i]
            if allele0 == allele1:
                if allele0 == ref_allele:
                    alt_allele = "."
                    gt = "0/0"
                else:
                    alt_allele = allele0
                    gt = "1/1"
            else:
                if allele0 == ref_allele:
                    alt_allele = allele1
                    gt = "0/1"
                elif allele1 == ref_allele:
                    alt_allele = allele0
                    gt = "0/1"
                else:
                    alt_allele = ",".join([allele0, allele1])
                    gt = "1/2"
            line = "\t".join([str(x) for x in [
                0,
                i + 1,
                ".",
                ref_allele,
                alt_allele,
                ".",
                ".",
                "GT",
                gt
            ]]) + "\n"
            fout.write(line)
    return

