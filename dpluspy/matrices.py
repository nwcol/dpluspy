"""
Matrix classes for storing different representations of sequence data.
"""

import numpy as np

from . import utils


class HaplotypeMatrix():
    """
    Wrapper class for potentially multiallelic haplotype matrices.
    """

    def __init__(
        self,
        haplotypes,
        positions,
        samples=None,
        populations=None
        ):
        self.haplotypes = np.asarray(haplotypes, dtype=np.int8)
        self.positions = np.asarray(positions, drype=np.int64)

        assert self.haplotypes.shape[1] % 2 == 0

        if samples is None:
            samples = list(range(self.n_samples))
        self.samples = samples

        if populations is None:
            populations = {"all": sample_names}
        self.populations = populations

    @property
    def n_samples(self):
        return int(self.haplotypes.shape[1] / 2)

    @property
    def n_haplotypes(self):
        return self.haplotypes.shape[1]

    @property
    def n_sites(self):
        return self.haplotypes.shape[0]

    @property
    def n_variant_sites(self):
        return np.sum(np.unique(self.haplotypes, axis=1) > 1)

    def slice_sample(self, sample):
        """Get the bare haplotype array for a specific sample."""
        idx = self.samples.index(sample)
        return self.haplotypes[:, 2 * idx:2 * (idx + 1)]

    def slice_population(self, population):
        """Get the bare haplotype array for a specific population."""
        samples = self.populations[population]
        idxs = []
        for sample in samples:
            idx = self.samples.index(sample)
            idxs += [2 * idx, 2 * (idx + 1)]
        return self.haplotypes[:, idxs]

    @classmethod
    def from_vcf(
        vcf_file,
        bed_file=None,
        pop_file=None,
        interval=None,
        apply_filter=False,
        ):
        """
        Load haplotypes from a VCF file.
        """
        haplotypes, positions, samples, populations = utils.read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            phased=True,
            interval=interval,
            apply_filter=apply_filter,
            )
        ret = cls(
            haplotypes,
            positions,
            samples=samples,
            populations=populations,
            )
        return ret



class GenotypeMatrix():
    """
    Wrapper class for biallelic genotype matrices.

    Parameters
    ----------
    genotypes : np.ndarray
        Shape (n_variants, n_diploids). Takes values 0, 1, 2, for homozygous
        reference, heterozygous, and homozygous alternate genotypes.
    """

    def __init__(
        self,
        genotypes,
        positions,
        samples=None,
        populations=None,
        ):
        self.genotypes = np.asarray(genotypes, dtype=np.int8)
        self.positions = np.asarray(positions, dtype=np.int64)

        if samples is None:
            n_samples = genotypes.shape[1]
            samples = list(range(n_samples))
        self.samples = samples

        if populations is None:
            populations = {"all": sample_names}
        self.populations = populations

    @property
    def n_sites(self):
        return self.genotypes.shape[0]

    def slice_sample(self, sample):
        """Get the genotype vector for a given sample."""
        idx = self.samples.index(sample)
        return self.genotypes[:, idx]

    def slice_population(self, population):
        """Get the genotype array for a given population."""
        samples = self.populations[population]
        idxs = [self.samples.index(sample) for sample in samples]
        return self.genotypes[:, idxs]

    @classmethod
    def from_vcf(
        vcf_file,
        bed_file=None,
        pop_file=None,
        interval=None,
        apply_filter=False,
        ):
        """
        Load genotypes from a VCF file.
        """
        genotypes, positions, samples, populations = utils.read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            phased=False,
            interval=interval,
            apply_filter=apply_filter,
            )
        ret = cls(
            genotypes,
            positions,
            samples=samples,
            populations=populations,
            )
        return ret

    @classmethod
    def from_tree_sequence():
        pass

    @classmethod
    def from_haplotype_matrix():
        pass


class GenoProbMatrix():
    """
    Wrapper for biallelic genotype probability (GP) matrices.

    Parameters
    ----------
    geno_probs : np.ndarray, shape (n_variants, 3 * n_diploids)
        For diploid i, columns i, i+1, i+2 hold P(aa), P(aA), P(AA), where A
        is the alternate allele, respectively.
    positions : np.ndarray, shape (n_variants,)
        0-indexed positions of sites in `geno_probs`.
    samples : list, optional
        List of sample names. Defaults to integers from 0 to `n_samples - 1`.
    populations : dict, optional
        Maps population names to lists of elements from `samples`.
    """

    def __init__(
        self,
        geno_probs,
        positions,
        samples=None,
        populations=None
        ):
        self.geno_probs = np.asarray(geno_probs, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.int64)

        assert self.geno_probs.shape[1] % 3 == 0

        if samples is None:
            samples = list(range(self.n_samples))
        self.samples = samples

        if populations is None:
            populations = {"all": sample_names}
        else:
            for population in populations:
                for sample in populations:
                    assert sample in populations[population]
        self.populations = populations

    @property
    def n_samples(self):
        return int(self.geno_probs.shape[1] / 3)

    @property
    def n_sites(self):
        return self.geno_probs.shape[0]

    def slice_sample(self, sample):
        """Get the bare genotype probability array for a specific sample."""
        idx = self.samples.index(sample)
        return self.geno_probs[:, 3 * idx:3 * (idx + 1)]

    def slice_population(self, population):
        """Get the bare genotype prob. array for a specific population."""
        samples = self.populations[population]
        idxs = []
        for sample in samples:
            idx = self.samples.index(sample)
            idxs += list(range(3 * idx, 3 * (idx + 1)))
        return self.geno_probs[:, idxs]

    @classmethod
    def from_vcf(
        cls,
        vcf_file,
        bed_file=None,
        pop_file=None,
        interval=None,
        apply_filter=False,
        ):
        """
        Load genotype probabilities from a VCF file.
        """
        geno_probs, positions, samples, populations = utils.read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            read_gps=True,
            interval=interval,
            apply_filter=apply_filter,
            )
        ret = cls(
            geno_probs,
            positions,
            samples=samples,
            populations=populations,
            )
        return ret

