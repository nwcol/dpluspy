"""
Tests for the `parsing` module.
"""

import numpy as np
import unittest
import os

from dpluspy import utils, parsing 


class TestMaps(unittest.TestCase):

    def test_loading_uniform_hapmap_map(self):
        filepath = os.path.join(os.path.dirname(__file__),
            'test_files/uniform_recmap.txt')
        rec_map = parsing._load_recombination_map(filepath)
        self.assertEqual(rec_map(0), 0)
        self.assertEqual(rec_map(1), 0)
        self.assertAlmostEqual(rec_map(5), 5e-8)
        # Beyond map ends, the map is assigned to its last value.
        self.assertAlmostEqual(rec_map(6), 5e-8)

    def test_loading_uniform_bedgraph_map(self):
        pass

    def test_loading_heterog_hapmap_map(self):
        filepath = os.path.join(os.path.dirname(__file__),
            'test_files/heterogeneous_recmap.txt')
        rec_map = parsing._load_recombination_map(filepath)
        self.assertEqual(rec_map(0), 0)
        self.assertEqual(rec_map(1), 0)
        self.assertAlmostEqual(rec_map(2), 1e-8)
        self.assertAlmostEqual(rec_map(3), 3e-8)
        self.assertAlmostEqual(rec_map(4), 6.5e-8)
        self.assertAlmostEqual(rec_map(5), 1e-7)
        self.assertAlmostEqual(rec_map(6), 1e-7)

    def test_uniform_map(self):
        recmap = parsing._get_uniform_recombination_map(1.5e-8, 1e7)
        unit_rate = utils._map_function(1.5e-8)
        self.assertAlmostEqual(recmap(1), 1 * unit_rate)
        self.assertAlmostEqual(recmap(1e7), 1e7 * unit_rate)
        self.assertAlmostEqual(recmap(1e6), 1e6 * unit_rate)
        

class TestReadGenotypes(unittest.TestCase):
    ## Tests of VCF loader options

    def test_default_genotype_reading(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/3_sample.vcf')
        expected_sites = np.array([1, 2, 3, 4, 5])
        expected_genotypes = np.array(
            [[[0, 1], [0, 1], [0, 0]],
             [[0, 1], [0, 1], [0, 1]],
             [[1, 1], [0, 0], [1, 1]],
             [[0, 1], [0, 1], [0, 0]],
             [[0, 1], [0, 1], [0, 1]]]
        )
        expected_sample_ids = ['sample1', 'sample2', 'sample3']
        sites, genotypes, sample_ids = parsing._read_genotypes(vcf_file)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        self.assertEqual(sample_ids, expected_sample_ids)

    def test_masked_genotype_reading(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/3_sample.vcf')
        bed_file = os.path.join(os.path.dirname(__file__),
            'test_files/mask_1_3_5.bed')
        expected_sites = np.array([1, 3, 5])
        expected_genotypes = np.array(
            [[[0, 1], [0, 1], [0, 0]],
             [[1, 1], [0, 0], [1, 1]],
             [[0, 1], [0, 1], [0, 1]]]
        )
        expected_sample_ids = ['sample1', 'sample2', 'sample3']
        sites, genotypes, sample_ids = parsing._read_genotypes(
            vcf_file, bed_file=bed_file)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        self.assertEqual(sample_ids, expected_sample_ids)
        # Intervals
        interval = (1, 3)
        expected_sites = np.array([1])
        expected_genotypes = np.array([[[0, 1], [0, 1], [0, 0]]])
        expected_sample_ids = ['sample1', 'sample2', 'sample3']
        sites, genotypes, sample_ids = parsing._read_genotypes(
            vcf_file, bed_file=bed_file, interval=interval)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        self.assertEqual(sample_ids, expected_sample_ids)
        interval = (3, 6)
        expected_sites = np.array([3, 5])
        expected_genotypes = np.array(
            [[[1, 1], [0, 0], [1, 1]],
             [[0, 1], [0, 1], [0, 1]]]
        )
        expected_sample_ids = ['sample1', 'sample2', 'sample3']
        sites, genotypes, sample_ids = parsing._read_genotypes(
            vcf_file, bed_file=bed_file, interval=interval)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        self.assertEqual(sample_ids, expected_sample_ids)

    def test_missing_genotype_data(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/missing_data.vcf')  
        expected_sites = np.array([1])
        expected_genotypes = np.array([[[1, 1]]])
        sites, genotypes, _ = parsing._read_genotypes(vcf_file)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        expected_sites = np.array([1, 2, 3])
        expected_genotypes = np.array([[[1, 1]], [[1, 0]], [[0, 0]]])
        sites, genotypes, _ = parsing._read_genotypes(
            vcf_file, missing_to_ref=True)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))

    def test_filtered_genotype_data(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/filter_pass.vcf')  
        expected_sites = np.array([2])
        expected_genotypes = np.array([[[0, 1]]])
        sites, genotypes, _ = parsing._read_genotypes(vcf_file, filtered=True)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))

    def test_invalid_genotype_data(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/invalid_genotypes.vcf')  
        expected_sites = np.array([3])
        expected_genotypes = np.array([[[0, 1]]])
        sites, genotypes, _ = parsing._read_genotypes(vcf_file)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        
    def test_multiallelic_genotype_data(self):
        vcf_file = os.path.join(os.path.dirname(__file__),
            'test_files/multiallelic.vcf')  
        # Default
        expected_sites = np.array([1])
        expected_genotypes = np.array([[[1, 1]]])
        sites, genotypes, _ = parsing._read_genotypes(vcf_file)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))
        # Allow multiallelic
        expected_sites = np.array([1, 2, 3])
        expected_genotypes = np.array([[[1, 1]], [[1, 2]], [[0, 2]]])
        sites, genotypes, _ = parsing._read_genotypes(
            vcf_file, multiallelic=True)
        self.assertTrue(genotypes.shape == expected_genotypes.shape)
        self.assertTrue(np.all(sites == expected_sites))
        self.assertTrue(np.all(genotypes == expected_genotypes))


class TestGenotypeEstimators(unittest.TestCase):

    def test_genotype_within(self):    
        # Single-haplotype tests
        map1 = np.array([0, 0])
        bins1 = np.array([0, 1])
        gt1 = np.array([[[1, 1]], [[1, 1]]])
        gt2 = np.array([[[1, 1]], [[1, 0]]])
        gt3 = np.array([[[1, 1]], [[0, 0]]])
        gt4 = np.array([[[1, 0]], [[1, 1]]])
        gt5 = np.array([[[1, 0]], [[1, 0]]])
        gt6 = np.array([[[1, 0]], [[0, 0]]])
        gt7 = np.array([[[0, 0]], [[1, 1]]])
        gt8 = np.array([[[0, 0]], [[1, 0]]])
        gt9 = np.array([[[0, 0]], [[0, 0]]])
        result = parsing._genotype_Dplus(gt1, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt2, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt3, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt4, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt5, map1, bins1)
        self.assertTrue(np.all(result == [1]))  
        result = parsing._genotype_Dplus(gt6, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._genotype_Dplus(gt7, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt8, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus(gt9, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        # Multiallelic genotypes  
        genotypes = np.array([[[0, 1]], [[2, 1]]])
        result = parsing._genotype_Dplus(genotypes, map1, bins1)
        self.assertTrue(np.all(result == [1]))  
        genotypes = np.array([[[0, 1]], [[0, 2]]])
        result = parsing._genotype_Dplus(genotypes, map1, bins1)
        self.assertTrue(np.all(result == [1]))  
        genotypes = np.array([[[2, 2]], [[1, 1]]])
        result = parsing._genotype_Dplus(genotypes, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        genotypes = np.array([[[0, 0]], [[2, 2]]])
        result = parsing._genotype_Dplus(genotypes, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        # More sites       
        bins = np.array([0, 0.75, 1.5, 4, 10])
        map6 = np.array([1, 2, 4, 4.25, 4.5, 6])
        genotypes1 = np.array(
            [[[0, 1]], [[0, 1]], [[0, 1]], [[0, 1]], [[0, 0]], [[0, 1]]])
        genotypes2 = np.array(
            [[[0, 1]], [[0, 1]], [[0, 1]], [[0, 1]], [[0, 0]], [[0, 0]]])
        result = parsing._genotype_Dplus(genotypes1, map6, bins)
        expected = np.array([1, 1, 6, 2])
        self.assertTrue(np.all(result == expected))
        # Two diploids
        genotypes12 = np.concatenate((genotypes1, genotypes2), axis=1)
        result = parsing._genotype_Dplus(genotypes12, map6, bins)
        expected = np.array([1, 1, 5, 1])
        self.assertTrue(np.all(result == expected))   

    def test_genotype_between(self):
        # Single-haplotype tests
        map1l = np.array([0])
        map1r = np.array([1])
        bins1 = np.array([0, 2])
        gt1 = np.array([[[1, 1]]])
        gt2 = np.array([[[1, 0]]])
        gt3 = np.array([[[0, 0]]])
        result = parsing._genotype_Dplus_between(gt1, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt1, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt1, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt2, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt2, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [1]))  
        result = parsing._genotype_Dplus_between(gt2, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt3, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt3, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._genotype_Dplus_between(gt3, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))   
   
    def test_cross_genotype_within(self):
        # Single-haplotype tests (not exhaustive)
        map1 = np.array([0, 0])
        bins1 = np.array([0, 1])
        gt1 = np.array([[[1, 1]], [[1, 1]]])
        gt2 = np.array([[[1, 1]], [[1, 0]]])
        gt3 = np.array([[[1, 1]], [[0, 0]]])
        gt4 = np.array([[[1, 0]], [[1, 1]]])
        gt5 = np.array([[[1, 0]], [[1, 0]]])
        gt6 = np.array([[[1, 0]], [[0, 0]]])
        gt7 = np.array([[[0, 0]], [[1, 1]]])
        gt8 = np.array([[[0, 0]], [[1, 0]]])
        gt9 = np.array([[[0, 0]], [[0, 0]]])
        result = parsing._cross_genotype_Dplus(gt1, gt1, map1, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._cross_genotype_Dplus(gt9, gt9, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._cross_genotype_Dplus(gt1, gt9, map1, bins1)
        self.assertTrue(np.all(result == [1]))  
        result = parsing._cross_genotype_Dplus(gt9, gt1, map1, bins1)
        self.assertTrue(np.all(result == [1]))  
        result = parsing._cross_genotype_Dplus(gt5, gt5, map1, bins1)
        self.assertTrue(np.all(result == [0.25]))  
        result = parsing._cross_genotype_Dplus(gt1, gt6, map1, bins1)
        self.assertTrue(np.all(result == [0.5]))  
        result = parsing._cross_genotype_Dplus(gt6, gt1, map1, bins1)
        self.assertTrue(np.all(result == [0.5]))  
        result = parsing._cross_genotype_Dplus(gt6, gt8, map1, bins1)
        self.assertTrue(np.all(result == [0.25]))  
        result = parsing._cross_genotype_Dplus(gt8, gt6, map1, bins1)
        self.assertTrue(np.all(result == [0.25]))  
        result = parsing._cross_genotype_Dplus(gt3, gt4, map1, bins1)
        self.assertTrue(np.all(result == [0.5]))  
        result = parsing._cross_genotype_Dplus(gt4, gt3, map1, bins1)
        self.assertTrue(np.all(result == [0.5]))  

    def test_cross_genotype_between(self):
        # Single-haplotype tests
        map1l = np.array([0])
        map1r = np.array([1])
        bins1 = np.array([0, 2])
        gt1 = np.array([[[1, 1]]])
        gt2 = np.array([[[1, 0]]])
        gt3 = np.array([[[0, 0]]])
        result = parsing._cross_genotype_Dplus_between(
            gt1, gt1, gt1, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))  
        result = parsing._cross_genotype_Dplus_between(
            gt3, gt3, gt3, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0]))    
        result = parsing._cross_genotype_Dplus_between(
            gt1, gt3, gt1, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [1]))    
        result = parsing._cross_genotype_Dplus_between(
            gt3, gt1, gt3, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [1]))      
        result = parsing._cross_genotype_Dplus_between(
            gt2, gt2, gt2, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.25]))    
        result = parsing._cross_genotype_Dplus_between(
            gt1, gt2, gt1, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))    
        result = parsing._cross_genotype_Dplus_between(
            gt2, gt1, gt3, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))    
        result = parsing._cross_genotype_Dplus_between(
            gt3, gt1, gt2, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))    
        result = parsing._cross_genotype_Dplus_between(
            gt2, gt1, gt1, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.25]))   
        result = parsing._cross_genotype_Dplus_between(
            gt1, gt2, gt2, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.25]))   
        result = parsing._cross_genotype_Dplus_between(
            gt1, gt2, gt3, gt1, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))    
        result = parsing._cross_genotype_Dplus_between(
            gt3, gt1, gt1, gt2, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))   
        result = parsing._cross_genotype_Dplus_between(
            gt2, gt1, gt1, gt3, map1l, map1r, bins1)
        self.assertTrue(np.all(result == [0.5]))  


class TestHaplotypeEstimators(unittest.TestCase):

    def test_haplotype_within(self):
        # Single haplotypes. arbitrarily, here 1=AB 2=Ab 3=aB 4=ab
        map1 = np.array([0, 0])
        bins1 = np.array([0, 1])
        h11 = np.array([[1, 1], [1, 1]])
        h12 = np.array([[1, 1], [1, 0]])
        h13 = np.array([[1, 0], [1, 1]])
        h14 = np.array([[1, 0], [1, 0]])
        h21 = np.array([[1, 1], [0, 1]])
        h22 = np.array([[1, 1], [0, 0]])
        h23 = np.array([[1, 0], [0, 1]])
        h24 = np.array([[1, 0], [0, 0]])
        h31 = np.array([[0, 1], [1, 1]])
        h32 = np.array([[0, 1], [1, 0]])
        h33 = np.array([[0, 0], [1, 1]])
        h34 = np.array([[0, 0], [1, 0]])
        h41 = np.array([[0, 1], [0, 1]])
        h42 = np.array([[0, 1], [0, 0]])
        h43 = np.array([[0, 0], [0, 1]])
        h44 = np.array([[0, 0], [0, 0]])
        result = parsing._haplotype_Dplus(h11, map1, bins1)
        self.assertTrue(np.all(result == [0])) 
        result = parsing._haplotype_Dplus(h12, map1, bins1)
        self.assertTrue(np.all(result == [0])) 
        result = parsing._haplotype_Dplus(h13, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h14, map1, bins1)
        self.assertTrue(np.all(result == [1]))
        result = parsing._haplotype_Dplus(h21, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h22, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h23, map1, bins1)
        self.assertTrue(np.all(result == [1]))
        result = parsing._haplotype_Dplus(h24, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h31, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h32, map1, bins1)
        self.assertTrue(np.all(result == [1]))
        result = parsing._haplotype_Dplus(h33, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h34, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h41, map1, bins1)
        self.assertTrue(np.all(result == [1]))
        result = parsing._haplotype_Dplus(h42, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h43, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        result = parsing._haplotype_Dplus(h44, map1, bins1)
        self.assertTrue(np.all(result == [0]))
        # Multiallelic haplotypes
        haplotypes = np.array([[1, 0], [0, 2]])
        result = parsing._haplotype_Dplus(haplotypes, map1, bins1)
        self.assertTrue(np.all(result == [1])) 
        haplotypes = np.array([[1, 2], [1, 2]])
        result = parsing._haplotype_Dplus(haplotypes, map1, bins1)
        self.assertTrue(np.all(result == [1])) 

    def test_haplotype_between(self):
        pass

    def test_cross_haplotype(self):
        pass

    def test_cross_haplotype_between(self):
        pass


class TestSumParsing(unittest.TestCase):

    def test_default_sum_parsing(self):
        pass 

    def test_two_pop_sum_parsing(self):
        pass

    def test_masked_sum_parsing(self):
        pass

    def test_sum_parsing_between(self):
        pass

    def test_sum_parsing_between_masked(self):
        pass


class TestDenomParsing(unittest.TestCase):

    def test_default_denom_parsing(self):
        pass

    def test_intervaled_denom_parsing(self):
        pass

    def test_maskless_uniform_denom_parsing(self):
        # This usage is targeted at simulated data
        unif_map = np.arange(100) * utils._map_function(1e-7)
        bins = np.logspace(-6, -4, 10)
        binsM = utils._map_function(bins)
        expected = np.append(_count_pairs(unif_map, binsM), 100)
        result = parsing.compute_denominators(
            interval=(1, 101), r=1e-7, r_bins=bins)
        self.assertTrue(np.all(result == expected))


class TestMutFacParsing(unittest.TestCase):

    def test_default_mut_fac_parsing(self):
        bedgraph_file = os.path.join(os.path.dirname(__file__),
            'test_files/mutation_map.bedgraph')
        npy_file = os.path.join(os.path.dirname(__file__),
            'test_files/mutation_map.npy')
        unif_map = os.path.join(os.path.dirname(__file__),
            'test_files/uniform_recmap.txt')
        het_map = os.path.join(os.path.dirname(__file__),
            'test_files/heterogeneous_recmap.txt')
        bins = np.array([0, 1e-8, 4e-8, 6e-8, 1e-7])

        expected = np.array([[3.4875e-16, 2.725e-16, 0, 0, 4.1e-8]])
        result = parsing.compute_mutation_factors(
            bedgraph_file, r=1e-8, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, r=1e-8, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            bedgraph_file, rec_map_file=unif_map, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, rec_map_file=unif_map, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))

        expected = np.array([1.25e-16, 3.4875e-16, 6.25e-17, 8.5e-17, 4.1e-8])
        result = parsing.compute_mutation_factors(
            bedgraph_file, rec_map_file=het_map, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, rec_map_file=het_map, interval=(1, 6), r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        
    def test_masked_mut_fac_parsing(self):
        mask_file = os.path.join(os.path.dirname(__file__),
            'test_files/mask_1_3_5.bed')
        bedgraph_file = os.path.join(os.path.dirname(__file__),
            'test_files/mutation_map.bedgraph')
        npy_file = os.path.join(os.path.dirname(__file__),
            'test_files/mutation_map.npy')
        unif_map = os.path.join(os.path.dirname(__file__),
            'test_files/uniform_recmap.txt')
        het_map = os.path.join(os.path.dirname(__file__),
            'test_files/heterogeneous_recmap.txt')
        bins = np.array([0, 1e-8, 4e-8, 6e-8, 1e-7])

        expected = np.array([[0, 1.475e-16, 0, 0, 2.35e-8]])
        result = parsing.compute_mutation_factors(
            bedgraph_file, r=1e-8, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, r=1e-8, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(bedgraph_file, 
            rec_map_file=unif_map, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, rec_map_file=unif_map, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))

        expected = np.array([0, 1.25e-16, 0, 2.25e-17, 2.35e-8])
        result = parsing.compute_mutation_factors(
            bedgraph_file, rec_map_file=het_map, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = parsing.compute_mutation_factors(
            npy_file, rec_map_file=het_map, bed_file=mask_file, r_bins=bins
        )[0]
        self.assertTrue(np.all(np.isclose(result, expected)))


## "Naive" pair-counting functions to check the fast vectorized ones.


def _count_pairs(site_map, bins, weights=None):
    """
    Get binned pair counts with a naive loop.
    """
    if weights is None:
        weights = np.ones(len(site_map))
    count = np.zeros(len(bins) - 1)
    for i, (map_l, weight_l) in enumerate(zip(site_map[:-1], weights[:-1])):
        for map_r, weight_r in zip(site_map[i + 1:], weights[i + 1:]):
            distance = map_r - map_l
            if distance >= bins[0] and distance < bins[-1]:
                idx = np.digitize(distance, bins) - 1
                count[idx] += weight_l * weight_r
 
    return count
 

def _count_pairs_between(
    site_map_l, 
    site_map_r, 
    bins, 
    weights_l=None,
    weights_r=None
):
    """
    Get binned pair counts between two discontinuous map segments with a naive 
    loop.
    """
    site_map_l = np.asarray(site_map_l)
    site_map_r = np.asarray(site_map_r)
    assert len(site_map_l) > 1
    assert len(site_map_r) > 1
    if weights_l is not None:
        assert len(site_map_l) == len(weights_l)
        assert len(site_map_r) == len(weights_r)
    assert np.all(site_map_l < site_map_r[0])
    if weights_l is None:
        weights_l = np.ones(len(site_map_l))
    if weights_r is None:
        weights_r = np.ones(len(site_map_r))
    count = np.zeros(len(bins) - 1)
    for map_l, weight_l in zip(site_map_l, weights_l):
        for map_r, weight_r in zip(site_map_r, weights_r):
            distance = map_r - map_l
            if distance >= bins[0] and distance < bins[-1]:
                idx = np.digitize(distance, bins) - 1
                count[idx] += weight_l * weight_r
    
    return count 


def sample_map(r, L):
    """
    Construct a random recombination map with length `L` and average rate `r`.
    """
    return np.cumsum(np.random.uniform(0, r * 2, L))


class TestLocusPairCounting(unittest.TestCase):

    def test_counting_within(self):
        pass

    def test_weighted_counting_within(self):
        pass

    def test_counting_between(self):
        pass

    def test_weighted_counting_between(self):
        pass

    def test_count_locus_pairs_exceptions(self):
        # Maps with zero values return zero sums
        bins = np.logspace(-6, -1, 10)
        empty = np.zeros(len(bins) - 1)
        result = parsing._count_locus_pairs(np.array([]), bins)
        self.assertTrue(np.all(empty == result))
        # Map/weight length mismatch raises an error
        with self.assertRaises(ValueError):
            site_map = np.array([0, 1e-6, 1e-4])
            weights = np.array([1, 1])
            parsing._count_locus_pairs(site_map, bins, weights=weights)
        # Maps must be monotonically increasing
        with self.assertRaises(ValueError):
            site_map = np.array([0, 1e-6, 1e-4, 1e-8])
            parsing._count_locus_pairs(site_map, bins)

    def test_count_locus_pairs_between_exceptions(self):
        bins = np.logspace(-6, -1, 10)
        empty = np.zeros(len(bins) - 1)
        result = parsing._count_locus_pairs_between(
            np.array([]), np.array([]), bins)
        self.assertTrue(np.all(empty == result))
        # Giving weights for one window only raises an error
        site_map_l = np.array([0, 1e-7])
        site_map_r = np.array([1e-6, 2e-6])
        weights = np.array([1, 0])
        with self.assertRaises(ValueError):
            parsing._count_locus_pairs_between(
                site_map_l, site_map_r, bins, weights_l=weights)
        # Map/length mismatches raise errors
        mismatch = np.array([1])
        with self.assertRaises(ValueError):
            parsing._count_locus_pairs_between(site_map_l, site_map_r, bins, 
                weights_l=weights, weights_r=mismatch)
        with self.assertRaises(ValueError):
            mismatch = np.array([1])
            parsing._count_locus_pairs_between(site_map_l, site_map_r, bins, 
                weights_l=mismatch, weights_r=mismatch)
        # The right map must have higher coords than the left
        mis_map = np.array([5e-8, 2e-7])
        with self.assertRaises(ValueError):
            parsing._count_locus_pairs_between(site_map_l, mis_map, bins)

    ## Tests below randomly sample maps, weights.

    def test_counting_within_stochastic(self):
        bins = np.concatenate(([0], np.logspace(-6, -1, 10)))
        site_map = sample_map(1e-8, 100)
        result = parsing._count_locus_pairs(site_map, bins)
        naive = _count_pairs(site_map, bins)
        self.assertTrue(np.all(result == naive))

        bins = np.logspace(-6, -1, 10)
        result = parsing._count_locus_pairs(site_map, bins)
        naive = _count_pairs(site_map, bins)
        self.assertTrue(np.all(result == naive))

        site_map = sample_map(1e-6, 100)
        bins = np.logspace(-6, -1, 10)
        result = parsing._count_locus_pairs(site_map, bins)
        naive = _count_pairs(site_map, bins)
        self.assertTrue(np.all(result == naive))

        site_map = sample_map(1e-2, 200)
        bins = np.logspace(-6, -1, 10)
        result = parsing._count_locus_pairs(site_map, bins)
        naive = _count_pairs(site_map, bins)
        self.assertTrue(np.all(result == naive))

        bins = np.logspace(-6, -0.35, 10)
        result = parsing._count_locus_pairs(site_map, bins)
        naive = _count_pairs(site_map, bins)
        self.assertTrue(np.all(result == naive))

    def test_counting_between_stochastic(self):
        bins0 = np.concatenate(([0], np.logspace(-6, -1, 10)))
        bins = np.logspace(-6, -1, 10)

        full_map = sample_map(1e-8, 200)
        left_map, right_map = full_map[:100], full_map[100:]
        result = parsing._count_locus_pairs_between(left_map, right_map, bins0)
        naive = _count_pairs_between(left_map, right_map, bins0)
        self.assertTrue(np.all(result == naive))
        result = parsing._count_locus_pairs_between(left_map, right_map, bins)
        naive = _count_pairs_between(left_map, right_map, bins)
        self.assertTrue(np.all(result == naive))
        # Longer map distances
        full_map = sample_map(1e-4, 200)
        left_map, right_map = full_map[:100], full_map[100:]
        result = parsing._count_locus_pairs_between(left_map, right_map, bins0)
        naive = _count_pairs_between(left_map, right_map, bins0)
        self.assertTrue(np.all(result == naive))
        result = parsing._count_locus_pairs_between(left_map, right_map, bins)
        naive = _count_pairs_between(left_map, right_map, bins)
        self.assertTrue(np.all(result == naive))
        # Large seperation between segments
        right_map += 0.03
        result = parsing._count_locus_pairs_between(left_map, right_map, bins0)
        naive = _count_pairs_between(left_map, right_map, bins0)
        self.assertTrue(np.all(result == naive))
        result = parsing._count_locus_pairs_between(left_map, right_map, bins)
        naive = _count_pairs_between(left_map, right_map, bins)
        self.assertTrue(np.all(result == naive))
        # Very large seperation between segments
        right_map += 0.30
        result = parsing._count_locus_pairs_between(left_map, right_map, bins0)
        naive = _count_pairs_between(left_map, right_map, bins0)
        self.assertTrue(np.all(result == naive))
        result = parsing._count_locus_pairs_between(left_map, right_map, bins)
        naive = _count_pairs_between(left_map, right_map, bins)
        self.assertTrue(np.all(result == naive))

    def test_weighted_counting_within_stochastic(self):
        # Because cumulative sums are involved, some small loss of precision is 
        # entailed; I use `np.isclose` rather than testing for equality
        weights = np.random.uniform(0, 1, 100)
        bins = np.concatenate(([0], np.logspace(-6, -1, 10)))
        site_map = sample_map(1e-8, 100)
        result = parsing._count_locus_pairs(site_map, bins, weights=weights)
        naive = _count_pairs(site_map, bins, weights=weights)
        self.assertTrue(np.all(np.isclose(result, naive)))

        bins = np.logspace(-6, -1, 10)
        site_map = sample_map(1e-8, 100)
        result = parsing._count_locus_pairs(site_map, bins, weights=weights)
        naive = _count_pairs(site_map, bins, weights=weights)
        self.assertTrue(np.all(np.isclose(result, naive)))

        bins = np.concatenate(([0], np.logspace(-6, -1, 10)))
        site_map = sample_map(1e-3, 100)
        result = parsing._count_locus_pairs(site_map, bins, weights=weights)
        naive = _count_pairs(site_map, bins, weights=weights)
        self.assertTrue(np.all(np.isclose(result, naive)))
        
        weights = np.random.choice([0, 1], size=200)
        bins = np.concatenate(([0], np.logspace(-6, -1, 10)))
        site_map = sample_map(1e-8, 200)
        result = parsing._count_locus_pairs(site_map, bins, weights=weights)
        naive = _count_pairs(site_map, bins, weights=weights)
        self.assertTrue(np.all(np.isclose(result, naive)))

        site_map = sample_map(1e-3, 200)
        result = parsing._count_locus_pairs(site_map, bins, weights=weights)
        naive = _count_pairs(site_map, bins, weights=weights)
        self.assertTrue(np.all(np.isclose(result, naive)))

    def test_weighted_counting_between_stochastic(self):
        bins0 = np.concatenate(([0], np.logspace(-6, -1, 10)))
        bins = np.logspace(-6, -1, 10)

        full_map = sample_map(1e-8, 200)
        left_map, right_map = full_map[:100], full_map[100:]
        weightsl = np.random.uniform(0, 1, 100)
        weightsr = np.random.uniform(0, 1, 100)
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))

        weightsl = np.random.choice([0, 1], size=100)
        weightsr = np.random.choice([0, 1], size=100)
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))

        full_map = sample_map(1e-4, 200)
        left_map, right_map = full_map[:100], full_map[100:]
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))

        full_map = sample_map(1e-3, 200)
        left_map, right_map = full_map[:100], full_map[100:]
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        # All pairs are out of bin range
        right_map += 0.30
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins0, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))
        result = parsing._count_locus_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        naive = _count_pairs_between(
            left_map, right_map, bins, weights_l=weightsl, weights_r=weightsr)
        self.assertTrue(np.all(np.isclose(result, naive)))

