import io
import os
import pandas as pd

def generateFASTAFromVCF(refFastaPath, vcfPath, metaPath):
    """
    Generate an effective FASTA file for simulation from a VCF file and a reference FASTA file.
    This function requires output from generateMetaFromVCF. 
    
    Parameters:
    refFastaPath (str): Path to the reference FASTA file.
    vcfPath (str): Path to the VCF file.
    metaPath (str): Path to the previously generated metadata file.
    """
    # Read the reference FASTA file 
    with open(refFastaPath, 'r') as f:
        ref_seq = ''.join([line.strip() for line in f if not line.startswith('>')])    
    
    # Read the vcf file to get the changed positions
    vcfTable = read_vcf(vcfPath)
    changedPositions = list(zip(vcfTable['CHROM'], vcfTable['POS']))
    
    # Read the metadata file to get the relevant positions
    with open(metaPath, 'r') as f:
        allPositions = [tuple(line.strip().split()) for line in f.readlines()]
    
    
    
    currIdx = 0
    finalFasta = ''
    for position in allPositions:
        if (currIdx < len(changedPositions) and position == changedPositions[currIdx]): 
            finalFasta += vcfTable['ALT'][currIdx].split(',')[0]  # Use the alternate allele from the VCF
            currIdx += 1
        else:
            finalFasta += ref_seq[int(position[1])]
        
            
    print(finalFasta)
        
    
    
    
    
    


def generateMetaFromVCF(vcfs, metaPath):
    """
    Generate a metadata file from VCF files.
    
    Parameters:
    vcfs (list): List of all used VCF file paths.
    refFastaPath (str): Path to the reference FASTA file.
    metaPath (str): Path to save the metadata file.
    """
    AllRelevantChroms = set()
    
    for vcfPath in vcfs:
        # Read the VCF file
        table = read_vcf(vcfPath)
        positions = list(zip(table['CHROM'], table['POS']))
        AllRelevantChroms.update(positions)
                
    sorted_chroms = sorted(AllRelevantChroms, key=lambda x: (x[0], int(x[1])))
    
    # Write sorted_chroms to metaPath, each pair on a row separated by a space
    with open(metaPath, 'w') as f:
        for chrom, pos in sorted_chroms:
            f.write(f"{chrom} {pos}\n")


def read_vcf(path):
    """
    Read a VCF file and return a DataFrame.
    Parameters:
    path (str): Path to the VCF file.
    Returns:
    pd.DataFrame: DataFrame containing the VCF data.
    """
    with open(path, 'r') as f:
        # Skip header lines starting with '##'  
        lines = []
        for line in f:
            if (not line.startswith('##')):
                lines.append(line)
                
        header = lines[0].strip().split()
        data = [line.strip().split() for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)
        df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
        return df
    
    

generateMetaFromVCF(['../data/test.vcf'], 'meta.txt')
generateFASTAFromVCF('../data/ref.fasta', '../data/test.vcf', 'meta.txt')
