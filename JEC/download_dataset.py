# Energy-flow package for CMS Open Data loader
import energyflow as ef
from energyflow.archs import PFN, EFN
from energyflow.utils import remap_pids


# ################################
# ########## PARAMETERS ##########
# ################################


cache_dir = "data"   # Where to save data (note "/datasets" is appended to the end)
amount = 1       # Fraction of files to download (if integer, # of files)

collection ='CMS2011AJets'
dataset='sim'
subdatasets = None                    
validate_files = True # Keep true if downloading for the first time
store_pfcs = True
store_gens=True
verbose = 2     

# ##############################
# ########## DOWNLOAD ##########
# ##############################


ef.mod.load(amount = amount,
                    cache_dir = cache_dir, 
                    collection = collection,
                    dataset = dataset,
                    subdatasets = subdatasets,
                    validate_files = validate_files,
                    store_pfcs = store_pfcs,
                    store_gens = store_gens,
                    verbose = 2,
)