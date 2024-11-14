import pickle
import numpy as np
import torch
import os
from gears.model import GEARS_Model
from gears.utils import get_similarity_network, create_cell_graph_dataset_for_prediction, create_cell_graph_for_prediction, GeneSimNetwork
from torch_geometric.data import DataLoader

class InferencePertData:
    """
    Just bits of pertdata which are required at inference time
    """
    def __init__(self, pert_data=None):
        # if given pertdata object filter to just include inference requirements
        obs = pert_data.adata.obs
        indices = obs.reset_index(drop=True).query('control==1').index.values
        X = pert_data.adata.X[indices]
        self.X = X
        self.obs = obs.query('control==1')
        self.ensembl_id_list = pert_data.gene_names.index.tolist()
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            instance = pickle.load(file)
        return instance
    
def create_cell_graph_dataset_for_inference(pert_gene, ctrl_X, gene_names,
                                             device, num_samples = 300):
    # Get the indices (and signs) of applied perturbation
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_X[np.random.randint(0, ctrl_X.shape[0], num_samples), :].toarray()
    # Create cell graphs
    cell_graphs = [create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs]
    return cell_graphs


class GEARSInference:
    """
    GEARS inference only model class
    """

    def __init__(self, inference_data_path, 
                 device = 'cuda'):
        
        self.device = device
        self.config = None
        inference_data = InferencePertData.load(inference_data_path)
        self.ctrl_X = inference_data.X[inference_data.obs.condition == 'ctrl'] # should only be control in file but add this filter just in case
        self.ctrl_mean = np.mean(self.ctrl_X, axis=0)
        self.ensembl_id_list = inference_data.ensembl_id_list
        self.gene_list = inference_data.gene_list
        self.pert_list = inference_data.pert_list
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)

    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb
                      }
        
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
    
    def predict(self, pert_list):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.model = self.model.to(self.device)
        self.model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        for pert in pert_list:
            
            cg = create_cell_graph_dataset_for_inference(pert, self.ctrl_X,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
             
                
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
    