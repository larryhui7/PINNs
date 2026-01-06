classdef AccumulateEdgesToNodeLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % AccumulateEdgesToNodeLayer   Accumulates edge features onto node features.
    methods
        function this = AccumulateEdgesToNodeLayer(args)
            arguments
                args.Name = ''
            end
            this.Name = args.Name;
            this.InputNames = ["node_embedding","edge_embedding","edges_of_node_list","mask"];
        end

        function nodeProcessorInput = predict(~,node_embeddings,edge_embeddings,edges_of_a_node_list,node_mask)
            % edge embeddings should be 0 on padded edges.
            all_mesh_sums = iMTimesSum(edge_embeddings,  edges_of_a_node_list);
            nodeProcessorInput = [ node_embeddings; all_mesh_sums];
            nodeProcessorInput = nodeProcessorInput.*node_mask(1,:,:);
        end
    end
end

function Z = iMTimesSum(X, Y)
% Input X: hidden nodes x batches x inputs to aggregate
% Input Y: inputs to aggregate x batches x outputs 
% Rearrange back to original dimensions to get edge-node rows-columns
X = permute(X,[1,3,2]); %CUB (place batches to the end)
Y = permute(Y,[1,3,2]); %UUB (UU = inputs x outputs x batches)
% Multiply per batch
Z = pagemtimes(X, Y); % CUB
% Reorder back
Z = permute(Z,[1,3,2]); % CBU
end