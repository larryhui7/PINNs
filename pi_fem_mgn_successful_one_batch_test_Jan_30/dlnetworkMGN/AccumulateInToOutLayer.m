classdef AccumulateInToOutLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % AccumulateEdgesToNodeLayer   Accumulates edge features onto node
    % features.
    methods
        function this = AccumulateInToOutLayer(args)
            arguments                
                args.Name = ''
            end
            this.Name = args.Name;
            this.InputNames = ["in_embeddings","out_embeddings","in_to_out_matrix","out_mask"];
        end

        function aggregationStep = predict(~, out_embedding, in_embeddings, in_to_out_matrix, out_mask)
            % edge embeddings should be 0 on padded edges.
            all_in_sums = iMTimesSum(in_embeddings,  in_to_out_matrix);
            aggregationStep = [out_embedding; all_in_sums];
            aggregationStep = aggregationStep.*out_mask(1,:,:);
        end
    end
end

function Z = iMTimesSum(X, Y)
% Rearrange back to original dimensions to get edge-node rows-columns
X = permute(X,[1,3,2]); %CUB
Y = permute(Y,[1,3,2]); %UUB (UU = edges x nodes)
% Multiply per batch
Z = pagemtimes(X, Y); % CUB
% Reorder back
Z = permute(Z,[1,3,2]); % CBU
end