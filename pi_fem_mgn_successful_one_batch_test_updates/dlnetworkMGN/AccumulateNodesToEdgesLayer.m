classdef AccumulateNodesToEdgesLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % AccumulateNodesToEdgesLayer   Accumulates node features onto edge
    % features.
    methods
        function this = AccumulateNodesToEdgesLayer(args)
            arguments
                args.Name = ''
            end
            this.Name = args.Name;
            this.InputNames = ["edge_embedding","node_embedding","send_receive_list","mask"];
        end

        function edgeProcessorInput = predict(~,edge_embeddings,node_embeddings,srList,edge_mask)
            % Use sub2ind to convert the sr_list (list of nodes that define
            % the edges) into a linear index into the batch x node
            % dimension of the node_embeddings.
            %
            % The node_embeddings are 
            % node_embedding_size x batch x num_nodes
            % We want to index into that batch x num_nodes
            % Indexing into sr_list(i) goes:
            % sr_list(1,1,1), sr_list(2,1,1), sr_list(1,2,1), ..., sr_list(1,batchSize,1)
            % so on.
            % The corresponding batch indices are:
            % 1,1,2,2,...,batchSize,batchSize,1,1,2,2,...,batchSize,batchSize,...
            % repeated for each node.
            rows = 1:size(srList,2); % -> 1:batchsize
            rows = [rows;rows];
            rows = rows(:); % stack one after another (1, 1, 2, 2 ...)
            rows = repmat(rows,size(srList,3),1); % repeat by node number
            cols = srList(:); % stack sr_list the same manner
            idx = sub2ind(size(node_embeddings,2:3), rows, cols);
            % compute node embeddings of sender-receivers.
            sr_embeddings = node_embeddings(:, idx);
            % reshape the sender-receiver embeddings together, and unfold
            % the (batch*node) into batch x node.
            % so sr_embeddings is 
            % 2*node_embedding_size x batch_size x num_nodes.
            sr_embeddings = reshape(sr_embeddings, [size(node_embeddings,1)*2, size(srList,2:3)]);
            edgeProcessorInput = [edge_embeddings; sr_embeddings];
            % Mask out padding edges.
            edgeProcessorInput = edgeProcessorInput.*edge_mask(1,:,:);
        end
    end
end