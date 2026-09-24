function layer = flattenIntoBatchLayer(args)
% Custom functionLayer: "CBU" -> "CB" (all nodes become training batch data, 
% removes necessity to iterate accross batches during training)

arguments
    args.Name string = ''
end
layer = functionLayer(@flattenIntoBatch, Acceleratable=true, Formattable = true, Name = args.Name);
end

function x = flattenIntoBatch(x)
x = stripdims(x);
x = reshape(x, size(x,1), []);
x = dlarray(x, "CB");
end