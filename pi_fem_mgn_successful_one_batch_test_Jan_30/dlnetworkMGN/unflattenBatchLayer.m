function layer = unflattenBatchLayer(args)
arguments
    args.Name string = ''
    args.USize (1,1) double {mustBeInteger, mustBePositive} = 1
end
layer = functionLayer(@(x)unflattenBatch(x,args.USize), Acceleratable = true, Formattable = true, Name = args.Name);
end

function x = unflattenBatch(x, usz)
x = stripdims(x);
x = reshape(x, size(x,1), size(x,2)/usz, usz);
x = dlarray(x, "CBU");
end