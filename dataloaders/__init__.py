import importlib
from torch.utils.data import DataLoader


def initialize_data_functions(args):
    """
    Retrieve dataloaders and visualization function. 
    
    Example:
        load_data, visualize_data = initialize_data_functions(args)
        dataloaders, dataset = load_data(config.dataset, config.loader)
    """
    try:
        dataset_module = f'dataloaders.{get_data_module(args)}'
        dataset_module = importlib.import_module(dataset_module)
    except Exception as e:
        print(f'Error: dataloaders.{get_data_module(args)}')
        raise e
    load_data = getattr(dataset_module, 'load_data')
    visualize_data = getattr(dataset_module, 'visualize_data')
    return load_data, visualize_data


def get_data_module(args):
    dataset_fname = args.dataset
    
    # Informer - time series forecasting
    if args.dataset in ['etth1', 'etth2', 'ettm1', 'ettm2', 
                        'ecl', 'traffic', 'weather']:
        args.dataset_type = 'informer'
        if args.dataset[:3] == 'ett':
            args.variant = int(args.dataset[-1])
            args.dataset = args.dataset[:-1]
        else:
            args.variant = None
        data_module = args.dataset_type
    
    elif args.dataset in ['etth', 'ettm']:
        print(f'Dataset type: {args.dataset_type}')
        print(f'-> dataset: {args.dataset}{args.variant}')
        data_module = args.dataset_type

    elif args.dataset in ['ODE_Lorenz', 'PDE_KS', 'Lorenz_Official', 'KS_Official']:
        args.dataset_type = "ctf"
        # args.variant: needs to be matrix ID, like 3 for X3
        print(f'Dataset type: {args.dataset_type}')
        print(f'-> dataset: {args.dataset}')
        print(f'-> pair_id: {args.pair_id}')
        print(f'-> train_ids: {args.train_ids}')
        print(f'-> reconstruct_id: {args.reconstruct_id}')
        print(f'-> forecast_id: {args.forecast_id}')
        print(f'-> forecast_length: {args.forecast_length}')
        print(f'-> burn_in: {args.burn_in}')
        data_module = args.dataset_type
        
    else:
        data_module = f'{args.dataset_type}.{dataset_fname}'
        raise NotImplementedError(f'{args.dataset} not implemented!')
    
    return data_module


def get_evaluation_loaders(dataloaders, batch_size):
    eval_dataloaders = [
        DataLoader(dataloader.dataset, 
                   shuffle=False,
                   batch_size=batch_size,
                   num_workers=0)
        for dataloader in dataloaders
    ]
    return eval_dataloaders