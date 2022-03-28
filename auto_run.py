from main import main as entry_point
import os


config = {
    'name': 'Test lower way number',
    'global arguments': {
        'num_global_iters': 12000,
        'model': 'dnn',
        'datast': 'Mnist',
        'algorithm': 'pFedMe',
    },
    'tasks': [
        {
            'class_per_client': 2,
            'batch_size': 24,
        },
        {
            'class_per_client': 3,
            'batch_size': 24,
        },
        {
            'class_per_client': 3,
            'batch_size': 36,
        },
        {
            'class_per_client': 5,
            'batch_size': 24,
        },
        {
            'class_per_client': 5,
            'batch_size': 36,
        },
        {
            'class_per_client': 5,
            'batch_size': 60,
        },
        {
            'class_per_client': 8,
            'batch_size': 24,
        },
        {
            'class_per_client': 8,
            'batch_size': 36,
        },
        {
            'class_per_client': 8,
            'batch_size': 60,
        },
        {
            'class_per_client': 10,
            'batch_size': 24,
        },
        {
            'class_per_client': 10,
            'batch_size': 60,
        },
        {
            'class_per_client': 10,
            'batch_size': 36,
        },
    ],
}


def main():
    print(f'Running {config["name"]}...')
    
    if not os.path.exists('reports'):
        os.mkdir('reports')
    
    for i, task in enumerate(config['tasks']):
        print('-------------------------')
        print(f'Task {i + 1}/{len(config["tasks"])}')
        
        arguments = []
        args = {**config['global arguments'], **task}
        for k, v in args.items():
            print(f'{k}: {v}')
            arguments.extend([f'--{k}', str(v)]) # create command line argument
        print('-------------------------')
        acc_over_clients, acc_over_samples = entry_point(arguments)
        result = f'Task {i + 1}: {acc_over_clients} / {acc_over_samples}'
        with open(os.path.join('reports', f'pfedme_mnist_{args["class_per_client"]}_{args["batch_size"]}'), 'w') as f:
            print(result, file=f)
        
            

if __name__ == '__main__':
    main()