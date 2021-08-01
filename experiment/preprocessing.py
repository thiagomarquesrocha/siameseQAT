
import mlflow
import click
from src.jobs.data_pipeline import DataPipeline

@click.command(
    help="SiameseQAT script for preprocessing"
)
@click.option("--dataset", default="eclipse", type=str, help="Database ex: eclipse, openoffice, netbeans")
@click.option("--preprocessor", default="bert", type=str, help="Database ex: eclipse, openoffice, netbeans")
def preprocessing(dataset, preprocessor):
    
    print("Params: dataset={}, preprocessing={}".format(dataset, preprocessor))
    
    op = {
      'eclipse' : {
        'DATASET' : 'eclipse',
        'DOMAIN' : 'eclipse'
      },
      'eclipse_small' : {
        'DATASET' : 'eclipse_small',
        'DOMAIN' : 'eclipse_small'
      },
      'netbeans' : {
        'DATASET' : 'netbeans',
        'DOMAIN' : 'netbeans'
      },
      'openoffice' : {
        'DATASET' : 'openoffice',
        'DOMAIN' : 'openoffice'
      },
      'firefox' : {
          'DATASET' : 'firefox',
          'DOMAIN' : 'firefox'
      },
      'eclipse_test' : {
          'DATASET' : 'eclipse_test',
          'DOMAIN' : 'eclipse_test'
      }
    }

    pipeline = DataPipeline(op[dataset]['DATASET'], op[dataset]['DOMAIN'], preprocessor)
    pipeline.run()

if __name__ == '__main__':
  preprocessing()