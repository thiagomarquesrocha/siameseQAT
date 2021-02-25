
from jobs.data_pipeline import DataPipeline

def get_colab_value(params):
    try:
        colab = params[2]
        if colab == 'colab':
          COLAB = 'drive/My Drive/Colab Notebooks/'
        else:
          COLAB = ''
    except:
      COLAB = ''
    return COLAB

def get_preprocessing_value(params):
    try:
        PREPROCESSING = params[3]
    except:
        PREPROCESSING = 'baseline'
    return PREPROCESSING

def main():

    params = sys.argv

    try: 
      dataset = params[1]
    except:
      print('###### Missing the dataset to be processed #########')
      print("Ex: $ python src/cli.py {eclipse, eclipse_small, netbeans, openoffice} colab {baseline, bert}")
      exit(1)

    COLAB = get_colab_value(params)
    PREPROCESSING = get_preprocessing_value(params)
    
    print("Params: dataset={}, colab={}, preprocessing={}".format(dataset, COLAB, PREPROCESSING))
    
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
      }
      'eclipse_test' : {
          'DATASET' : 'eclipse_test',
          'DOMAIN' : 'eclipse_test'
      }
    }

    pipeline = DataPipeline(op[dataset]['DATASET'], op[dataset]['DOMAIN'], COLAB, PREPROCESSING)
    pipeline.run()

if __name__ == '__main__':
  main()