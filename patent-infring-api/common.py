

from pymemcache.client import base
import json, sys


class CacheSetUp:
    patends_loaded = False
    companpy_loaded = False

    patent_test_file ='./data/patent_test.json' 
    company_test_file ='./data/company_test.json' 

    patent_file ='./data/patents.json' 
    company_file ='./data/company_products.json' 
    
    # True: for real dataase, False for test dataset
    datasource_type = False

    memcached_client = base.Client(('localhost', 11211))

    def __init__(self):
        self.load_data_into_memcached(self.datasource_type)

    def reload(self, datasource_type):
        if datasource_type == self.datasource_type:
            return
        self.load_data_into_memcached(datasource_type)
     
    # Load and cache data into Memcached
    def load_data_into_memcached(self, datasource_type):
        filePath = self.patent_test_file

        if datasource_type == True: 
            filePath = self.patent_file 

        self.datasource_type = datasource_type
        # Load patent source data
        with open(filePath, 'r') as file:
            patent_data = json.load(file)
            for patent in patent_data:  # Iterate directly over the array of sub-JSONs
                self.memcached_client.set(f"patent:{patent['publication_number']}", json.dumps(patent))
    
    # Helper function to load company data directly from company.json
    # TODO: need to optimize this function
    def load_company_data(self, datasource_type, company_name):
        filePath = self.company_test_file

        if datasource_type == True: 
            filePath = self.company_file 

        self.datasource_type = datasource_type

        with open(filePath, 'r') as file:
            company_data = json.load(file)
            for company in company_data['companies']:
                if company['name'].lower() == company_name.lower():
                    return company
        return None
    

    def getPatent(self, patent_id):
        return self.memcached_client.get(patent_id)
    
    def getAnalysis(self, analysis_id):
        return self.memcached_client.get(analysis_id)
    
    def set(self, analysis_id, json_body):
        self.memcached_client.set(analysis_id, json_body)