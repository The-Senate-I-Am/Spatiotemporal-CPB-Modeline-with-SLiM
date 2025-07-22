class Field:
    # Creates a field with unique field_id, latitude, longitude, and an empty data list
    # initializes cluster_num to -1 by default 
    def __init__(self, field_id, latitude, longitude, cluster=-1):
        self.field_id = field_id
        self.latitude = latitude
        self.longitude = longitude
        self.data = []  # List of tuples: (year, count, gdd)
        self.cluster = cluster

    # Adds a data point for a specific year to the field's data list 
    def add_data_point(self, year, count, gdd):
        self.data.append((year, count, gdd))
        
    # Sets the cluster number for the field
    def set_cluster(self, cluster):
        self.cluster = cluster

    def __repr__(self):
        return (f"Field(field_id={self.field_id}, latitude={self.latitude}, "
                f"longitude={self.longitude}, data={self.data})")
        
        
class Cluster:
    # Creates a cluster with a unique cluster_id and an empty list of fields
    def __init__(self, cluster_id, latitude=None, longitude=None):
        self.cluster_id = cluster_id
        self.latitude = latitude
        self.longitude = longitude
        self.fields = []  # List of Field objects in this cluster
        self.data = [] # List of averaged data for the cluster in the format: (avg_count, avg_gdd)
        

    # Adds a field to the cluster
    def add_field(self, field):
        self.fields.append(field)
        
    # Sets the coordinates of the cluster after adding the fields
    def calculate_coordinates(self):
        if not self.fields:
            self.latitude = None
            self.longitude = None
            return
        self.latitude = sum(field.latitude for field in self.fields) / len(self.fields)
        self.longitude = sum(field.longitude for field in self.fields) / len(self.fields)
        
        
        
    # Fills the cluster's data attribute with average counts and GDD 
    def average_data(self):
        #Generate an intermediate data structure to hold all the data for each type in a list to average later
        inter_data = []
        for i in range(1,len(self.fields[0].data[0])): # disregard the year in the 0-index of the data
            inter_data.append([])
        
        for field in self.fields:
            for datapoint in field.data:
                for i in range(1, len(datapoint)):
                    inter_data[i-1].append(datapoint[i])
                    
        # Calculate the average for each type of data and store it in the cluster's data attribute
        avg_data = []
        for i in range(len(inter_data)):
            avg_data.append(sum(inter_data[i]) / len(inter_data[i]))
        self.data = avg_data
        

    def __repr__(self):
        return f"Cluster(cluster_id={self.cluster_id}, fields={self.fields})"