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
        self.data = []    # List of tuples: (year, estimated, count, gdd) aggregated or for the cluster
        self.START_YEAR = 2014  # The starting year for the data

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
        
        
        
    # Fills the cluster's data attribute with average counts and GDD for each year
    def fill_data(self, year_start=2014, year_end=2024):
        # Initialize an intermediate dictionary to hold all the data for each year
        inter_data = dict()
        for i in range(year_start, year_end + 1):
            inter_data[i] = ([],[]) # Initialize with empty lists for collecting count and gdd data for that specific year
        for field in self.fields:
            for datapoint in field.data:
                inter_data[datapoint[0]][0].append(datapoint[1])
                inter_data[datapoint[0]][1].append(datapoint[2])
                
        # Convert that intermediate data into a final list of tuples with (year, estimated, avg_count, and avg_gdd)
        # If no data is available for a year, it will be marked with a True for estimated, indicating that it needs to be estimated later
        final_data = []
        for year in inter_data.keys():
            if len(inter_data[year][0]) == 0:    
                final_data.append((year, True, None, None))
            else:
                avg_count = sum(inter_data[year][0]) / len(inter_data[year][0])
                avg_gdd = sum(inter_data[year][1]) / len(inter_data[year][1])
                final_data.append((year, False, avg_count, avg_gdd))
                
        self.data = final_data
        
        
        
    # Estimates the unfilled data from the field data for this cluster. If data is not already filled, this calls fill_data().
    def estimate_data(self, year_start=2014, year_end=2024):
        if not self.data:
            self.fill_data(year_start, year_end)
            
        #TODO find a function to use to estimate the data based on the data around it.
        
        
        
        
        

    def __repr__(self):
        return f"Cluster(cluster_id={self.cluster_id}, fields={self.fields})"