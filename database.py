import psycopg2

class Database:

    def __init__(self,database,user,password,host,port):
        self.conn=psycopg2.connect(database=database,user=user,password=password,host=host,port=port)
        self.cur=self.conn.cursor()
    
    def close(self):
        self.cur.close()
        self.conn.close()
    
    def query(self,filename):
        self.cur.execute('select * from patient where filename=%s',(filename,))
        return self.cur.fetchall()    

# database=Database("postgres","postgres","123456","10.17.12.166","5432")
# print(database.query("IM_091715.png"))
# database.close()


