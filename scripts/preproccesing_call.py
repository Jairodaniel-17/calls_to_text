# primero se traen los audios de la base de datos
from connection.connection_db import MySQLConnection

# consulta audio y id de la tabla audio


def get_audios(query):
    with MySQLConnection() as connection:
        result = connection.execute_query(query)
        return result
