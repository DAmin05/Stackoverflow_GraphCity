#!/bin/sh
sed -i '1c export const strataAddress = "http://localhost:8081/"' fpViewer/localLib/strata.js
sed -i '2c node app.js temp=./temp port=8081' graph-strata/run.sh
sed -i '13c const localPort = 8080' Graph_City_Web/app_addon.js
sed -i '16c const strataAddress = "http://localhost:8081/"' Graph_City_Web/app_addon.js
sed -i '24c const hostAddress = "http://localhost:8080"' Graph_City_Web/scripts/main.js
sed -i '25c const localHost = `http://localhost:8080/`' Graph_City_Web/scripts/main.js
sed -i '8c const localHost = `http://localhost:8080/`' Graph_City_Web/scripts/dag_view_server.js
sed -i '9c const hostAddress = "http://localhost:8080"' Graph_City_Web/scripts/dag_view_server.js
sed -i '11c var PREFIX = "http://localhost:8081/"' Graph_City_Web/scripts/dag_view_server.js
