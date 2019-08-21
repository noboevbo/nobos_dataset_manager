Getting started:

# Install and connect Postgresql Docker
```bash
docker-compose up -d
psql --host localhost --port 1111 --user dennis
```

# Create Database
```sql
create database ground_truth_store;
create database ground_truth_store_test;
```
# Connect to database
```bash
\connect ground_truth_store
```

# Create a use
```sql
create user gt_worker with encrypted password 'PASSWORD';
grant all privileges on database ground_truth_store to gt_worker;
grant all privileges on database ground_truth_store_test to gt_worker;

\connect ground_truth_store
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gt_worker;

\connect ground_truth_store_test
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gt_worker;
```

# Change dennis PW
```sql
ALTER USER dennis WITH PASSWORD 'newpass';
```

# How to Mirror DB
```bash
pg_dump -C -h localhost -p 1111 -U dennis ground_truth_store | psql -h SERVERNAME -p 1111 -U REMOTEUSERNAME ground_truth_store
pg_dump -C -h SERVERNAME -p 1111 -U REMOTEUSERNAME ground_truth_store | psql -h localhost -p 1111 -U dennis ground_truth_store
```
```bash
vim .pgpass
```
```bash
localhost:1111:ground_truth_store:dennis:PASSWORD
SERVERNAME:1111:ground_truth_store:REMOTEUSERNAME:PASSWORD
```
```sql
chmod 600 .pgpass
```