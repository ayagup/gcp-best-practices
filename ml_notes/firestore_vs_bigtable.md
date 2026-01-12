# Firestore vs Bigtable

## **Firestore**
- **Type**: NoSQL document database
- **Use Cases**: 
  - Mobile/web apps with real-time sync
  - User profiles, catalogs, game state
  - Apps requiring offline support
- **Data Model**: Documents in collections (JSON-like)
- **Scalability**: Automatic, scales to millions of concurrent connections
- **Queries**: Rich querying with indexes, complex filters
- **Transactions**: ACID transactions supported
- **Real-time**: Built-in real-time listeners
- **Pricing**: Pay per read/write/delete operation

## **Bigtable**
- **Type**: Wide-column NoSQL database
- **Use Cases**:
  - Time-series data (IoT, monitoring)
  - Financial data, analytics workloads
  - Large-scale batch/stream processing
  - Low-latency access to massive datasets (1TB+)
- **Data Model**: Sparse, distributed sorted map (row key → column families)
- **Scalability**: Petabyte-scale, linear performance
- **Queries**: Single row key lookups or scans, no secondary indexes
- **Transactions**: Single-row atomicity only
- **Real-time**: Low latency (sub-10ms), but no real-time sync
- **Pricing**: Pay per node-hour + storage

## **Key Differences**

| Feature | Firestore | Bigtable |
|---------|-----------|----------|
| **Minimum Scale** | Free tier available | Minimum 1 node (~$0.65/hr) |
| **Query Flexibility** | High (SQL-like queries) | Low (key-based only) |
| **Real-time Updates** | Yes | No |
| **Best For** | < 1TB, transactional | > 1TB, analytical |
| **Learning Curve** | Easier | Steeper (requires schema design expertise) |

**Choose Firestore** for app backends with complex queries and real-time needs.  
**Choose Bigtable** for high-throughput analytics on massive datasets.