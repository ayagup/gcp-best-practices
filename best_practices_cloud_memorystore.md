# Cloud Memorystore Best Practices

## Overview
Cloud Memorystore is a fully managed in-memory data store service for Redis and Memcached. This guide covers best practices for performance, scalability, high availability, and cost optimization.

---

## 1. Choosing Between Redis and Memcached

### Use Redis When:
✅ Need **data persistence** capabilities
✅ Require **complex data structures** (lists, sets, sorted sets, hashes)
✅ Need **pub/sub messaging**
✅ Require **transactions** and **Lua scripting**
✅ Need **high availability** with automatic failover
✅ Require **replication** for read scaling
✅ Need **snapshot backups** and point-in-time recovery

### Use Memcached When:
✅ Need **simple key-value caching** only
✅ Want **multi-threaded performance** (better CPU utilization)
✅ Need **horizontal scaling** with consistent hashing
✅ Have **simple cache invalidation** requirements
✅ Cost optimization (Memcached is cheaper)
✅ **Ephemeral caching** (data loss acceptable)

### Feature Comparison

| Feature | Redis | Memcached |
|---------|-------|-----------|
| Data Structures | Strings, Lists, Sets, Sorted Sets, Hashes, Bitmaps, HyperLogLog | Strings only |
| Persistence | Yes (RDB, AOF) | No |
| Replication | Yes (read replicas) | No |
| High Availability | Yes (automatic failover) | No |
| Pub/Sub | Yes | No |
| Transactions | Yes | No |
| Lua Scripting | Yes | No |
| Multi-threading | Single-threaded | Multi-threaded |
| Max Item Size | 512 MB | 1 MB |

---

## 2. Redis Configuration & Sizing

### Instance Tiers

**Basic Tier**
- ✅ Single Redis node
- ✅ Lower cost
- ❌ No replication or automatic failover
- ❌ No SLA
- Best for: Development, testing, non-critical caching

**Standard Tier** (Recommended for Production)
- ✅ **High availability** with automatic failover
- ✅ **Read replicas** for scaling reads
- ✅ **99.9% SLA**
- ✅ **Cross-zone replication**
- Best for: Production workloads, critical applications

### Memory Sizing
✅ Choose memory based on **data size + overhead**
✅ Redis overhead: ~25-30% for replication, fragmentation
✅ Plan for **growth** (3-6 months capacity)
✅ Available sizes: 1 GB to 300 GB

### Sizing Formula
```
Required Memory = (Data Size × 1.3) + (Buffer for growth)

Example:
- Expected data: 10 GB
- With overhead: 10 GB × 1.3 = 13 GB
- With growth buffer: 13 GB × 1.5 = ~20 GB
- Choose: 25 GB instance (or 32 GB for safety)
```

### Performance Capacity
**Standard Tier (per memory size)**
- 1-5 GB: ~12,000 ops/sec
- 10-15 GB: ~25,000 ops/sec
- 20-30 GB: ~50,000 ops/sec
- 40+ GB: ~100,000+ ops/sec

### Redis Version Selection
✅ Use **latest stable version** when possible
✅ Current options: Redis 5.0, 6.x, 7.0
✅ Review **new features** and compatibility
✅ Test thoroughly before upgrading production

---

## 3. High Availability & Disaster Recovery

### Standard Tier HA Configuration
✅ **Automatic failover** in case of node failure
✅ **Primary node** handles writes
✅ **Replica node** in different zone for HA
✅ **Automatic promotion** of replica to primary
✅ Typical failover time: **1-2 minutes**

### Read Replicas (Redis Only)
✅ Up to **5 read replicas** per instance
✅ Distribute **read load** across replicas
✅ **Asynchronous replication** from primary
✅ Use for **read-heavy workloads**
✅ **Regional replicas** only (same region as primary)

```python
# Python client - Connect to read replica
import redis

# Primary (read/write)
primary_client = redis.Redis(
    host='10.0.0.3',
    port=6379,
    decode_responses=True
)

# Read replica (read-only)
replica_client = redis.Redis(
    host='10.0.0.4',
    port=6379,
    decode_responses=True
)

# Write to primary
primary_client.set('key', 'value')

# Read from replica
value = replica_client.get('key')
```

### Backup & Recovery (Redis Only)

**Automated Backups**
✅ Available for **Standard tier** only
✅ **Daily automated snapshots**
✅ Retention: Up to **14 days**
✅ **Point-in-time recovery** within retention window
✅ Minimal performance impact

**Manual Backups**
✅ Create **on-demand snapshots** before major changes
✅ Export to **Cloud Storage** for long-term archival
✅ Use for **disaster recovery** and **data migration**

```bash
# Create manual backup (export to Cloud Storage)
gcloud redis instances export gs://bucket-name/backup.rdb \
  --source=my-redis-instance \
  --region=us-central1

# Import from backup
gcloud redis instances import gs://bucket-name/backup.rdb \
  --destination=my-redis-instance \
  --region=us-central1
```

### Disaster Recovery Strategy
✅ Enable **automated backups** for Standard tier
✅ Export critical data to **Cloud Storage**
✅ Consider **multiple regions** for critical workloads
✅ Document and test **recovery procedures**
✅ Set **RPO/RTO** requirements

---

## 4. Performance Optimization

### Connection Management
✅ Use **connection pooling** (reuse connections)
✅ Set appropriate **pool size** (10-50 connections typical)
✅ Configure **connection timeouts** properly
✅ Monitor **connection count** (avoid exhaustion)
✅ Use **persistent connections** (not per-request)

### Python Connection Pooling
```python
import redis
from redis.connection import ConnectionPool

# Create connection pool
pool = ConnectionPool(
    host='10.0.0.3',
    port=6379,
    max_connections=20,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
    decode_responses=True
)

# Use pool for clients
client = redis.Redis(connection_pool=pool)

# Operations use pooled connections
client.set('key', 'value')
value = client.get('key')
```

### Key Design Best Practices
✅ Use **descriptive key names** with namespaces
✅ Keep keys **short** but readable
✅ Use **consistent naming convention**
✅ Include **version numbers** for schema changes
✅ Use **colons** as separator (Redis convention)

```
✅ Good key naming:
user:12345:profile
user:12345:settings:theme
session:abc123def456
cache:product:99:v2
stats:daily:2025-12-25:pageviews

❌ Bad key naming:
userprofile12345 (hard to parse)
u:12345 (not descriptive)
very_long_descriptive_key_name_that_wastes_memory
```

### Data Expiration (TTL)
✅ **Always set TTL** for cached data
✅ Use **appropriate expiration** based on data volatility
✅ Avoid **persistent keys** for cache data
✅ Use **EXPIRE** or **SETEX** commands

```python
# Set key with expiration
client.setex('session:abc123', 3600, 'session_data')  # 1 hour

# Set key then add expiration
client.set('cache:product:99', 'product_data')
client.expire('cache:product:99', 300)  # 5 minutes

# Check TTL
ttl = client.ttl('session:abc123')
```

### Optimal Data Structures

**Use the Right Structure for the Job**

```python
# ✅ Strings: Simple values, counters
client.set('user:12345:name', 'Alice')
client.incr('pageviews')

# ✅ Hashes: Objects with multiple fields
client.hset('user:12345', mapping={
    'name': 'Alice',
    'email': 'alice@example.com',
    'age': 30
})

# ✅ Lists: Ordered collections, queues
client.lpush('notifications:user:12345', 'New message')
client.lpush('notifications:user:12345', 'Friend request')

# ✅ Sets: Unique collections, tags
client.sadd('user:12345:interests', 'coding', 'music', 'travel')

# ✅ Sorted Sets: Leaderboards, rankings
client.zadd('leaderboard', {'user1': 1000, 'user2': 950, 'user3': 900})

# ✅ Bitmaps: Flags, presence tracking
client.setbit('user:active:2025-12-25', 12345, 1)  # User 12345 active today
```

### Pipelining for Bulk Operations
✅ **Reduce network round trips**
✅ Batch multiple commands
✅ Significantly faster for bulk operations

```python
# Without pipeline (slow - multiple round trips)
for i in range(1000):
    client.set(f'key:{i}', f'value:{i}')

# With pipeline (fast - single round trip)
pipe = client.pipeline()
for i in range(1000):
    pipe.set(f'key:{i}', f'value:{i}')
pipe.execute()
```

### Lua Scripts for Atomic Operations
✅ Execute **complex operations atomically**
✅ **Reduce network round trips**
✅ Ensure **consistency** without transactions

```python
# Atomic increment with max limit (Lua script)
lua_script = """
local current = redis.call('GET', KEYS[1])
if not current then
    redis.call('SET', KEYS[1], 1)
    return 1
elseif tonumber(current) < tonumber(ARGV[1]) then
    redis.call('INCR', KEYS[1])
    return tonumber(current) + 1
else
    return tonumber(current)
end
"""

increment_with_limit = client.register_script(lua_script)
result = increment_with_limit(keys=['counter'], args=[100])  # Max 100
```

---

## 5. Caching Patterns & Strategies

### Cache-Aside (Lazy Loading)
✅ **Most common pattern**
✅ Application checks cache first
✅ On miss, load from database and cache
✅ Good for **read-heavy workloads**

```python
def get_user(user_id):
    # Try cache first
    cache_key = f'user:{user_id}'
    user = redis_client.get(cache_key)
    
    if user:
        return json.loads(user)  # Cache hit
    
    # Cache miss - load from database
    user = database.get_user(user_id)
    
    # Store in cache (1 hour TTL)
    redis_client.setex(cache_key, 3600, json.dumps(user))
    
    return user
```

### Write-Through
✅ Write to cache and database **simultaneously**
✅ Ensures cache is **always up-to-date**
✅ Higher write latency
✅ Good for **write-heavy workloads** where consistency matters

```python
def update_user(user_id, user_data):
    # Write to database
    database.update_user(user_id, user_data)
    
    # Write to cache
    cache_key = f'user:{user_id}'
    redis_client.setex(cache_key, 3600, json.dumps(user_data))
```

### Write-Behind (Write-Back)
✅ Write to **cache first**, database later (async)
✅ **Lower latency** for writes
✅ Risk of **data loss** if cache fails before DB write
✅ Good for **high write throughput** requirements

```python
def update_user(user_id, user_data):
    # Write to cache immediately
    cache_key = f'user:{user_id}'
    redis_client.setex(cache_key, 3600, json.dumps(user_data))
    
    # Queue database write (async)
    write_queue.add({
        'operation': 'update_user',
        'user_id': user_id,
        'data': user_data
    })
```

### Cache Invalidation Strategies

**Time-Based (TTL)**
✅ Simplest approach
✅ Set expiration time
✅ May serve stale data until expiration

**Event-Based**
✅ Invalidate cache on data updates
✅ More complex but more accurate

```python
def update_product(product_id, product_data):
    # Update database
    database.update_product(product_id, product_data)
    
    # Invalidate cache
    cache_key = f'product:{product_id}'
    redis_client.delete(cache_key)
    
    # Or update cache immediately (write-through)
    redis_client.setex(cache_key, 3600, json.dumps(product_data))
```

**Pattern-Based**
✅ Delete keys matching pattern
✅ Use carefully (SCAN, not KEYS in production)

```python
# Invalidate all user-related caches
cursor = 0
while True:
    cursor, keys = redis_client.scan(cursor, match='user:*', count=100)
    if keys:
        redis_client.delete(*keys)
    if cursor == 0:
        break
```

---

## 6. Session Management

### Session Store Pattern
✅ Store **user sessions** in Redis
✅ Fast session lookup
✅ Automatic expiration with TTL
✅ Shared across multiple app servers

```python
import json
import uuid

def create_session(user_id, session_data):
    session_id = str(uuid.uuid4())
    session_key = f'session:{session_id}'
    
    # Store session with 24-hour expiration
    redis_client.setex(
        session_key,
        86400,  # 24 hours
        json.dumps({
            'user_id': user_id,
            'created': time.time(),
            **session_data
        })
    )
    
    return session_id

def get_session(session_id):
    session_key = f'session:{session_id}'
    session_data = redis_client.get(session_key)
    
    if session_data:
        # Extend session expiration on access
        redis_client.expire(session_key, 86400)
        return json.loads(session_data)
    
    return None

def destroy_session(session_id):
    session_key = f'session:{session_id}'
    redis_client.delete(session_key)
```

### Session Best Practices
✅ Set **appropriate TTL** based on inactivity timeout
✅ **Extend TTL** on user activity
✅ Store only **essential session data**
✅ Use **secure session IDs** (UUID)
✅ Implement **session cleanup** for expired sessions

---

## 7. Pub/Sub Messaging (Redis Only)

### Pub/Sub Pattern
✅ **Real-time messaging** between components
✅ **Publish** messages to channels
✅ **Subscribe** to channels for messages
✅ Fire-and-forget (messages not persisted)

```python
import redis
import threading

# Publisher
def publish_message():
    client = redis.Redis(host='10.0.0.3', port=6379)
    client.publish('notifications', 'New order received')

# Subscriber
def subscribe_to_channel():
    client = redis.Redis(host='10.0.0.3', port=6379)
    pubsub = client.pubsub()
    pubsub.subscribe('notifications')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"Received: {message['data']}")

# Run subscriber in background thread
subscriber_thread = threading.Thread(target=subscribe_to_channel)
subscriber_thread.daemon = True
subscriber_thread.start()

# Publish messages
publish_message()
```

### Pattern Subscriptions
```python
# Subscribe to multiple channels with pattern
pubsub = client.pubsub()
pubsub.psubscribe('notifications:*')

# Matches: notifications:orders, notifications:users, etc.
```

### Pub/Sub Best Practices
✅ Use for **real-time notifications** and events
✅ Don't rely on pub/sub for **critical messages** (use message queue instead)
✅ Messages are **lost if no subscribers** listening
✅ Consider **Pub/Sub service** for more robust messaging
✅ Monitor **subscriber count** and message rate

---

## 8. Monitoring & Alerting

### Key Metrics to Monitor
✅ **Memory usage**: % of allocated memory
✅ **CPU utilization**: % CPU usage (Redis is single-threaded)
✅ **Connected clients**: Number of active connections
✅ **Operations per second**: Read/write throughput
✅ **Cache hit rate**: Hit / (Hit + Miss) ratio
✅ **Evicted keys**: Keys removed due to memory pressure
✅ **Replication lag**: For Standard tier instances
✅ **Network bytes in/out**: Bandwidth usage

### Setting Up Monitoring
```bash
# View instance metrics in Cloud Console
# Or use Cloud Monitoring API

gcloud redis instances describe my-redis-instance \
  --region=us-central1
```

### Alert Thresholds (Recommended)
- **Memory usage**: Alert at 80%, critical at 90%
- **CPU utilization**: Alert at 70%, critical at 85%
- **Connected clients**: Alert at 90% of max connections
- **Cache hit rate**: Alert if drops below 80%
- **Evicted keys**: Alert if consistently > 0
- **Replication lag**: Alert if > 5 seconds

### Cloud Monitoring Dashboard
```python
# Key metrics to include in dashboard:
# - Memory utilization over time
# - Operations per second (read/write)
# - Cache hit rate
# - Connected clients
# - CPU utilization
# - Eviction rate
```

### Redis INFO Command
```python
# Get Redis statistics
info = client.info()

print(f"Used memory: {info['used_memory_human']}")
print(f"Connected clients: {info['connected_clients']}")
print(f"Total commands processed: {info['total_commands_processed']}")
print(f"Keyspace hits: {info['keyspace_hits']}")
print(f"Keyspace misses: {info['keyspace_misses']}")

# Calculate hit rate
hit_rate = info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses'])
print(f"Cache hit rate: {hit_rate:.2%}")
```

---

## 9. Security Best Practices

### Network Security
✅ Use **Private IP** (VPC) for secure connectivity
✅ Enable **authorized networks** if using VPC peering
✅ Use **VPC Service Controls** for perimeter security
✅ **No public IP exposure** for production instances

### Authentication
✅ Enable **AUTH** (password authentication) for Redis
✅ Use **strong passwords** (random, long)
✅ Rotate passwords periodically
✅ Store passwords in **Secret Manager**

```bash
# Create instance with AUTH enabled
gcloud redis instances create my-redis-instance \
  --region=us-central1 \
  --tier=standard \
  --size=5 \
  --auth-enabled

# Get AUTH string
gcloud redis instances describe my-redis-instance \
  --region=us-central1 \
  --format="value(authString)"
```

```python
# Connect with AUTH
client = redis.Redis(
    host='10.0.0.3',
    port=6379,
    password='your-auth-string',
    decode_responses=True
)
```

### Encryption
✅ **In-transit encryption**: TLS/SSL for connections
✅ **At-rest encryption**: Automatic with Google-managed keys
✅ **CMEK**: Customer-managed encryption keys for compliance

```bash
# Create instance with in-transit encryption
gcloud redis instances create my-redis-instance \
  --region=us-central1 \
  --tier=standard \
  --size=5 \
  --transit-encryption-mode=SERVER_AUTHENTICATION
```

### Access Control
✅ Use **IAM roles** for instance management
✅ Implement **least privilege** access
✅ Separate **development and production** instances

### IAM Roles
- `roles/redis.viewer`: View instance configuration
- `roles/redis.editor`: Modify instance configuration
- `roles/redis.admin`: Full control over instances

---

## 10. Cost Optimization

### Sizing Optimization
✅ **Right-size memory** based on actual usage
✅ Monitor memory utilization (target 70-80%)
✅ Start small and **scale up as needed**
✅ Use **Basic tier** for dev/test environments

### Tier Selection
✅ **Basic tier**: 50-70% cheaper than Standard
✅ Use Basic for **non-critical workloads**
✅ Use Standard for **production workloads**

### Regional Selection
✅ Choose **region close to compute resources**
✅ Avoid **cross-region traffic** (latency + cost)
✅ Consider **dual-region** for DR (higher cost)

### Data Optimization
✅ Set **TTL on all cached data**
✅ Use **appropriate data structures** (hashes vs strings)
✅ Keep **keys short** but descriptive
✅ Compress large values before storing
✅ Clean up **unused keys** regularly

### Memory Efficiency Tips
```python
# ❌ Storing individual fields (more memory)
client.set('user:12345:name', 'Alice')
client.set('user:12345:email', 'alice@example.com')
client.set('user:12345:age', '30')

# ✅ Using hash (more efficient)
client.hset('user:12345', mapping={
    'name': 'Alice',
    'email': 'alice@example.com',
    'age': 30
})
```

---

## 11. Memcached Best Practices

### Memcached Configuration
✅ Choose appropriate **memory size** (1 GB to 300 GB)
✅ Use **multiple nodes** for larger deployments
✅ Configure **connection pool** in application
✅ Set **default TTL** for all cached items

### Memcached Connection
```python
from google.cloud import memcache
import bmemcached

# Connect to Memcached instance
client = bmemcached.Client([('10.0.0.5:11211')])

# Set value with TTL
client.set('key', 'value', time=3600)  # 1 hour

# Get value
value = client.get('key')

# Delete value
client.delete('key')
```

### Memcached Limitations
❌ No data persistence
❌ No replication
❌ No pub/sub
❌ No complex data structures
❌ 1 MB value size limit

### When Memcached Shines
✅ **Simple caching** needs
✅ **Multi-threaded** applications (better CPU utilization)
✅ **Cost-sensitive** projects
✅ **Horizontally scalable** caching layer

---

## 12. Common Anti-Patterns to Avoid

❌ **Using Redis as primary database**: Not designed for durability
❌ **Storing large objects**: > 1 MB values (use Cloud Storage + Redis reference)
❌ **Not setting TTL**: Memory fills up, evictions begin
❌ **Using KEYS command**: Blocks Redis (use SCAN instead)
❌ **Not using connection pooling**: Connection overhead
❌ **Ignoring memory limits**: OOM errors
❌ **Not monitoring metrics**: Miss performance issues
❌ **Hotkey problems**: One key accessed very frequently (consider client-side cache)
❌ **Using Basic tier for production**: No HA, no SLA
❌ **Cross-region connections**: High latency, high cost
❌ **Not implementing cache invalidation**: Serving stale data
❌ **Synchronous cache updates**: Blocks application flow

---

## 13. Migration to Cloud Memorystore

### Migration Strategies

**1. Dual Write** (Recommended)
- Write to both old and new Redis
- Gradually shift reads to new instance
- Validate data consistency
- Cutover when confident

**2. Export/Import**
```bash
# Export from source Redis
redis-cli --rdb /tmp/dump.rdb

# Import to Cloud Memorystore
gcloud redis instances import gs://bucket/dump.rdb \
  --destination=my-redis-instance \
  --region=us-central1
```

**3. Live Migration Tools**
- Use RIOT (Redis Input/Output Tool)
- Use redis-migrate tool
- Continuous replication during migration

### Pre-Migration Checklist
- [ ] Assess current memory usage
- [ ] Identify data access patterns
- [ ] Choose appropriate tier (Basic vs Standard)
- [ ] Select memory size
- [ ] Plan for testing period
- [ ] Document rollback procedure
- [ ] Update application configuration
- [ ] Test connection and performance

---

## Quick Reference Checklist

### Redis
- [ ] Use Standard tier for production (HA + SLA)
- [ ] Enable AUTH for security
- [ ] Enable in-transit encryption
- [ ] Set up automated backups
- [ ] Use connection pooling
- [ ] Set TTL on all cached data
- [ ] Use appropriate data structures
- [ ] Monitor memory and CPU usage
- [ ] Set up alerting for key metrics
- [ ] Use read replicas for read-heavy workloads
- [ ] Implement proper cache invalidation
- [ ] Use pipelining for bulk operations
- [ ] Avoid KEYS command (use SCAN)
- [ ] Keep keys short but descriptive

### Memcached
- [ ] Use for simple key-value caching
- [ ] Configure multiple nodes for scale
- [ ] Set default TTL for all items
- [ ] Use connection pooling
- [ ] Monitor memory usage
- [ ] Implement client-side consistent hashing

---

## Additional Resources

- [Cloud Memorystore Documentation](https://cloud.google.com/memorystore/docs)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Redis Commands Reference](https://redis.io/commands)
- [Memcached Documentation](https://memcached.org/)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
