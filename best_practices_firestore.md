# Firestore Best Practices

## Overview
Firestore is a fully managed, serverless NoSQL document database designed for mobile, web, and server applications. This guide covers best practices for data modeling, queries, security, and performance optimization.

---

## 1. Data Modeling Best Practices

### Document Structure
✅ **Documents** are JSON-like structures (up to 1 MB)
✅ **Collections** contain documents
✅ **Subcollections** nest under documents for hierarchical data
✅ Structure: `collection/document/subcollection/document`

### Document Design Principles
✅ Keep documents **small and focused** (< 100 KB recommended)
✅ Avoid **deeply nested data** (use subcollections instead)
✅ Denormalize frequently accessed data
✅ Use **document references** for relationships
✅ Limit arrays to **reasonable sizes** (< 1000 elements)

### Example Document Structure
```javascript
// Good: Focused document
users/user123: {
  name: "Alice Johnson",
  email: "alice@example.com",
  created: timestamp,
  settings: {
    theme: "dark",
    notifications: true
  }
}

// Use subcollection for related data
users/user123/orders/order456: {
  items: [...],
  total: 99.99,
  status: "shipped"
}
```

### Flat vs Nested Data

**Flat Structure** (Recommended for queried fields)
```javascript
// ✅ Good: Easy to query and index
{
  productId: "ABC123",
  productName: "Laptop",
  productPrice: 999.99,
  productCategory: "Electronics"
}
```

**Nested Structure** (For grouped data not queried separately)
```javascript
// ✅ Good: Related data accessed together
{
  productId: "ABC123",
  product: {
    name: "Laptop",
    price: 999.99,
    category: "Electronics"
  }
}
```

---

## 2. Collections vs Subcollections

### When to Use Root Collections
✅ Data needs to be **queried across all documents**
✅ Global lists (all products, all users)
✅ **Collection group queries** needed

### When to Use Subcollections
✅ Data is **scoped to a parent** document
✅ **Hierarchical relationships** (user → orders → items)
✅ Keep parent document size manageable
✅ Different **access patterns** than parent

### Examples

**Root Collection Pattern**
```
products/                   ← All products queryable
  product1/
  product2/
  
users/                      ← All users
  user1/
  user2/
```

**Subcollection Pattern**
```
users/
  user1/
    orders/                 ← Orders for user1
      order1/
      order2/
    settings/
      preferences/
  user2/
    orders/                 ← Orders for user2
      order3/
```

**Collection Group Pattern** (Query across subcollections)
```javascript
// Query all orders across all users
db.collectionGroup('orders')
  .where('status', '==', 'pending')
  .get();
```

---

## 3. Document ID Design

### Auto-Generated IDs (Recommended)
✅ Use **Firestore auto-generated IDs** (default)
✅ Random, distributed, prevents hotspots
✅ 20-character string

```javascript
// Auto-generated ID
db.collection('users').add({
  name: "Alice",
  email: "alice@example.com"
});

// Or explicitly
const docRef = db.collection('users').doc(); // Auto ID
await docRef.set({ ... });
```

### Custom Document IDs
✅ Use when you have **natural unique identifiers**
✅ Examples: email, username, external system ID
✅ Ensure IDs are **well-distributed** (avoid sequential)

```javascript
// Custom ID (email)
db.collection('users').doc('alice@example.com').set({
  name: "Alice",
  created: timestamp
});

// Custom ID (external system)
db.collection('products').doc('SKU-12345').set({
  name: "Laptop",
  price: 999.99
});
```

### Document ID Anti-Patterns
❌ **Sequential IDs**: `user_001`, `user_002` (creates hotspots)
❌ **Timestamps**: `20251225120000` (creates hotspots)
❌ **Monotonically increasing**: Any sequential pattern

---

## 4. Querying Best Practices

### Simple Queries
✅ Use **where()** for filtering
✅ Use **orderBy()** for sorting
✅ Use **limit()** for pagination
✅ Create **composite indexes** for multiple filters

```javascript
// Simple query
const query = db.collection('products')
  .where('category', '==', 'Electronics')
  .where('price', '<', 1000)
  .orderBy('price', 'desc')
  .limit(10);

const snapshot = await query.get();
snapshot.forEach(doc => {
  console.log(doc.id, doc.data());
});
```

### Composite Indexes
✅ **Automatically created** for single-field queries
✅ **Manual creation required** for:
  - Multiple equality filters + inequality filter
  - Multiple orderBy clauses
  - orderBy + inequality on different fields

```javascript
// Requires composite index
db.collection('products')
  .where('category', '==', 'Electronics')
  .where('inStock', '==', true)
  .orderBy('price', 'asc');

// Firestore will suggest index creation in error message
// Or create manually in Firebase Console or using firebase deploy
```

### Query Limitations
❌ No **OR queries** across different fields (use `array-contains-any` or multiple queries)
❌ No **!= operator** (use `<` and `>` separately)
❌ No **full-text search** (use Algolia, Elasticsearch, or BigQuery)
❌ Maximum **200 inequality filters** per query (across IN/array-contains-any)

### Working Around Limitations

**OR Queries**
```javascript
// ❌ Not supported: field1 == value1 OR field2 == value2

// ✅ Solution 1: Multiple queries + client-side merge
const query1 = db.collection('products').where('category', '==', 'Electronics');
const query2 = db.collection('products').where('category', '==', 'Books');

const [snapshot1, snapshot2] = await Promise.all([query1.get(), query2.get()]);
const combined = [...snapshot1.docs, ...snapshot2.docs];

// ✅ Solution 2: Use array-contains-any (if possible)
db.collection('products').where('category', 'in', ['Electronics', 'Books']);
```

**NOT EQUAL Queries**
```javascript
// ❌ Not supported: field != value

// ✅ Solution: Use < and > separately
const lessThan = db.collection('products').where('price', '<', 100).get();
const greaterThan = db.collection('products').where('price', '>', 100).get();
// Merge results on client
```

---

## 5. Pagination Strategies

### Cursor-Based Pagination (Recommended)
✅ Use **startAfter()** for efficient pagination
✅ Save last document as cursor
✅ Consistent results even with data changes

```javascript
// First page
let query = db.collection('products')
  .orderBy('name')
  .limit(25);

let snapshot = await query.get();
let lastDoc = snapshot.docs[snapshot.docs.length - 1];

// Next page
query = db.collection('products')
  .orderBy('name')
  .startAfter(lastDoc)
  .limit(25);

snapshot = await query.get();
```

### Offset-Based Pagination (Avoid)
❌ **Not recommended**: Offset requires reading and discarding documents
❌ Slow and expensive for large offsets

```javascript
// ❌ Bad: Reading 1000 docs to skip them
db.collection('products')
  .orderBy('name')
  .offset(1000)
  .limit(25);
```

---

## 6. Real-Time Listeners

### Snapshot Listeners
✅ Use **onSnapshot()** for real-time updates
✅ Automatically receives changes
✅ Includes **local changes** before server confirmation

```javascript
// Listen to document changes
const unsubscribe = db.collection('chatrooms').doc('room1')
  .onSnapshot(doc => {
    console.log("Current data:", doc.data());
  }, error => {
    console.error("Listen error:", error);
  });

// Listen to query results
const unsubscribe = db.collection('products')
  .where('inStock', '==', true)
  .onSnapshot(snapshot => {
    snapshot.docChanges().forEach(change => {
      if (change.type === 'added') {
        console.log('New product:', change.doc.data());
      }
      if (change.type === 'modified') {
        console.log('Modified product:', change.doc.data());
      }
      if (change.type === 'removed') {
        console.log('Removed product:', change.doc.data());
      }
    });
  });

// Cleanup: Unsubscribe when done
unsubscribe();
```

### Listener Best Practices
✅ **Unsubscribe** when component unmounts
✅ Handle **errors** in listener callback
✅ Use **local cache** to reduce listener costs
✅ Be mindful of **document read charges** (each listener snapshot counts)
✅ Limit **number of active listeners**

---

## 7. Batch Operations & Transactions

### Batch Writes
✅ Up to **500 operations** per batch
✅ All succeed or all fail (atomic)
✅ Faster than individual writes

```javascript
const batch = db.batch();

// Add multiple operations
const ref1 = db.collection('users').doc('user1');
batch.set(ref1, { name: 'Alice' });

const ref2 = db.collection('users').doc('user2');
batch.update(ref2, { lastLogin: timestamp });

const ref3 = db.collection('users').doc('user3');
batch.delete(ref3);

// Commit all at once
await batch.commit();
```

### Transactions
✅ **Read-then-write** operations
✅ **Atomic**: All succeed or all fail
✅ **Consistent**: Read operations see consistent snapshot
✅ Maximum **500 documents** per transaction

```javascript
const orderRef = db.collection('orders').doc('order123');
const inventoryRef = db.collection('inventory').doc('product456');

await db.runTransaction(async (transaction) => {
  // Read phase
  const orderDoc = await transaction.get(orderRef);
  const inventoryDoc = await transaction.get(inventoryRef);
  
  if (!inventoryDoc.exists) {
    throw new Error("Product does not exist");
  }
  
  const currentStock = inventoryDoc.data().quantity;
  const orderQuantity = orderDoc.data().quantity;
  
  if (currentStock < orderQuantity) {
    throw new Error("Not enough stock");
  }
  
  // Write phase
  transaction.update(inventoryRef, {
    quantity: currentStock - orderQuantity
  });
  
  transaction.update(orderRef, {
    status: 'confirmed'
  });
});
```

### Transaction Best Practices
✅ **Read before write** (all reads first, then all writes)
✅ Keep transactions **short** (< 270 seconds timeout)
✅ Handle **transaction failures** with retry logic
✅ Avoid **contention** on frequently updated documents
✅ Don't modify data outside transaction within transaction function

---

## 8. Security Rules

### Security Rules Best Practices
✅ **Always set security rules** (default denies all)
✅ Use **authentication** to identify users
✅ Follow **principle of least privilege**
✅ Validate **data types and formats**
✅ Test rules thoroughly

### Example Security Rules
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    
    // Users can only read/write their own data
    match /users/{userId} {
      allow read, write: if request.auth != null 
                         && request.auth.uid == userId;
    }
    
    // Public read, authenticated write
    match /posts/{postId} {
      allow read: if true;
      allow create: if request.auth != null 
                    && request.resource.data.author == request.auth.uid;
      allow update, delete: if request.auth != null 
                            && resource.data.author == request.auth.uid;
    }
    
    // Data validation
    match /products/{productId} {
      allow create: if request.auth != null 
                    && request.resource.data.keys().hasAll(['name', 'price'])
                    && request.resource.data.price is number
                    && request.resource.data.price > 0;
    }
    
    // Role-based access
    match /admin/{document=**} {
      allow read, write: if request.auth != null 
                         && get(/databases/$(database)/documents/users/$(request.auth.uid)).data.role == 'admin';
    }
    
    // Subcollection access
    match /users/{userId}/orders/{orderId} {
      allow read: if request.auth != null 
                  && request.auth.uid == userId;
    }
  }
}
```

### Security Rule Functions
```javascript
// Helper function for reusable logic
function isOwner(userId) {
  return request.auth != null && request.auth.uid == userId;
}

function isAdmin() {
  return request.auth != null 
         && get(/databases/$(database)/documents/users/$(request.auth.uid)).data.role == 'admin';
}

match /posts/{postId} {
  allow read: if true;
  allow write: if isOwner(resource.data.author) || isAdmin();
}
```

### Testing Security Rules
```javascript
// Use Firebase Emulator Suite for testing
firebase emulators:start

// Or use Firebase Console Rules Playground
```

---

## 9. Performance Optimization

### Read Optimization
✅ Use **get()** for one-time reads
✅ Use **onSnapshot()** only when real-time needed
✅ Enable **offline persistence** (reduces reads)
✅ Use **startAfter()** for pagination (not offset)
✅ Limit query results with **limit()**

### Write Optimization
✅ Use **batch writes** for multiple operations
✅ Use **FieldValue.increment()** for counters
✅ Avoid **read-then-write** when possible
✅ Use **server timestamps** instead of client timestamps

```javascript
// Good: Increment without reading
db.collection('stats').doc('pageviews').update({
  count: admin.firestore.FieldValue.increment(1)
});

// Good: Server timestamp
db.collection('posts').add({
  title: "Hello",
  created: admin.firestore.FieldValue.serverTimestamp()
});

// Good: Array operations without reading
db.collection('users').doc('user123').update({
  tags: admin.firestore.FieldValue.arrayUnion('premium')
});
```

### Offline Persistence (Mobile/Web)
✅ Enable **offline persistence** for better UX
✅ Reduces **read costs** (cached data)
✅ Automatic **conflict resolution**

```javascript
// Enable persistence (web)
firebase.firestore().enablePersistence()
  .catch((err) => {
    if (err.code == 'failed-precondition') {
      // Multiple tabs open
    } else if (err.code == 'unimplemented') {
      // Browser doesn't support
    }
  });

// Enable persistence (mobile - iOS/Android)
// Enabled by default
```

### Caching Strategies
✅ Use **source options** to control cache behavior

```javascript
// Prefer cache, fall back to server
const snapshot = await db.collection('products').doc('product1')
  .get({ source: 'cache' })
  .catch(() => db.collection('products').doc('product1').get({ source: 'server' }));

// Force server read (bypass cache)
const snapshot = await db.collection('products').doc('product1')
  .get({ source: 'server' });
```

---

## 10. Data Aggregation Patterns

### Distributed Counters
✅ Use for **high-frequency counters** (avoid contention)
✅ Shard counter across multiple documents

```javascript
// Create sharded counter
function createCounter(counterRef, numShards) {
  const batch = db.batch();
  
  for (let i = 0; i < numShards; i++) {
    const shardRef = counterRef.collection('shards').doc(i.toString());
    batch.set(shardRef, { count: 0 });
  }
  
  return batch.commit();
}

// Increment counter (random shard)
function incrementCounter(counterRef, numShards) {
  const shardId = Math.floor(Math.random() * numShards).toString();
  const shardRef = counterRef.collection('shards').doc(shardId);
  
  return shardRef.update({
    count: admin.firestore.FieldValue.increment(1)
  });
}

// Get counter value
async function getCount(counterRef) {
  const snapshot = await counterRef.collection('shards').get();
  return snapshot.docs.reduce((total, doc) => total + doc.data().count, 0);
}
```

### Aggregation in Documents
✅ Store **aggregate values** in parent documents
✅ Update via **Cloud Functions** or transactions
✅ Trade-off: Consistency vs read performance

```javascript
// Store aggregates
users/user123: {
  name: "Alice",
  orderCount: 42,
  totalSpent: 1250.50
}

// Update via Cloud Function
exports.updateUserStats = functions.firestore
  .document('users/{userId}/orders/{orderId}')
  .onCreate(async (snap, context) => {
    const order = snap.data();
    const userRef = db.collection('users').doc(context.params.userId);
    
    return userRef.update({
      orderCount: admin.firestore.FieldValue.increment(1),
      totalSpent: admin.firestore.FieldValue.increment(order.total)
    });
  });
```

---

## 11. Cloud Functions Integration

### Trigger Functions on Firestore Events
✅ **onCreate**: New document created
✅ **onUpdate**: Document modified
✅ **onDelete**: Document deleted
✅ **onWrite**: Any write operation

```javascript
const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();

// Trigger on document creation
exports.onUserCreate = functions.firestore
  .document('users/{userId}')
  .onCreate(async (snap, context) => {
    const user = snap.data();
    
    // Send welcome email
    await sendWelcomeEmail(user.email);
    
    // Create default settings
    await snap.ref.collection('settings').doc('preferences').set({
      theme: 'light',
      notifications: true
    });
  });

// Trigger on document update
exports.onPostUpdate = functions.firestore
  .document('posts/{postId}')
  .onUpdate(async (change, context) => {
    const before = change.before.data();
    const after = change.after.data();
    
    // If published status changed
    if (!before.published && after.published) {
      await notifySubscribers(context.params.postId);
    }
  });

// Trigger on document delete
exports.onUserDelete = functions.firestore
  .document('users/{userId}')
  .onDelete(async (snap, context) => {
    const userId = context.params.userId;
    
    // Clean up user data (subcollections)
    await deleteCollection(db.collection(`users/${userId}/orders`));
    await deleteCollection(db.collection(`users/${userId}/settings`));
  });
```

### Cloud Functions Best Practices
✅ Keep functions **idempotent** (handle retries)
✅ Use **event ID** to detect duplicates
✅ Handle **errors gracefully**
✅ Set appropriate **timeouts and memory**
✅ Use **batching** for bulk operations

---

## 12. Monitoring & Cost Management

### Key Metrics to Monitor
✅ **Document reads**: Per collection/query
✅ **Document writes**: Creates, updates, deletes
✅ **Document deletes**: Track cleanup operations
✅ **Listener usage**: Active listeners count
✅ **Storage**: Total database size
✅ **Bandwidth**: Network egress

### Cost Optimization Strategies
✅ Use **caching** and **offline persistence**
✅ Limit **listener scope** (specific queries, not broad)
✅ Use **batch operations** to reduce write costs
✅ Implement **pagination** (don't load all data)
✅ Clean up **old data** periodically
✅ Use **select() for specific fields** instead of full documents (where supported)

### Monitoring Setup
```javascript
// Log Firestore operations in app
db.collection('products').get()
  .then(snapshot => {
    console.log(`Read ${snapshot.size} documents`);
  });

// Use Firebase Console for usage metrics
// Or export to BigQuery for detailed analysis
```

---

## 13. Common Anti-Patterns to Avoid

❌ **Large arrays**: > 1000 elements (use subcollections)
❌ **Large documents**: > 100 KB (split into subcollections)
❌ **Deeply nested data**: > 3 levels (use subcollections)
❌ **Hot documents**: Frequently updated single document (use sharding)
❌ **Unbounded queries**: No limit() clause
❌ **Offset pagination**: Use cursor pagination instead
❌ **Reading to count**: Use aggregations or counters
❌ **Client-side filtering**: Use queries instead
❌ **No security rules**: Always set rules
❌ **Complex client-side joins**: Denormalize data
❌ **Using Firestore for analytics**: Use BigQuery instead
❌ **Sequential document IDs**: Creates hotspots

---

## 14. Migration & Data Management

### Data Export/Import
```bash
# Export Firestore data
gcloud firestore export gs://bucket-name

# Import Firestore data
gcloud firestore import gs://bucket-name/export-folder

# Export specific collections
gcloud firestore export gs://bucket-name \
  --collection-ids='users,products'
```

### Bulk Data Operations
✅ Use **Admin SDK** for bulk operations
✅ Use **Firestore batch operations** (500 ops limit)
✅ Use **Cloud Functions** with batching for large-scale operations

```javascript
// Bulk delete example
async function deleteCollection(collectionRef, batchSize = 500) {
  const query = collectionRef.limit(batchSize);
  
  return new Promise((resolve, reject) => {
    deleteQueryBatch(query, resolve).catch(reject);
  });
}

async function deleteQueryBatch(query, resolve) {
  const snapshot = await query.get();
  
  if (snapshot.size === 0) {
    resolve();
    return;
  }
  
  const batch = db.batch();
  snapshot.docs.forEach((doc) => {
    batch.delete(doc.ref);
  });
  
  await batch.commit();
  
  // Recurse on next batch
  process.nextTick(() => {
    deleteQueryBatch(query, resolve);
  });
}
```

---

## Quick Reference Checklist

- [ ] Keep documents small (< 100 KB recommended)
- [ ] Use subcollections for hierarchical data
- [ ] Use auto-generated IDs (avoid sequential)
- [ ] Create composite indexes for complex queries
- [ ] Use cursor-based pagination (not offset)
- [ ] Implement security rules (never leave open)
- [ ] Use batch operations for multiple writes
- [ ] Enable offline persistence for mobile/web
- [ ] Unsubscribe from listeners when done
- [ ] Use distributed counters for high-frequency updates
- [ ] Denormalize data for read performance
- [ ] Use Cloud Functions for server-side logic
- [ ] Monitor read/write costs regularly
- [ ] Implement proper error handling
- [ ] Test security rules thoroughly
- [ ] Use transactions for multi-document consistency
- [ ] Cache frequently accessed data
- [ ] Clean up old data periodically

---

## Additional Resources

- [Firestore Documentation](https://firebase.google.com/docs/firestore)
- [Data Modeling Guide](https://firebase.google.com/docs/firestore/data-model)
- [Security Rules Reference](https://firebase.google.com/docs/firestore/security/get-started)
- [Best Practices](https://firebase.google.com/docs/firestore/best-practices)
- [Pricing Calculator](https://firebase.google.com/pricing)

---

*Last Updated: December 25, 2025*
