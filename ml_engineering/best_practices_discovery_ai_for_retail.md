# Discovery AI for Retail Best Practices

*Last Updated: January 4, 2026*

## Overview

Discovery AI for Retail (Retail Search) provides intelligent product search, personalized browse experiences, autocomplete, and visual search capabilities powered by Google's retail-specialized machine learning models.

---

## 1. Product Catalog Setup

### Import Product Catalog

```python
from google.cloud import retail_v2
from google.cloud.retail_v2 import Product, PriceInfo, Image
from google.api_core import retry

class RetailSearchClient:
    """Client for Retail Search operations."""
    
    def __init__(self, project_id, location='global', catalog_id='default_catalog'):
        self.project_id = project_id
        self.location = location
        self.catalog_id = catalog_id
        
        self.product_service = retail_v2.ProductServiceClient()
        
        # Build resource paths
        self.catalog_path = self.product_service.catalog_path(
            project_id, location, catalog_id
        )
        self.branch_path = self.product_service.branch_path(
            project_id, location, catalog_id, 'default_branch'
        )
    
    def create_product(
        self,
        product_id,
        title,
        categories,
        price,
        currency='USD',
        images=None,
        description='',
        attributes=None,
        availability='IN_STOCK'
    ):
        """Create product in catalog."""
        
        if images is None:
            images = []
        if attributes is None:
            attributes = {}
        
        # Build product
        product = Product(
            name=f"{self.branch_path}/products/{product_id}",
            id=product_id,
            type_=Product.Type.PRIMARY,
            primary_product_id=product_id,
            categories=categories,
            title=title,
            description=description,
            price_info=PriceInfo(
                currency_code=currency,
                price=price,
                original_price=price,
            ),
            availability=availability,
            images=[Image(uri=img) for img in images],
            attributes=attributes,
        )
        
        request = retail_v2.CreateProductRequest(
            parent=self.branch_path,
            product=product,
            product_id=product_id,
        )
        
        try:
            response = self.product_service.create_product(request=request)
            print(f"✓ Created product: {product_id}")
            return response
        except Exception as e:
            print(f"✗ Error creating product {product_id}: {e}")
            return None
    
    def batch_import_products(self, gcs_input_uri):
        """Import products from Cloud Storage."""
        
        from google.cloud.retail_v2 import ProductInputConfig, GcsSource, ImportProductsRequest
        
        input_config = ProductInputConfig(
            gcs_source=GcsSource(
                input_uris=[gcs_input_uri],
                data_schema='product',
            )
        )
        
        request = ImportProductsRequest(
            parent=self.branch_path,
            input_config=input_config,
            reconciliation_mode=ImportProductsRequest.ReconciliationMode.INCREMENTAL,
        )
        
        print(f"Starting batch import from: {gcs_input_uri}")
        
        operation = self.product_service.import_products(request=request)
        
        print("Waiting for operation to complete...")
        response = operation.result(timeout=1800)  # 30 minutes
        
        if response.error_samples:
            print(f"⚠️  Import completed with {len(response.error_samples)} errors")
            for error in response.error_samples[:5]:
                print(f"  - {error}")
        else:
            print(f"✓ Import completed successfully")
        
        return response
    
    def update_product_inventory(self, product_id, price=None, availability=None):
        """Update product price and availability."""
        
        product_name = f"{self.branch_path}/products/{product_id}"
        
        try:
            product = self.product_service.get_product(name=product_name)
        except Exception as e:
            print(f"✗ Product not found: {product_id}")
            return None
        
        from google.protobuf import field_mask_pb2
        
        update_mask_paths = []
        
        if price is not None:
            product.price_info.price = price
            update_mask_paths.append('price_info.price')
        
        if availability is not None:
            product.availability = availability
            update_mask_paths.append('availability')
        
        update_mask = field_mask_pb2.FieldMask(paths=update_mask_paths)
        
        request = retail_v2.UpdateProductRequest(
            product=product,
            update_mask=update_mask,
        )
        
        response = self.product_service.update_product(request=request)
        
        print(f"✓ Updated product: {product_id}")
        
        return response

# Example usage
client = RetailSearchClient(
    project_id='my-project',
    location='global',
    catalog_id='default_catalog'
)

# Create single product
# client.create_product(
#     product_id='PROD-001',
#     title='Premium Wireless Headphones',
#     categories=['Electronics', 'Audio', 'Headphones'],
#     price=199.99,
#     images=['https://example.com/headphones.jpg'],
#     description='High-quality wireless headphones with noise cancellation',
#     attributes={
#         'brand': 'TechBrand',
#         'color': 'Black',
#         'connectivity': 'Bluetooth 5.0',
#     }
# )

# Batch import from Cloud Storage
# client.batch_import_products('gs://my-bucket/products.json')
```

---

## 2. Search Implementation

### Product Search

```python
from google.cloud.retail_v2 import SearchRequest, SearchServiceClient

class ProductSearch:
    """Implement product search functionality."""
    
    def __init__(self, project_id, location='global', catalog_id='default_catalog'):
        self.project_id = project_id
        self.location = location
        self.catalog_id = catalog_id
        
        self.search_service = SearchServiceClient()
        
        self.placement = (
            f"projects/{project_id}/locations/{location}/catalogs/{catalog_id}/"
            f"placements/default_search"
        )
    
    def search(
        self,
        query,
        visitor_id,
        page_size=20,
        offset=0,
        filter_expression='',
        order_by='',
        facets=None
    ):
        """Search for products."""
        
        request = SearchRequest(
            placement=self.placement,
            query=query,
            visitor_id=visitor_id,
            page_size=page_size,
            offset=offset,
            filter=filter_expression,
            order_by=order_by,
        )
        
        # Add facets
        if facets:
            request.facet_specs.extend([
                SearchRequest.FacetSpec(
                    facet_key=SearchRequest.FacetSpec.FacetKey(key=facet)
                )
                for facet in facets
            ])
        
        print(f"Searching for: '{query}'")
        
        try:
            response = self.search_service.search(request=request)
            
            results = []
            
            for result in response.results:
                product = result.product
                
                results.append({
                    'id': product.id,
                    'title': product.title,
                    'categories': list(product.categories),
                    'price': product.price_info.price,
                    'currency': product.price_info.currency_code,
                    'images': [img.uri for img in product.images],
                    'attributes': dict(product.attributes),
                })
            
            print(f"✓ Found {len(results)} results")
            
            # Display results
            for i, result in enumerate(results[:5], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Price: ${result['price']} {result['currency']}")
                print(f"   Categories: {' > '.join(result['categories'])}")
            
            # Display facets
            if response.facets:
                print("\n=== Facets ===\n")
                
                for facet in response.facets:
                    print(f"{facet.key}:")
                    for value in facet.values[:5]:  # Top 5
                        print(f"  {value.value} ({value.count})")
            
            return {
                'results': results,
                'total_count': response.total_size,
                'facets': response.facets,
            }
            
        except Exception as e:
            print(f"✗ Search error: {e}")
            return {'results': [], 'total_count': 0, 'facets': []}
    
    def search_with_filters(self, query, visitor_id, filters):
        """Search with faceted filters.
        
        Filters format:
        {
            'category': ['Electronics', 'Audio'],
            'price': {'min': 50, 'max': 500},
            'brand': ['BrandA', 'BrandB'],
        }
        """
        
        filter_parts = []
        
        # Category filter
        if 'category' in filters:
            categories = ' OR '.join(f'"{cat}"' for cat in filters['category'])
            filter_parts.append(f"categories: ANY({categories})")
        
        # Price range filter
        if 'price' in filters:
            if 'min' in filters['price']:
                filter_parts.append(f"price_info.price >= {filters['price']['min']}")
            if 'max' in filters['price']:
                filter_parts.append(f"price_info.price <= {filters['price']['max']}")
        
        # Brand filter
        if 'brand' in filters:
            brands = ' OR '.join(f'"{brand}"' for brand in filters['brand'])
            filter_parts.append(f"attributes.brand: ANY({brands})")
        
        # Availability filter
        if 'availability' in filters:
            filter_parts.append(f"availability: \"{filters['availability']}\"")
        
        filter_expression = ' AND '.join(f"({part})" for part in filter_parts)
        
        print(f"Filter: {filter_expression}")
        
        return self.search(
            query=query,
            visitor_id=visitor_id,
            filter_expression=filter_expression,
            facets=['categories', 'attributes.brand', 'price_info.price']
        )

# Example usage
search_client = ProductSearch(project_id='my-project')

# Simple search
# results = search_client.search(
#     query='wireless headphones',
#     visitor_id='user-12345',
#     page_size=20
# )

# Search with filters
filters = {
    'category': ['Electronics', 'Audio'],
    'price': {'min': 50, 'max': 300},
    'availability': 'IN_STOCK',
}

# results = search_client.search_with_filters(
#     query='headphones',
#     visitor_id='user-12345',
#     filters=filters
# )
```

---

## 3. Autocomplete

```python
from google.cloud.retail_v2 import CompleteQueryRequest, CompletionServiceClient

class AutocompleteService:
    """Implement autocomplete functionality."""
    
    def __init__(self, project_id, location='global', catalog_id='default_catalog'):
        self.project_id = project_id
        self.location = location
        self.catalog_id = catalog_id
        
        self.completion_service = CompletionServiceClient()
        
        self.catalog_path = (
            f"projects/{project_id}/locations/{location}/catalogs/{catalog_id}"
        )
    
    def get_suggestions(self, query, visitor_id, max_suggestions=5):
        """Get autocomplete suggestions."""
        
        request = CompleteQueryRequest(
            catalog=self.catalog_path,
            query=query,
            visitor_id=visitor_id,
            max_suggestions=max_suggestions,
        )
        
        try:
            response = self.completion_service.complete_query(request=request)
            
            suggestions = []
            
            for result in response.completion_results:
                suggestions.append({
                    'suggestion': result.suggestion,
                    'attributes': dict(result.attributes) if result.attributes else {},
                })
            
            print(f"Autocomplete for '{query}':")
            for i, sugg in enumerate(suggestions, 1):
                print(f"  {i}. {sugg['suggestion']}")
            
            return suggestions
            
        except Exception as e:
            print(f"✗ Autocomplete error: {e}")
            return []
    
    def get_recent_searches(self, visitor_id, max_results=10):
        """Get user's recent searches."""
        
        # This would typically come from user event history
        # For demo purposes, returning mock data
        
        recent_searches = [
            'wireless headphones',
            'laptop bag',
            'usb cable',
        ]
        
        return recent_searches[:max_results]

# Example usage
autocomplete = AutocompleteService(project_id='my-project')

# suggestions = autocomplete.get_suggestions(
#     query='wire',
#     visitor_id='user-12345',
#     max_suggestions=5
# )
```

---

## 4. User Event Tracking

```python
from google.cloud.retail_v2 import UserEventServiceClient, UserEvent

class RetailEventTracker:
    """Track user events for personalization."""
    
    def __init__(self, project_id, location='global', catalog_id='default_catalog'):
        self.project_id = project_id
        self.location = location
        self.catalog_id = catalog_id
        
        self.user_event_service = UserEventServiceClient()
        
        self.parent = (
            f"projects/{project_id}/locations/{location}/catalogs/{catalog_id}"
        )
    
    def track_search(self, visitor_id, query, products_viewed=None):
        """Track search event."""
        
        if products_viewed is None:
            products_viewed = []
        
        user_event = UserEvent(
            event_type='search',
            visitor_id=visitor_id,
            search_query=query,
            product_details=[
                UserEvent.ProductDetail(product={'id': pid})
                for pid in products_viewed
            ],
        )
        
        request = retail_v2.WriteUserEventRequest(
            parent=self.parent,
            user_event=user_event,
        )
        
        response = self.user_event_service.write_user_event(request=request)
        
        print(f"✓ Tracked search: '{query}' by {visitor_id}")
        
        return response
    
    def track_product_view(self, visitor_id, product_id):
        """Track product detail view."""
        
        user_event = UserEvent(
            event_type='detail-page-view',
            visitor_id=visitor_id,
            product_details=[
                UserEvent.ProductDetail(product={'id': product_id})
            ],
        )
        
        request = retail_v2.WriteUserEventRequest(
            parent=self.parent,
            user_event=user_event,
        )
        
        response = self.user_event_service.write_user_event(request=request)
        
        print(f"✓ Tracked product view: {product_id} by {visitor_id}")
        
        return response
    
    def track_add_to_cart(self, visitor_id, product_id, quantity=1):
        """Track add to cart event."""
        
        user_event = UserEvent(
            event_type='add-to-cart',
            visitor_id=visitor_id,
            product_details=[
                UserEvent.ProductDetail(
                    product={'id': product_id},
                    quantity=quantity
                )
            ],
        )
        
        request = retail_v2.WriteUserEventRequest(
            parent=self.parent,
            user_event=user_event,
        )
        
        response = self.user_event_service.write_user_event(request=request)
        
        print(f"✓ Tracked add-to-cart: {product_id} (qty: {quantity}) by {visitor_id}")
        
        return response
    
    def track_purchase(self, visitor_id, product_ids, revenue, currency='USD', transaction_id=None):
        """Track purchase event."""
        
        from google.cloud.retail_v2 import PurchaseTransaction
        import time
        
        if transaction_id is None:
            transaction_id = f"ORDER-{int(time.time())}"
        
        user_event = UserEvent(
            event_type='purchase-complete',
            visitor_id=visitor_id,
            product_details=[
                UserEvent.ProductDetail(product={'id': pid})
                for pid in product_ids
            ],
            purchase_transaction=PurchaseTransaction(
                id=transaction_id,
                revenue=revenue,
                currency_code=currency,
            ),
        )
        
        request = retail_v2.WriteUserEventRequest(
            parent=self.parent,
            user_event=user_event,
        )
        
        response = self.user_event_service.write_user_event(request=request)
        
        print(f"✓ Tracked purchase: {len(product_ids)} items, ${revenue} by {visitor_id}")
        
        return response

# Example usage
event_tracker = RetailEventTracker(project_id='my-project')

# Track user journey
visitor_id = 'user-12345'

# event_tracker.track_search(visitor_id, 'wireless headphones')
# event_tracker.track_product_view(visitor_id, 'PROD-001')
# event_tracker.track_add_to_cart(visitor_id, 'PROD-001', quantity=1)
# event_tracker.track_purchase(visitor_id, ['PROD-001'], revenue=199.99)
```

---

## 5. Personalized Browse

```python
class PersonalizedBrowse:
    """Implement personalized browse experiences."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def get_personalized_products(
        self,
        visitor_id,
        category=None,
        page_size=20,
        boost_spec=None
    ):
        """Get personalized product recommendations for browse."""
        
        from google.cloud.retail_v2 import SearchRequest
        
        # Build filter
        filter_expression = ''
        if category:
            filter_expression = f'categories: "{category}"'
        
        request = SearchRequest(
            placement=self.search_client.placement,
            visitor_id=visitor_id,
            page_size=page_size,
            filter=filter_expression,
        )
        
        # Add personalization boost
        if boost_spec:
            request.boost_spec.CopyFrom(boost_spec)
        
        response = self.search_client.search_service.search(request=request)
        
        products = []
        
        for result in response.results:
            product = result.product
            products.append({
                'id': product.id,
                'title': product.title,
                'price': product.price_info.price,
            })
        
        print(f"✓ Got {len(products)} personalized products for {visitor_id}")
        
        return products
    
    def get_trending_products(self, category=None, page_size=20):
        """Get trending products."""
        
        # Order by popularity or recent purchases
        order_by = 'popularity desc'
        
        filter_expression = ''
        if category:
            filter_expression = f'categories: "{category}"'
        
        results = self.search_client.search(
            query='*',  # All products
            visitor_id='anonymous',
            page_size=page_size,
            filter_expression=filter_expression,
            order_by=order_by
        )
        
        return results['results']
    
    def get_new_arrivals(self, days=30, page_size=20):
        """Get new arrival products."""
        
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        filter_expression = f'publish_time >= "{cutoff_str}"'
        
        order_by = 'publish_time desc'
        
        results = self.search_client.search(
            query='*',
            visitor_id='anonymous',
            page_size=page_size,
            filter_expression=filter_expression,
            order_by=order_by
        )
        
        return results['results']

# Example usage
search_client = ProductSearch(project_id='my-project')
browse = PersonalizedBrowse(search_client)

# personalized = browse.get_personalized_products(
#     visitor_id='user-12345',
#     category='Electronics',
#     page_size=20
# )

# trending = browse.get_trending_products(category='Electronics')
# new_arrivals = browse.get_new_arrivals(days=30)
```

---

## 6. Search Tuning

```python
class SearchTuning:
    """Configure and tune search relevance."""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.boost_rules = []
        self.filter_rules = []
    
    def add_boost_rule(self, condition, boost_factor):
        """Add boost rule to promote certain products.
        
        Example:
        condition = 'attributes.brand = "Premium"'
        boost_factor = 2.0
        """
        
        self.boost_rules.append({
            'condition': condition,
            'boost': boost_factor,
        })
        
        print(f"✓ Added boost rule: {condition} (boost: {boost_factor}x)")
    
    def add_filter_rule(self, filter_action, condition):
        """Add filter rule.
        
        Actions: 'exclude', 'include', 'redirect'
        """
        
        self.filter_rules.append({
            'action': filter_action,
            'condition': condition,
        })
        
        print(f"✓ Added filter rule: {filter_action} where {condition}")
    
    def configure_synonyms(self, synonyms):
        """Configure search synonyms.
        
        Example:
        synonyms = {
            'laptop': ['notebook', 'portable computer'],
            'phone': ['mobile', 'smartphone', 'cell phone'],
        }
        """
        
        print(f"✓ Configured {len(synonyms)} synonym groups")
        
        return synonyms
    
    def configure_redirects(self, redirects):
        """Configure search redirects.
        
        Example:
        redirects = {
            'support': '/support-page',
            'return policy': '/returns',
        }
        """
        
        print(f"✓ Configured {len(redirects)} search redirects")
        
        return redirects

# Example tuning configuration
tuning = SearchTuning(project_id='my-project')

# Boost premium brands
tuning.add_boost_rule('attributes.brand = "Premium"', boost_factor=2.0)

# Boost new products
tuning.add_boost_rule('publish_time >= "2024-01-01"', boost_factor=1.5)

# Exclude out-of-stock
tuning.add_filter_rule('exclude', 'availability = "OUT_OF_STOCK"')

# Configure synonyms
synonyms = {
    'laptop': ['notebook', 'portable computer', 'laptop computer'],
    'phone': ['mobile', 'smartphone', 'cell phone', 'mobile phone'],
    'headphones': ['earphones', 'headset', 'earbuds'],
}

tuning.configure_synonyms(synonyms)
```

---

## 7. Analytics and Monitoring

```python
class RetailAnalytics:
    """Monitor and analyze search performance."""
    
    def __init__(self):
        self.metrics = []
    
    def calculate_search_metrics(self, events):
        """Calculate search performance metrics."""
        
        from collections import Counter
        
        total_searches = 0
        searches_with_results = 0
        searches_with_clicks = 0
        
        query_counts = Counter()
        zero_result_queries = []
        
        for event in events:
            if event['type'] == 'search':
                total_searches += 1
                query = event['query']
                query_counts[query] += 1
                
                if event.get('results_count', 0) > 0:
                    searches_with_results += 1
                else:
                    zero_result_queries.append(query)
                
                if event.get('clicked', False):
                    searches_with_clicks += 1
        
        # Calculate metrics
        result_rate = searches_with_results / total_searches if total_searches > 0 else 0
        click_rate = searches_with_clicks / total_searches if total_searches > 0 else 0
        
        print("\n=== Search Metrics ===\n")
        print(f"Total Searches: {total_searches:,}")
        print(f"Searches with Results: {searches_with_results:,} ({result_rate:.1%})")
        print(f"Search Click Rate: {click_rate:.1%}")
        print(f"Zero-Result Queries: {len(zero_result_queries)}")
        
        # Top queries
        print("\nTop 10 Queries:")
        for query, count in query_counts.most_common(10):
            print(f"  {query}: {count}")
        
        # Zero-result queries
        if zero_result_queries:
            print(f"\nSample Zero-Result Queries:")
            for query in list(set(zero_result_queries))[:5]:
                print(f"  - {query}")
        
        return {
            'total_searches': total_searches,
            'result_rate': result_rate,
            'click_rate': click_rate,
            'top_queries': query_counts.most_common(10),
            'zero_result_queries': zero_result_queries,
        }
    
    def analyze_conversion_funnel(self, events):
        """Analyze conversion funnel."""
        
        funnel = {
            'searches': 0,
            'product_views': 0,
            'add_to_cart': 0,
            'purchases': 0,
        }
        
        for event in events:
            event_type = event['type']
            
            if event_type == 'search':
                funnel['searches'] += 1
            elif event_type == 'product_view':
                funnel['product_views'] += 1
            elif event_type == 'add_to_cart':
                funnel['add_to_cart'] += 1
            elif event_type == 'purchase':
                funnel['purchases'] += 1
        
        # Calculate conversion rates
        print("\n=== Conversion Funnel ===\n")
        
        print(f"Searches: {funnel['searches']:,}")
        
        if funnel['searches'] > 0:
            view_rate = funnel['product_views'] / funnel['searches']
            print(f"Product Views: {funnel['product_views']:,} ({view_rate:.1%})")
            
            cart_rate = funnel['add_to_cart'] / funnel['searches']
            print(f"Add to Cart: {funnel['add_to_cart']:,} ({cart_rate:.1%})")
            
            purchase_rate = funnel['purchases'] / funnel['searches']
            print(f"Purchases: {funnel['purchases']:,} ({purchase_rate:.1%})")
        
        return funnel
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Retail Search API
- [ ] Import product catalog
- [ ] Configure search placement
- [ ] Set up event tracking
- [ ] Test with sample queries

### Search Optimization
- [ ] Configure synonyms
- [ ] Set up boost rules
- [ ] Implement faceted search
- [ ] Configure autocomplete
- [ ] Set up query redirects

### Personalization
- [ ] Track user events
- [ ] Enable personalized browse
- [ ] Implement recommendations
- [ ] Configure A/B tests
- [ ] Monitor performance

### Production
- [ ] Monitor zero-result queries
- [ ] Analyze conversion funnel
- [ ] Track search metrics
- [ ] Optimize relevance
- [ ] Handle edge cases

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
