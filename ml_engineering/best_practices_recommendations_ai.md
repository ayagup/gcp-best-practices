# Recommendations AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Recommendations AI provides personalized product recommendations using machine learning models trained on catalog data and user behavior, optimizing for clicks, conversions, and revenue.

---

## 1. Catalog Management

### Upload Product Catalog

```python
from google.cloud import recommendationengine_v1beta1 as recommendationengine
from google.cloud.recommendationengine_v1beta1 import ProductCatalogService
from google.protobuf import field_mask_pb2

class RecommendationsClient:
    """Client for Recommendations AI operations."""
    
    def __init__(self, project_id, location='global'):
        self.project_id = project_id
        self.location = location
        self.catalog_client = ProductCatalogService.ProductCatalogServiceClient()
        self.catalog_path = f"projects/{project_id}/locations/{location}/catalogs/default_catalog"
    
    def create_product(self, product_id, title, categories, price, availability='IN_STOCK'):
        """Create or update a product in catalog."""
        
        from google.cloud.recommendationengine_v1beta1 import ProductCatalogItem, PriceInfo
        
        product = ProductCatalogItem(
            id=product_id,
            title=title,
            categories=categories,
            price_info=PriceInfo(
                price=price,
                currency_code='USD',
            ),
            stock_state=availability,
        )
        
        parent = f"{self.catalog_path}/catalogItems"
        
        request = recommendationengine.CreateCatalogItemRequest(
            parent=parent,
            catalog_item=product
        )
        
        try:
            response = self.catalog_client.create_catalog_item(request=request)
            print(f"✓ Created product: {product_id}")
            return response
        except Exception as e:
            print(f"✗ Error creating product {product_id}: {e}")
            return None
    
    def batch_create_products(self, products):
        """Create multiple products in batch."""
        
        print(f"Creating {len(products)} products...")
        
        results = {'success': 0, 'failed': 0}
        
        for product in products:
            result = self.create_product(
                product_id=product['id'],
                title=product['title'],
                categories=product['categories'],
                price=product['price'],
                availability=product.get('availability', 'IN_STOCK')
            )
            
            if result:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        print(f"\nBatch creation complete:")
        print(f"  Success: {results['success']}")
        print(f"  Failed: {results['failed']}")
        
        return results
    
    def update_product(self, product_id, updates):
        """Update product information."""
        
        # Get existing product
        name = f"{self.catalog_path}/catalogItems/{product_id}"
        
        try:
            product = self.catalog_client.get_catalog_item(name=name)
        except Exception as e:
            print(f"✗ Product not found: {product_id}")
            return None
        
        # Apply updates
        update_mask_paths = []
        
        if 'title' in updates:
            product.title = updates['title']
            update_mask_paths.append('title')
        
        if 'price' in updates:
            product.price_info.price = updates['price']
            update_mask_paths.append('price_info.price')
        
        if 'availability' in updates:
            product.stock_state = updates['availability']
            update_mask_paths.append('stock_state')
        
        update_mask = field_mask_pb2.FieldMask(paths=update_mask_paths)
        
        request = recommendationengine.UpdateCatalogItemRequest(
            catalog_item=product,
            update_mask=update_mask
        )
        
        response = self.catalog_client.update_catalog_item(request=request)
        
        print(f"✓ Updated product: {product_id}")
        
        return response

# Example usage
client = RecommendationsClient(project_id='my-project')

# Create single product
# client.create_product(
#     product_id='SKU-001',
#     title='Premium Laptop',
#     categories=['Electronics', 'Computers', 'Laptops'],
#     price=1299.99,
#     availability='IN_STOCK'
# )

# Batch create products
products = [
    {
        'id': 'SKU-001',
        'title': 'Premium Laptop',
        'categories': ['Electronics', 'Computers'],
        'price': 1299.99
    },
    {
        'id': 'SKU-002',
        'title': 'Wireless Mouse',
        'categories': ['Electronics', 'Accessories'],
        'price': 29.99
    },
    {
        'id': 'SKU-003',
        'title': 'USB-C Cable',
        'categories': ['Electronics', 'Cables'],
        'price': 12.99
    },
]

# client.batch_create_products(products)
```

---

## 2. User Event Tracking

### Track User Events

```python
from google.cloud.recommendationengine_v1beta1 import UserEventService, UserEvent

class EventTracker:
    """Track user events for recommendations."""
    
    def __init__(self, project_id, location='global'):
        self.project_id = project_id
        self.location = location
        self.event_client = UserEventService.UserEventServiceClient()
        self.catalog_path = f"projects/{project_id}/locations/{location}/catalogs/default_catalog"
        self.event_store_path = f"{self.catalog_path}/eventStores/default_event_store"
    
    def track_product_detail_view(self, user_id, product_id, session_id=None):
        """Track when user views product details."""
        
        event = UserEvent(
            event_type='detail-page-view',
            user_info={'visitor_id': user_id},
            product_detail={'id': product_id},
        )
        
        if session_id:
            event.user_info.user_id = session_id
        
        request = recommendationengine.WriteUserEventRequest(
            parent=self.event_store_path,
            user_event=event
        )
        
        response = self.event_client.write_user_event(request=request)
        
        print(f"✓ Tracked detail view: User {user_id} viewed {product_id}")
        
        return response
    
    def track_add_to_cart(self, user_id, product_id, quantity=1):
        """Track when user adds product to cart."""
        
        event = UserEvent(
            event_type='add-to-cart',
            user_info={'visitor_id': user_id},
            product_detail={'id': product_id, 'quantity': quantity},
        )
        
        request = recommendationengine.WriteUserEventRequest(
            parent=self.event_store_path,
            user_event=event
        )
        
        response = self.event_client.write_user_event(request=request)
        
        print(f"✓ Tracked add-to-cart: User {user_id} added {product_id} (qty: {quantity})")
        
        return response
    
    def track_purchase(self, user_id, product_ids, revenue, currency='USD'):
        """Track purchase conversion."""
        
        purchase_transaction = {
            'id': f"ORDER-{user_id}-{int(time.time())}",
            'revenue': revenue,
            'currency_code': currency,
        }
        
        event = UserEvent(
            event_type='purchase-complete',
            user_info={'visitor_id': user_id},
            product_details=[{'id': pid} for pid in product_ids],
            purchase_transaction=purchase_transaction,
        )
        
        request = recommendationengine.WriteUserEventRequest(
            parent=self.event_store_path,
            user_event=event
        )
        
        response = self.event_client.write_user_event(request=request)
        
        print(f"✓ Tracked purchase: User {user_id} purchased {len(product_ids)} items (${revenue})")
        
        return response
    
    def track_home_page_view(self, user_id):
        """Track home page view."""
        
        event = UserEvent(
            event_type='home-page-view',
            user_info={'visitor_id': user_id},
        )
        
        request = recommendationengine.WriteUserEventRequest(
            parent=self.event_store_path,
            user_event=event
        )
        
        response = self.event_client.write_user_event(request=request)
        
        print(f"✓ Tracked home page view: User {user_id}")
        
        return response

# Example usage
import time

tracker = EventTracker(project_id='my-project')

# Track user journey
user_id = 'user-12345'

# tracker.track_home_page_view(user_id)
# tracker.track_product_detail_view(user_id, 'SKU-001')
# tracker.track_add_to_cart(user_id, 'SKU-001', quantity=1)
# tracker.track_purchase(user_id, ['SKU-001'], revenue=1299.99)
```

---

## 3. Get Recommendations

### Predict Products

```python
from google.cloud.recommendationengine_v1beta1 import PredictionService

class RecommendationPredictor:
    """Get product recommendations."""
    
    def __init__(self, project_id, location='global'):
        self.project_id = project_id
        self.location = location
        self.prediction_client = PredictionService.PredictionServiceClient()
        self.placement_path = (
            f"projects/{project_id}/locations/{location}/catalogs/default_catalog/"
            f"eventStores/default_event_store/placements/recently_viewed_default"
        )
    
    def get_recommendations(
        self,
        user_id,
        placement_id='recently_viewed_default',
        max_results=10,
        product_id=None
    ):
        """Get personalized recommendations for user."""
        
        placement = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"catalogs/default_catalog/eventStores/default_event_store/"
            f"placements/{placement_id}"
        )
        
        user_event = {'event_type': 'detail-page-view'}
        
        if user_id:
            user_event['user_info'] = {'visitor_id': user_id}
        
        if product_id:
            user_event['product_event_detail'] = {'product_details': [{'id': product_id}]}
        
        request = recommendationengine.PredictRequest(
            name=placement,
            user_event=user_event,
            page_size=max_results,
        )
        
        try:
            response = self.prediction_client.predict(request=request)
            
            recommendations = []
            
            for result in response:
                for item in result.results:
                    recommendations.append({
                        'product_id': item.id,
                        'metadata': dict(item.metadata) if item.metadata else {},
                    })
            
            print(f"✓ Got {len(recommendations)} recommendations for user {user_id}")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['product_id']}")
            
            return recommendations
            
        except Exception as e:
            print(f"✗ Error getting recommendations: {e}")
            return []
    
    def get_similar_items(self, product_id, max_results=10):
        """Get products similar to given product."""
        
        placement = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"catalogs/default_catalog/eventStores/default_event_store/"
            f"placements/others_you_may_like_default"
        )
        
        user_event = {
            'event_type': 'detail-page-view',
            'product_event_detail': {
                'product_details': [{'id': product_id}]
            }
        }
        
        request = recommendationengine.PredictRequest(
            name=placement,
            user_event=user_event,
            page_size=max_results,
        )
        
        response = self.prediction_client.predict(request=request)
        
        similar_items = []
        
        for result in response:
            for item in result.results:
                similar_items.append(item.id)
        
        print(f"✓ Found {len(similar_items)} similar items for {product_id}")
        
        return similar_items
    
    def get_frequently_bought_together(self, product_id, max_results=5):
        """Get products frequently bought together."""
        
        placement = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"catalogs/default_catalog/eventStores/default_event_store/"
            f"placements/frequently_bought_together_default"
        )
        
        user_event = {
            'event_type': 'shopping-cart-page-view',
            'product_event_detail': {
                'product_details': [{'id': product_id}]
            }
        }
        
        request = recommendationengine.PredictRequest(
            name=placement,
            user_event=user_event,
            page_size=max_results,
        )
        
        response = self.prediction_client.predict(request=request)
        
        items = []
        
        for result in response:
            for item in result.results:
                items.append(item.id)
        
        print(f"✓ Found {len(items)} frequently bought together items")
        
        return items

# Example usage
predictor = RecommendationPredictor(project_id='my-project')

# Get personalized recommendations
# recs = predictor.get_recommendations(user_id='user-12345', max_results=10)

# Get similar items
# similar = predictor.get_similar_items(product_id='SKU-001', max_results=5)

# Get frequently bought together
# fbt = predictor.get_frequently_bought_together(product_id='SKU-001', max_results=5)
```

---

## 4. Optimization Objectives

### Configure Model Objectives

```python
class ModelOptimizer:
    """Configure recommendation model objectives."""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.objectives = {}
    
    def set_optimization_objective(self, objective_type='maximize_cvr'):
        """Set model optimization objective.
        
        Objectives:
        - maximize_cvr: Maximize conversion rate
        - maximize_ctr: Maximize click-through rate
        - maximize_revenue: Maximize revenue per recommendation
        """
        
        self.objectives['type'] = objective_type
        
        print(f"✓ Set optimization objective: {objective_type}")
        
        return self.objectives
    
    def configure_business_rules(self, rules):
        """Configure business rules for recommendations.
        
        Rules can include:
        - Boost specific categories
        - Filter out-of-stock items
        - Promote new arrivals
        - Apply price range filters
        """
        
        self.objectives['business_rules'] = rules
        
        print(f"✓ Configured {len(rules)} business rules")
        
        return self.objectives
    
    def set_diversity_level(self, level='medium'):
        """Set recommendation diversity.
        
        Levels:
        - low: More similar items
        - medium: Balanced
        - high: More diverse items
        """
        
        self.objectives['diversity'] = level
        
        print(f"✓ Set diversity level: {level}")
        
        return self.objectives

# Example configuration
optimizer = ModelOptimizer(project_id='my-project')

# optimizer.set_optimization_objective('maximize_revenue')

business_rules = [
    {'rule': 'filter', 'field': 'stock_state', 'value': 'IN_STOCK'},
    {'rule': 'boost', 'field': 'categories', 'value': 'New Arrivals', 'boost': 1.5},
    {'rule': 'filter', 'field': 'price', 'min': 10, 'max': 1000},
]

# optimizer.configure_business_rules(business_rules)
# optimizer.set_diversity_level('medium')
```

---

## 5. A/B Testing

```python
import random

class ABTestManager:
    """Manage A/B tests for recommendations."""
    
    def __init__(self):
        self.experiments = {}
        self.results = []
    
    def create_experiment(self, name, variants, traffic_split=None):
        """Create A/B test experiment.
        
        Args:
            name: Experiment name
            variants: List of variant configurations
            traffic_split: Dict of variant -> traffic percentage
        """
        
        if traffic_split is None:
            # Equal split
            split = 1.0 / len(variants)
            traffic_split = {v['id']: split for v in variants}
        
        self.experiments[name] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'active': True,
        }
        
        print(f"✓ Created experiment: {name}")
        print(f"  Variants: {len(variants)}")
        print(f"  Traffic split: {traffic_split}")
        
        return self.experiments[name]
    
    def assign_variant(self, experiment_name, user_id):
        """Assign user to experiment variant."""
        
        experiment = self.experiments.get(experiment_name)
        
        if not experiment or not experiment['active']:
            return None
        
        # Deterministic assignment based on user_id
        hash_value = hash(user_id) % 100 / 100
        
        cumulative = 0
        for variant in experiment['variants']:
            cumulative += experiment['traffic_split'][variant['id']]
            if hash_value < cumulative:
                return variant
        
        return experiment['variants'][-1]
    
    def track_experiment_event(self, experiment_name, user_id, variant_id, event_type, value=None):
        """Track event for experiment analysis."""
        
        event = {
            'experiment': experiment_name,
            'user_id': user_id,
            'variant': variant_id,
            'event_type': event_type,
            'value': value,
            'timestamp': time.time(),
        }
        
        self.results.append(event)
        
        return event
    
    def analyze_experiment(self, experiment_name):
        """Analyze experiment results."""
        
        experiment_results = [
            r for r in self.results
            if r['experiment'] == experiment_name
        ]
        
        if not experiment_results:
            print(f"No results for experiment: {experiment_name}")
            return None
        
        # Group by variant
        from collections import defaultdict
        
        variant_stats = defaultdict(lambda: {
            'views': 0,
            'clicks': 0,
            'conversions': 0,
            'revenue': 0,
        })
        
        for result in experiment_results:
            variant = result['variant']
            event_type = result['event_type']
            
            if event_type == 'view':
                variant_stats[variant]['views'] += 1
            elif event_type == 'click':
                variant_stats[variant]['clicks'] += 1
            elif event_type == 'conversion':
                variant_stats[variant]['conversions'] += 1
                variant_stats[variant]['revenue'] += result.get('value', 0)
        
        # Calculate metrics
        print(f"\n=== Experiment Results: {experiment_name} ===\n")
        
        for variant, stats in variant_stats.items():
            ctr = stats['clicks'] / stats['views'] if stats['views'] > 0 else 0
            cvr = stats['conversions'] / stats['clicks'] if stats['clicks'] > 0 else 0
            avg_revenue = stats['revenue'] / stats['conversions'] if stats['conversions'] > 0 else 0
            
            print(f"Variant: {variant}")
            print(f"  Views: {stats['views']}")
            print(f"  Clicks: {stats['clicks']}")
            print(f"  CTR: {ctr:.2%}")
            print(f"  Conversions: {stats['conversions']}")
            print(f"  CVR: {cvr:.2%}")
            print(f"  Revenue: ${stats['revenue']:.2f}")
            print(f"  Avg Revenue: ${avg_revenue:.2f}\n")
        
        return variant_stats

# Example A/B test
ab_manager = ABTestManager()

variants = [
    {'id': 'control', 'config': {'model': 'default'}},
    {'id': 'variant_a', 'config': {'model': 'diversity_boost'}},
    {'id': 'variant_b', 'config': {'model': 'revenue_optimized'}},
]

# ab_manager.create_experiment(
#     name='recommendation_algorithm_test',
#     variants=variants,
#     traffic_split={'control': 0.33, 'variant_a': 0.33, 'variant_b': 0.34}
# )
```

---

## 6. Monitoring and Evaluation

```python
class RecommendationMetrics:
    """Track recommendation performance metrics."""
    
    def __init__(self):
        self.metrics = []
    
    def calculate_ctr(self, impressions, clicks):
        """Calculate click-through rate."""
        return clicks / impressions if impressions > 0 else 0
    
    def calculate_cvr(self, clicks, conversions):
        """Calculate conversion rate."""
        return conversions / clicks if clicks > 0 else 0
    
    def calculate_coverage(self, recommended_items, total_catalog_items):
        """Calculate catalog coverage."""
        unique_recommended = set(recommended_items)
        return len(unique_recommended) / total_catalog_items if total_catalog_items > 0 else 0
    
    def calculate_diversity(self, recommendations_list):
        """Calculate recommendation diversity."""
        # Average number of unique categories in recommendations
        unique_categories = []
        
        for recs in recommendations_list:
            categories = set(rec.get('category') for rec in recs)
            unique_categories.append(len(categories))
        
        return sum(unique_categories) / len(unique_categories) if unique_categories else 0
    
    def calculate_novelty(self, recommended_items, user_history):
        """Calculate recommendation novelty."""
        # Percentage of recommended items not in user history
        recommended_set = set(recommended_items)
        history_set = set(user_history)
        
        novel_items = recommended_set - history_set
        
        return len(novel_items) / len(recommended_set) if recommended_set else 0
    
    def generate_report(self, data):
        """Generate comprehensive metrics report."""
        
        print("\n=== Recommendation Metrics Report ===\n")
        
        ctr = self.calculate_ctr(data['impressions'], data['clicks'])
        cvr = self.calculate_cvr(data['clicks'], data['conversions'])
        coverage = self.calculate_coverage(data['recommended_items'], data['catalog_size'])
        
        print(f"Impressions: {data['impressions']:,}")
        print(f"Clicks: {data['clicks']:,}")
        print(f"CTR: {ctr:.2%}")
        print(f"\nConversions: {data['conversions']:,}")
        print(f"CVR: {cvr:.2%}")
        print(f"\nCatalog Coverage: {coverage:.2%}")
        print(f"Revenue: ${data.get('revenue', 0):,.2f}")
        
        return {
            'ctr': ctr,
            'cvr': cvr,
            'coverage': coverage,
        }
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Recommendations AI API
- [ ] Upload product catalog
- [ ] Configure event tracking
- [ ] Set up placements
- [ ] Test with sample data

### Event Tracking
- [ ] Track detail page views
- [ ] Track add-to-cart events
- [ ] Track purchase conversions
- [ ] Include user IDs consistently
- [ ] Handle anonymous users

### Optimization
- [ ] Set optimization objective
- [ ] Configure business rules
- [ ] Implement A/B testing
- [ ] Monitor performance metrics
- [ ] Iterate based on results

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
