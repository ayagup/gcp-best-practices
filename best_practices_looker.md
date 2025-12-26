# Looker Best Practices

*Last Updated: December 26, 2025*

## Overview

Looker is an enterprise-grade business intelligence (BI) and data analytics platform that enables organizations to explore, analyze, and share real-time business insights. Built on LookML, Looker's proprietary modeling language, it provides a semantic layer that ensures consistent metrics and definitions across the organization.

---

## 1. LookML Modeling

### Project Structure

**Best Practices:**
- Organize models logically by business domain
- Use consistent naming conventions
- Implement version control with Git
- Document models thoroughly

```lookml
# Directory structure
project_name/
├── models/
│   ├── sales.model.lkml
│   ├── marketing.model.lkml
│   └── finance.model.lkml
├── views/
│   ├── customers.view.lkml
│   ├── orders.view.lkml
│   └── products.view.lkml
├── explores/
│   ├── sales_analytics.explore.lkml
│   └── customer_journey.explore.lkml
├── dashboards/
│   ├── executive_summary.dashboard.lookml
│   └── sales_performance.dashboard.lookml
└── manifest.lkml

# manifest.lkml - Project configuration
project_name: "data_analytics"

# Define constants
constant: DATE_FORMAT {
  value: "%Y-%m-%d"
}

constant: CONNECTION_NAME {
  value: "bigquery_production"
}

# Remote include for shared models
remote_dependency: shared_lookml {
  url: "https://github.com/company/shared-lookml"
  ref: "main"
}

# Localization settings
localization_settings: {
  default_locale: en
  localization_level: permissive
}
```

### View Files

**Create Reusable Views:**
```lookml
# views/customers.view.lkml
view: customers {
  sql_table_name: `project.dataset.customers` ;;
  
  # Primary Key
  dimension: customer_id {
    primary_key: yes
    type: string
    sql: ${TABLE}.customer_id ;;
    description: "Unique customer identifier"
  }
  
  # Dimensions
  dimension: email {
    type: string
    sql: ${TABLE}.email ;;
    tags: ["pii", "sensitive"]
  }
  
  dimension: first_name {
    type: string
    sql: ${TABLE}.first_name ;;
    group_label: "Personal Information"
  }
  
  dimension: last_name {
    type: string
    sql: ${TABLE}.last_name ;;
    group_label: "Personal Information"
  }
  
  # Derived dimension
  dimension: full_name {
    type: string
    sql: CONCAT(${first_name}, ' ', ${last_name}) ;;
  }
  
  # Date dimensions
  dimension_group: created {
    type: time
    timeframes: [
      raw,
      time,
      date,
      week,
      month,
      quarter,
      year
    ]
    sql: ${TABLE}.created_at ;;
    datatype: timestamp
  }
  
  # Tier dimension
  dimension: age_tier {
    type: tier
    tiers: [18, 25, 35, 45, 55, 65]
    style: integer
    sql: ${TABLE}.age ;;
  }
  
  # Case when dimension
  dimension: customer_segment {
    type: string
    sql: CASE
      WHEN ${lifetime_value} >= 10000 THEN 'VIP'
      WHEN ${lifetime_value} >= 5000 THEN 'Premium'
      WHEN ${lifetime_value} >= 1000 THEN 'Standard'
      ELSE 'Basic'
    END ;;
  }
  
  # Yesno dimension
  dimension: is_active {
    type: yesno
    sql: ${TABLE}.status = 'active' ;;
  }
  
  # Location dimension
  dimension: location {
    type: location
    sql_latitude: ${TABLE}.latitude ;;
    sql_longitude: ${TABLE}.longitude ;;
  }
  
  # Measures
  measure: count {
    type: count
    drill_fields: [customer_id, full_name, email, created_date]
  }
  
  measure: total_lifetime_value {
    type: sum
    sql: ${TABLE}.lifetime_value ;;
    value_format_name: usd
    drill_fields: [customer_id, full_name, lifetime_value]
  }
  
  measure: average_lifetime_value {
    type: average
    sql: ${TABLE}.lifetime_value ;;
    value_format_name: usd_0
  }
  
  measure: distinct_count {
    type: count_distinct
    sql: ${customer_id} ;;
  }
  
  # Filtered measure
  measure: active_customers {
    type: count
    filters: [is_active: "yes"]
  }
  
  # Percent of total
  measure: percent_of_total_revenue {
    type: percent_of_total
    sql: ${total_lifetime_value} ;;
    value_format_name: percent_2
  }
}
```

### Model Files

**Define Explores and Relationships:**
```lookml
# models/sales.model.lkml
connection: "bigquery_production"

include: "/views/*.view.lkml"
include: "/dashboards/*.dashboard.lookml"

# Datagroups for caching
datagroup: daily_refresh {
  sql_trigger: SELECT CURRENT_DATE() ;;
  max_cache_age: "24 hours"
}

datagroup: hourly_refresh {
  sql_trigger: SELECT FLOOR(UNIX_SECONDS(CURRENT_TIMESTAMP()) / 3600) ;;
  max_cache_age: "1 hour"
}

# Explore definition
explore: orders {
  label: "Sales Orders"
  description: "Analyze order data with customer and product information"
  
  # Caching
  persist_with: daily_refresh
  
  # Access control
  access_filter: {
    field: customers.region
    user_attribute: allowed_regions
  }
  
  # Join customers
  join: customers {
    type: left_outer
    relationship: many_to_one
    sql_on: ${orders.customer_id} = ${customers.customer_id} ;;
    fields: [customers.customer_id, customers.full_name, customers.email]
  }
  
  # Join products
  join: products {
    type: left_outer
    relationship: many_to_one
    sql_on: ${orders.product_id} = ${products.product_id} ;;
  }
  
  # Join order items
  join: order_items {
    type: left_outer
    relationship: one_to_many
    sql_on: ${orders.order_id} = ${order_items.order_id} ;;
  }
  
  # Aggregate awareness
  aggregate_table: orders_by_date {
    query: {
      dimensions: [orders.created_date]
      measures: [orders.count, orders.total_revenue]
    }
    materialization: {
      datagroup_trigger: daily_refresh
    }
  }
  
  # SQL always where (global filter)
  sql_always_where: ${orders.created_date} >= '2020-01-01' ;;
  
  # SQL always having (for measures)
  sql_always_having: ${orders.total_revenue} > 0 ;;
  
  # Conditionally show fields
  conditionally_filter: {
    filters: [orders.created_date: "7 days"]
    unless: [orders.order_id]
  }
}

# Extended explore
explore: +orders {
  label: "Enhanced Sales Orders"
  
  # Additional join
  join: customer_segments {
    type: left_outer
    relationship: many_to_one
    sql_on: ${customers.customer_segment} = ${customer_segments.segment_name} ;;
  }
}

# Derived table explore
explore: sales_summary {
  from: orders
  view_name: orders
  
  # Always aggregate
  always_filter: {
    filters: [orders.created_date: "30 days"]
  }
}
```

### Advanced LookML Patterns

**Derived Tables:**
```lookml
# Native Derived Table (NDT)
view: customer_metrics {
  derived_table: {
    sql:
      SELECT
        customer_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(total_amount) AS lifetime_value,
        AVG(total_amount) AS avg_order_value,
        MIN(created_at) AS first_order_date,
        MAX(created_at) AS last_order_date
      FROM ${orders.SQL_TABLE_NAME}
      GROUP BY customer_id
    ;;
    
    datagroup_trigger: daily_refresh
    
    # Indexes for performance
    indexes: ["customer_id", "lifetime_value"]
  }
  
  dimension: customer_id {
    primary_key: yes
    type: string
    sql: ${TABLE}.customer_id ;;
  }
  
  dimension: order_count {
    type: number
    sql: ${TABLE}.order_count ;;
  }
  
  dimension: lifetime_value {
    type: number
    sql: ${TABLE}.lifetime_value ;;
    value_format_name: usd
  }
}

# Persistent Derived Table (PDT)
view: monthly_sales_summary {
  derived_table: {
    sql:
      SELECT
        DATE_TRUNC(created_at, MONTH) AS month,
        product_category,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value
      FROM ${orders.SQL_TABLE_NAME}
      WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
      GROUP BY 1, 2
    ;;
    
    # Persist as table in database
    persist_for: "24 hours"
    
    # Distribution key (for Redshift, etc.)
    distribution: "month"
    
    # Sort keys
    sortkeys: ["month", "product_category"]
  }
}

# Incremental PDT
view: daily_events_incremental {
  derived_table: {
    sql:
      SELECT
        DATE(event_timestamp) AS event_date,
        event_type,
        user_id,
        COUNT(*) AS event_count
      FROM events_table
      WHERE {% condition %} event_timestamp {% endcondition %}
      GROUP BY 1, 2, 3
    ;;
    
    # Incremental update
    increment_key: "event_date"
    increment_offset: 3
    
    datagroup_trigger: hourly_refresh
  }
}
```

---

## 2. Performance Optimization

### Query Performance

**Best Practices:**
- Use aggregate awareness
- Implement persistent derived tables
- Optimize join relationships
- Use symmetric aggregates

```lookml
# Aggregate awareness
explore: orders {
  # Define aggregate tables
  aggregate_table: orders_daily {
    query: {
      dimensions: [
        orders.created_date,
        customers.region,
        products.category
      ]
      measures: [
        orders.count,
        orders.total_revenue,
        orders.average_order_value
      ]
    }
    materialization: {
      datagroup_trigger: daily_refresh
    }
  }
  
  aggregate_table: orders_monthly {
    query: {
      dimensions: [
        orders.created_month,
        customers.region
      ]
      measures: [
        orders.count,
        orders.total_revenue
      ]
    }
    materialization: {
      datagroup_trigger: daily_refresh
    }
  }
}

# Symmetric aggregates
view: orders {
  measure: total_revenue {
    type: sum
    sql: ${TABLE}.total_amount ;;
  }
  
  # More efficient for large datasets
  measure: total_revenue_symmetric {
    type: sum
    sql: ${TABLE}.total_amount ;;
    symmetric_aggregates: yes
  }
}
```

### Caching Strategy

**Implement Effective Caching:**
```lookml
# Model-level caching
connection: "bigquery_production"

# Datagroups
datagroup: etl_refresh {
  sql_trigger:
    SELECT MAX(updated_at)
    FROM etl_metadata.pipeline_runs
    WHERE status = 'completed'
  ;;
  max_cache_age: "24 hours"
}

datagroup: real_time_refresh {
  sql_trigger: SELECT FLOOR(UNIX_SECONDS(CURRENT_TIMESTAMP()) / 300) ;;
  max_cache_age: "5 minutes"
}

# Explore-level caching
explore: real_time_orders {
  persist_with: real_time_refresh
  
  # Force no cache for specific scenarios
  persist_for: "0 seconds"  # Disable caching
}

explore: historical_sales {
  persist_with: etl_refresh
  
  # Cache for specific duration
  persist_for: "12 hours"
}

# View-level caching for PDTs
view: expensive_calculation {
  derived_table: {
    sql: ...complex query... ;;
    
    # Cache strategy
    persist_for: "24 hours"
    
    # Or use datagroup
    datagroup_trigger: etl_refresh
  }
}
```

---

## 3. Access Control and Security

### User Attributes

**Implement Row-Level Security:**
```lookml
# Define user attributes in Admin panel:
# - user_region (values: North, South, East, West)
# - user_department (values: Sales, Marketing, Finance)
# - user_level (values: analyst, manager, executive)

# Apply access filters in explore
explore: orders {
  # Filter by user's region
  access_filter: {
    field: customers.region
    user_attribute: user_region
  }
  
  # Filter by user's department
  access_filter: {
    field: orders.department
    user_attribute: user_department
  }
  
  # Multiple filters
  access_filter: {
    field: orders.visibility_level
    user_attribute: user_level
  }
}

# Use user attributes in dimensions
view: orders {
  dimension: can_view_details {
    type: yesno
    sql:
      {% if _user_attributes['user_level'] == 'executive' %}
        TRUE
      {% else %}
        FALSE
      {% endif %}
    ;;
  }
  
  # Conditionally show sensitive data
  dimension: customer_email {
    type: string
    sql:
      {% if _user_attributes['can_view_pii'] == 'yes' %}
        ${TABLE}.email
      {% else %}
        '***REDACTED***'
      {% endif %}
    ;;
  }
}
```

### Field-Level Security

**Control Field Visibility:**
```lookml
view: customers {
  # Hide field from all users
  dimension: ssn {
    type: string
    sql: ${TABLE}.ssn ;;
    hidden: yes
  }
  
  # Conditionally hide based on user attribute
  dimension: salary {
    type: number
    sql: ${TABLE}.salary ;;
    hidden: yes
    required_access_grants: [can_view_financial_data]
  }
  
  measure: total_revenue {
    type: sum
    sql: ${TABLE}.revenue ;;
    required_access_grants: [can_view_revenue]
  }
}

# Define access grants in model
access_grant: can_view_financial_data {
  user_attribute: department
  allowed_values: ["Finance", "Executive"]
}

access_grant: can_view_revenue {
  user_attribute: user_level
  allowed_values: ["manager", "executive"]
}

access_grant: can_view_pii {
  user_attribute: has_pii_access
  allowed_values: ["yes"]
}

# Apply grants in explore
explore: customers {
  required_access_grants: [can_view_financial_data]
}
```

---

## 4. Dashboard Design

### Dashboard LookML

**Create Interactive Dashboards:**
```lookml
# dashboards/sales_dashboard.dashboard.lookml
- dashboard: sales_performance
  title: Sales Performance Dashboard
  layout: newspaper
  preferred_viewer: dashboards-next
  description: "Executive dashboard for sales analytics"
  
  # Filters
  filters:
  - name: date_range
    title: Date Range
    type: date_filter
    default_value: 30 days
    allow_multiple_values: false
    required: true
    
  - name: region_filter
    title: Region
    type: field_filter
    explore: orders
    field: customers.region
    default_value: ""
    
  - name: product_category
    title: Product Category
    type: field_filter
    explore: orders
    field: products.category
  
  # Elements
  elements:
  - name: total_revenue
    title: Total Revenue
    type: single_value
    explore: orders
    measures: [orders.total_revenue]
    filters:
      orders.created_date: "{{ date_range._parameter_value }}"
      customers.region: "{{ region_filter._parameter_value }}"
    sorts: [orders.total_revenue desc]
    limit: 1
    custom_color_enabled: true
    show_single_value_title: true
    show_comparison: true
    comparison_type: progress_percentage
    comparison_reverse_colors: false
    comparison_label: vs. Previous Period
    
  - name: revenue_trend
    title: Revenue Trend
    type: looker_line
    explore: orders
    dimensions: [orders.created_date]
    measures: [orders.total_revenue, orders.count]
    filters:
      orders.created_date: "{{ date_range._parameter_value }}"
    sorts: [orders.created_date]
    limit: 500
    x_axis_gridlines: false
    y_axis_gridlines: true
    show_view_names: false
    show_y_axis_labels: true
    show_y_axis_ticks: true
    y_axis_tick_density: default
    y_axis_tick_density_custom: 5
    show_x_axis_label: true
    show_x_axis_ticks: true
    x_axis_label: Date
    y_axis_scale_mode: linear
    x_axis_reversed: false
    y_axis_reversed: false
    plot_size_by_field: false
    trellis: ''
    stacking: ''
    limit_displayed_rows: false
    legend_position: center
    point_style: none
    show_value_labels: false
    label_density: 25
    x_axis_scale: auto
    y_axis_combined: true
    show_null_points: true
    interpolation: linear
    
  - name: top_customers
    title: Top 10 Customers
    type: looker_grid
    explore: orders
    dimensions: [customers.customer_id, customers.full_name]
    measures: [orders.total_revenue, orders.count]
    filters:
      orders.created_date: "{{ date_range._parameter_value }}"
    sorts: [orders.total_revenue desc]
    limit: 10
    show_view_names: false
    show_row_numbers: true
    truncate_column_names: false
    hide_totals: false
    hide_row_totals: false
    table_theme: white
    limit_displayed_rows: false
    enable_conditional_formatting: true
    conditional_formatting:
    - type: high to low
      value: orders.total_revenue
      background_color: "#62bad4"
      font_color: "#FFFFFF"
      palette:
        name: Red to Yellow to Green
        colors:
        - "#F36254"
        - "#FCF040"
        - "#4FBC89"
    
  - name: sales_by_region
    title: Sales by Region
    type: looker_geo_choropleth
    explore: orders
    dimensions: [customers.region]
    measures: [orders.total_revenue]
    filters:
      orders.created_date: "{{ date_range._parameter_value }}"
    sorts: [orders.total_revenue desc]
    limit: 500
    map: usa
    map_projection: ''
    quantize_colors: false
    reverse_map_colors: false
```

### Dashboard Parameters

**Add Dynamic Parameters:**
```lookml
view: orders {
  # Define parameter
  parameter: metric_selector {
    type: unquoted
    allowed_value: {
      label: "Revenue"
      value: "revenue"
    }
    allowed_value: {
      label: "Order Count"
      value: "count"
    }
    allowed_value: {
      label: "Average Order Value"
      value: "aov"
    }
    default_value: "revenue"
  }
  
  # Dynamic measure based on parameter
  measure: dynamic_metric {
    label_from_parameter: metric_selector
    type: number
    sql:
      {% if metric_selector._parameter_value == 'revenue' %}
        ${total_revenue}
      {% elsif metric_selector._parameter_value == 'count' %}
        ${count}
      {% elsif metric_selector._parameter_value == 'aov' %}
        ${average_order_value}
      {% else %}
        NULL
      {% endif %}
    ;;
  }
  
  # Date range parameter
  parameter: date_granularity {
    type: unquoted
    allowed_value: {
      label: "Day"
      value: "day"
    }
    allowed_value: {
      label: "Week"
      value: "week"
    }
    allowed_value: {
      label: "Month"
      value: "month"
    }
    default_value: "day"
  }
  
  # Dynamic dimension
  dimension: dynamic_date {
    label_from_parameter: date_granularity
    type: string
    sql:
      {% if date_granularity._parameter_value == 'day' %}
        ${created_date}
      {% elsif date_granularity._parameter_value == 'week' %}
        ${created_week}
      {% elsif date_granularity._parameter_value == 'month' %}
        ${created_month}
      {% else %}
        ${created_date}
      {% endif %}
    ;;
  }
}
```

---

## 5. Embedded Analytics

### Embedding Configuration

**Best Practices:**
- Use SSO for authentication
- Implement proper access controls
- Monitor embedded usage
- Optimize for performance

```javascript
// Embed dashboard in web application
<script src="https://your-instance.looker.com/embed/embed.js"></script>

<div id="dashboard-container"></div>

<script>
  // Configuration
  const embedConfig = {
    url: 'https://your-instance.looker.com/embed/dashboards/sales_performance',
    container: '#dashboard-container',
    params: {
      // URL parameters
      'Date Range': '30 days',
      'Region': 'West',
    },
    // Appearance
    theme: 'custom_theme',
    // Filters
    filters: {
      'orders.created_date': '30 days',
      'customers.region': 'West'
    },
    // Sandbox attributes
    sandboxAttrs: [
      'allow-scripts',
      'allow-same-origin',
      'allow-forms',
      'allow-popups',
      'allow-popups-to-escape-sandbox'
    ]
  };
  
  // Initialize embed
  LookerEmbedSDK.init('your-instance.looker.com', {
    url: '/looker/auth',
    withCredentials: true
  });
  
  // Create dashboard
  LookerEmbedSDK.createDashboardWithId('sales_performance')
    .appendTo('#dashboard-container')
    .withClassName('embedded-dashboard')
    .withFilters(embedConfig.filters)
    .withParams(embedConfig.params)
    .withTheme('custom_theme')
    .on('dashboard:loaded', function(event) {
      console.log('Dashboard loaded', event);
    })
    .on('dashboard:run:start', function(event) {
      console.log('Dashboard running', event);
    })
    .on('dashboard:run:complete', function(event) {
      console.log('Dashboard complete', event);
    })
    .on('drillmenu:click', function(event) {
      console.log('Drill clicked', event);
    })
    .build()
    .connect()
    .then(dashboard => {
      console.log('Dashboard embedded successfully');
      
      // Update filters dynamically
      dashboard.updateFilters({
        'orders.created_date': '60 days'
      });
      
      // Run dashboard
      dashboard.run();
    })
    .catch(error => {
      console.error('Error embedding dashboard:', error);
    });
</script>
```

### SSO Embed Authentication

**Server-Side Embed URL Generation:**
```python
# Python example for generating signed embed URL
import base64
import binascii
import hashlib
import json
import time
from urllib.parse import quote, urlencode

def create_signed_embed_url(
    host,
    secret,
    external_user_id,
    permissions,
    models,
    session_length=3600,
    force_logout_login=False
):
    """Generate signed Looker embed URL."""
    
    # Embed user info
    embed_user = {
        'external_user_id': external_user_id,
        'first_name': 'John',
        'last_name': 'Doe',
        'session_length': session_length,
        'force_logout_login': force_logout_login,
        'permissions': permissions,
        'models': models,
        'user_attributes': {
            'user_region': 'West',
            'user_department': 'Sales',
            'can_view_pii': 'yes'
        },
        'access_filters': {
            'orders': {
                'region': 'West'
            }
        }
    }
    
    # Create nonce
    nonce = str(int(time.time()))
    
    # JSON encode
    json_embed_user = json.dumps(embed_user, separators=(',', ':'))
    
    # Create signature
    msg = f"{host}\n{nonce}\n{json_embed_user}"
    signature = base64.b64encode(
        hashlib.sha1(
            f"{msg}".encode('utf-8') + secret.encode('utf-8')
        ).digest()
    ).decode('utf-8')
    
    # Build URL parameters
    params = {
        'nonce': nonce,
        'time': nonce,
        'session_length': session_length,
        'external_user_id': external_user_id,
        'permissions': json.dumps(permissions),
        'models': json.dumps(models),
        'user_attributes': json.dumps(embed_user['user_attributes']),
        'access_filters': json.dumps(embed_user['access_filters']),
        'first_name': embed_user['first_name'],
        'last_name': embed_user['last_name'],
        'force_logout_login': str(force_logout_login).lower(),
        'signature': signature
    }
    
    # Build embed URL
    query_string = urlencode(params)
    embed_url = f"https://{host}/login/embed/{quote(query_string)}"
    
    return embed_url

# Usage
embed_url = create_signed_embed_url(
    host='your-instance.looker.com',
    secret='your-embed-secret',
    external_user_id='user123',
    permissions=[
        'access_data',
        'see_looks',
        'see_user_dashboards',
        'explore'
    ],
    models=['sales', 'marketing'],
    session_length=3600
)

print(f"Embed URL: {embed_url}")
```

---

## 6. API Integration

### Looker API

**Best Practices:**
- Use API for automation
- Implement proper error handling
- Cache API responses
- Monitor API usage

```python
import looker_sdk
from looker_sdk import models40 as models

# Initialize SDK
sdk = looker_sdk.init40()

# Example 1: Run query and get results
def run_inline_query(model, view, fields, filters=None, limit=500):
    """Run an inline query."""
    
    query_config = models.WriteQuery(
        model=model,
        view=view,
        fields=fields,
        filters=filters or {},
        limit=str(limit)
    )
    
    query = sdk.create_query(query_config)
    
    # Run query
    results = sdk.run_query(
        query_id=query.id,
        result_format='json'
    )
    
    return json.loads(results)

# Usage
results = run_inline_query(
    model='sales',
    view='orders',
    fields=[
        'orders.created_date',
        'orders.total_revenue',
        'orders.count'
    ],
    filters={
        'orders.created_date': '30 days'
    }
)

# Example 2: Run dashboard and get results
def run_dashboard(dashboard_id, filters=None):
    """Run all dashboard tiles."""
    
    dashboard = sdk.dashboard(dashboard_id)
    
    results = {}
    for element in dashboard.dashboard_elements:
        if element.query_id:
            # Run query for each tile
            tile_results = sdk.run_query(
                query_id=element.query_id,
                result_format='json',
                apply_formatting=True
            )
            results[element.title] = json.loads(tile_results)
    
    return results

# Example 3: Schedule dashboard delivery
def schedule_dashboard_delivery(
    dashboard_id,
    email_recipients,
    schedule_type='daily',
    time='09:00'
):
    """Schedule dashboard email delivery."""
    
    scheduled_plan = models.WriteScheduledPlan(
        name=f"Dashboard {dashboard_id} - {schedule_type}",
        dashboard_id=dashboard_id,
        scheduled_plan_destination=[
            models.ScheduledPlanDestination(
                type='email',
                address=email,
                format='pdf_landscape'
            ) for email in email_recipients
        ],
        crontab=f"0 {time.split(':')[0]} * * *",  # Daily at specified time
        enabled=True
    )
    
    created_plan = sdk.create_scheduled_plan(scheduled_plan)
    return created_plan

# Example 4: Create user and assign permissions
def create_looker_user(email, first_name, last_name, role_ids):
    """Create new Looker user."""
    
    user = models.WriteUser(
        email=email,
        first_name=first_name,
        last_name=last_name,
        is_disabled=False
    )
    
    created_user = sdk.create_user(user)
    
    # Assign roles
    sdk.set_user_roles(
        user_id=created_user.id,
        body=role_ids
    )
    
    # Set user attributes
    sdk.set_user_attribute_user_value(
        user_id=created_user.id,
        user_attribute_id='user_region',
        body=models.WriteUserAttributeWithValue(value='West')
    )
    
    return created_user

# Example 5: Export data
def export_look_to_csv(look_id, filename):
    """Export Look results to CSV."""
    
    look = sdk.look(look_id)
    
    # Run query and get CSV
    results = sdk.run_query(
        query_id=look.query_id,
        result_format='csv'
    )
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(results)
    
    print(f"Exported to {filename}")
```

---

## 7. Testing and Version Control

### Git Integration

**Best Practices:**
- Use feature branches
- Code review process
- Automated testing
- Deployment pipeline

```bash
# Git workflow for LookML development

# 1. Create feature branch
git checkout -b feature/new-sales-dashboard

# 2. Make changes to LookML files
# Edit views, models, dashboards

# 3. Test changes in development mode
# Use Looker UI to validate

# 4. Commit changes
git add .
git commit -m "Add new sales performance dashboard with regional filters"

# 5. Push to remote
git push origin feature/new-sales-dashboard

# 6. Create pull request
# Review changes in Looker UI
# Get code review from team

# 7. Merge to production
git checkout main
git merge feature/new-sales-dashboard
git push origin main

# 8. Deploy to production
# Looker automatically deploys from main branch
```

### LookML Validator

**Automated Testing:**
```python
# Python script for LookML validation
import subprocess
import json

def run_lookml_tests(project_name):
    """Run LookML validation tests."""
    
    # Content validator
    print("Running content validator...")
    result = subprocess.run(
        ['lookml-tools', 'validate', project_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Validation errors:\n{result.stdout}")
        return False
    
    # Run queries to test explores
    print("Testing explores...")
    explores_to_test = [
        'orders',
        'customers',
        'products'
    ]
    
    for explore in explores_to_test:
        test_result = test_explore(project_name, explore)
        if not test_result:
            print(f"Error testing explore: {explore}")
            return False
    
    print("All tests passed!")
    return True

def test_explore(model, explore):
    """Test that explore runs without errors."""
    
    # Use Looker SDK to run test query
    try:
        sdk = looker_sdk.init40()
        query = models.WriteQuery(
            model=model,
            view=explore,
            fields=[f'{explore}.count'],
            limit='1'
        )
        
        created_query = sdk.create_query(query)
        results = sdk.run_query(
            query_id=created_query.id,
            result_format='json'
        )
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
```

---

## 8. Common Anti-Patterns

### ❌ Anti-Pattern 1: No Datagroups
**Problem:** Cache never refreshes or always queries database
**Solution:** Implement proper datagroups based on ETL schedule

### ❌ Anti-Pattern 2: Hardcoded Values
**Problem:** Need to update multiple places
**Solution:** Use constants and parameters

### ❌ Anti-Pattern 3: Duplicate Logic
**Problem:** Same calculation in multiple views
**Solution:** Use extends or create reusable views

### ❌ Anti-Pattern 4: No Access Controls
**Problem:** All users see all data
**Solution:** Implement access filters and field-level security

### ❌ Anti-Pattern 5: Poor Model Organization
**Problem:** Hard to maintain and find code
**Solution:** Organize by business domain, use clear naming

---

## 9. Quick Reference Checklist

### LookML Development
- [ ] Organize models by business domain
- [ ] Use consistent naming conventions
- [ ] Document views and fields
- [ ] Implement proper datagroups
- [ ] Use Git version control

### Performance
- [ ] Create aggregate awareness tables
- [ ] Use persistent derived tables
- [ ] Implement effective caching
- [ ] Optimize join relationships
- [ ] Monitor query performance

### Security
- [ ] Implement access filters
- [ ] Use user attributes
- [ ] Apply field-level security
- [ ] Hide sensitive data
- [ ] Audit user access

### Dashboards
- [ ] Design with user experience in mind
- [ ] Use parameters for flexibility
- [ ] Implement drill-downs
- [ ] Optimize load times
- [ ] Test across devices

### Embedding
- [ ] Use SSO authentication
- [ ] Implement proper permissions
- [ ] Monitor embedded usage
- [ ] Optimize for performance
- [ ] Test embed functionality

---

*Best Practices for Google Cloud Data Engineer Certification*
