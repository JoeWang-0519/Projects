# 1.
select
	listing_id,
    name,
    description,
    price
from listing
where accommodates = 3
limit 5;

# 2.
select
	listing_id,
    name,
    description
from listing
where (price between 500 and 700) and (room_type = 'private room')
limit 5;

# 3. 
with superhost_list as
	(select host_id from host where host_is_superhost = 1 limit 5)
select
	listing_id,
    name,
    description
from listing
where host_id in (select host_id from superhost_list)
;

# 4.
select
	listing_id,
    date,
    available,
    price,
    minimum_nights,
    maximum_nights
from calendar
where listing_id = ''
limit 5;

# 5.
select
	id,
	listing_id,
	date,
    reviewer_id,
    comments
from comment
where listing_id = ''
limit 5;

# 6.
select
	id,
    listing_id,
    date,
    reviewer_id,
    comments
from comment
where revewer_id in (select reviewer_id from reviewer where name = '')
limit 5;

# 7.
select 
	host_id,
    host_name,
    host_location,
    host_is_superhost
from host
where host_name = ''
;

# 8.
select 
    start_date,
    end_date,
    total_cost
from transaction
where host_id in (select host_id from host where host_name = '');

#9.
select
	date,
    reviewer_id,
    comments
from comment
where listing_id in (select listing_id from listing where host_id in
							(select host_id from host where host_name = '')
					)
order by date
limit 5;

# 10.
select
	start_date,
    end_date,
    total_cost
from transaction
where listing_id in (select listing_id from listing where host_id in
							(select host_id from host where host_name = '')
	)
limit 5;

