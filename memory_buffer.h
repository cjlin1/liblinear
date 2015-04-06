#ifndef _MEMORY_BUFFER
#define _MEMORY_BUFFER

#include <assert.h>

// Simple memory buffer for RAII
template<typename TItemType>
class CBuffer {
public:
	CBuffer() : count( 0 ), ptr( 0 ) {}
	explicit CBuffer( size_t count );
	~CBuffer();

	size_t Count() const { return count; }

	TItemType* Detach();
	void Realloc( size_t new_size );

	TItemType& operator[]( int index ) { assert( ptr != 0 ); return ptr[index]; }
	TItemType* operator->() { assert( count == 1 ); return ptr; }
	operator TItemType*() { return ptr; } // bad idea, but less code changed

	// Better way, may be in future
	TItemType* Ptr() { assert( ptr != 0 ); return ptr; }

private:
	size_t count;
	TItemType* ptr;

	// prohibited
	CBuffer( const CBuffer<TItemType>& other );
	const CBuffer<TItemType>& operator=( const CBuffer<TItemType>& other );
};

// ----------------------------------------------------------------------------

template<typename TItemType>
inline CBuffer<TItemType>::CBuffer( size_t _count ) :
count( _count ),
	ptr( 0 )
{
	assert( count > 0 );
	ptr = new TItemType[count];
}

template<typename TItemType>
inline CBuffer<TItemType>::~CBuffer()
{
	if( ptr != 0 ) {
		delete[] ptr;
		ptr = 0;
	}
}

template<typename TItemType>
inline TItemType* CBuffer<TItemType>::Detach()
{
	TItemType* tmp = ptr;
	ptr = 0;
	count = 0;
	return ptr;
}

template<typename TItemType>
inline void CBuffer<TItemType>::Realloc( size_t new_count )
{
	assert( new_count > count );
	TItemType* buffer = new TItemType[new_count];
	if( count > 0 ) {
		::memcpy( buffer, ptr, count * sizeof( TItemType ) );
		delete[] Detach();
	}

	ptr = buffer;
	count = new_count;
}

#endif // _MEMORY_BUFFER
