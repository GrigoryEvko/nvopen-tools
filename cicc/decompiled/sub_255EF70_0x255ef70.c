// Function: sub_255EF70
// Address: 0x255ef70
//
__int64 __fastcall sub_255EF70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        const __m128i *a9,
        const __m128i *a10,
        int a11,
        int a12,
        const __m128i *a13)
{
  const __m128i *v13; // rbx
  const __m128i *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 result; // rax

  v13 = a9;
  if ( a9 == a13 )
  {
    v16 = 0;
  }
  else
  {
    v14 = a9;
    v15 = 0;
    do
    {
      do
        ++v14;
      while ( v14 != a10 && (v14->m128i_i64[0] == -1 || v14->m128i_i64[0] == -2) );
      ++v15;
    }
    while ( v14 != a13 );
    v16 = v15;
  }
  v17 = *(unsigned int *)(a1 + 8);
  if ( v17 + v16 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v17 + v16, 0x10u, a5, a6);
    v17 = *(unsigned int *)(a1 + 8);
  }
  v18 = (__m128i *)(*(_QWORD *)a1 + 16 * v17);
  if ( a9 != a13 )
  {
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v13);
      do
        ++v13;
      while ( v13 != a10 && (v13->m128i_i64[0] == -1 || v13->m128i_i64[0] == -2) );
      ++v18;
    }
    while ( v13 != a13 );
    v17 = *(unsigned int *)(a1 + 8);
  }
  result = v16 + v17;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
