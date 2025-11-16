// Function: sub_C22490
// Address: 0xc22490
//
const __m128i *__fastcall sub_C22490(const __m128i **a1, unsigned __int64 a2)
{
  const __m128i *v3; // rdi
  const __m128i *result; // rax
  const __m128i *v5; // rcx
  __m128i *v6; // r13
  signed __int64 v7; // r12
  __int64 v8; // rax
  __m128i *v9; // rdx

  if ( a2 > 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v3 = *a1;
  result = v3;
  if ( a2 > a1[2] - v3 )
  {
    v5 = a1[1];
    v6 = 0;
    v7 = (char *)v5 - (char *)v3;
    if ( a2 )
    {
      v8 = sub_22077B0(16 * a2);
      v3 = *a1;
      v5 = a1[1];
      v6 = (__m128i *)v8;
      result = *a1;
    }
    if ( v5 != v3 )
    {
      v9 = v6;
      do
      {
        if ( v9 )
          *v9 = _mm_loadu_si128(result);
        ++result;
        ++v9;
      }
      while ( result != v5 );
    }
    if ( v3 )
      result = (const __m128i *)j_j___libc_free_0(v3, (char *)a1[2] - (char *)v3);
    *a1 = v6;
    a1[1] = (__m128i *)((char *)v6 + v7);
    a1[2] = &v6[a2];
  }
  return result;
}
