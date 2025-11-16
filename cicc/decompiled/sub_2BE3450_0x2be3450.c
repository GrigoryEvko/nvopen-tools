// Function: sub_2BE3450
// Address: 0x2be3450
//
unsigned __int64 __fastcall sub_2BE3450(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__m128i *)a1[6];
  if ( v2 == (__m128i *)(a1[8] - 24) )
    return sub_2BE3350(a1, a2);
  if ( v2 )
  {
    *v2 = _mm_loadu_si128(a2);
    v2[1].m128i_i64[0] = a2[1].m128i_i64[0];
    v2 = (__m128i *)a1[6];
  }
  result = (unsigned __int64)&v2[1].m128i_u64[1];
  a1[6] = result;
  return result;
}
