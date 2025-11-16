// Function: sub_24FD750
// Address: 0x24fd750
//
__int64 __fastcall sub_24FD750(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdx
  __int64 *v3; // rax
  __m128i v4; // xmm0
  __int64 v5; // rdx
  __int64 result; // rax
  __m128i v7; // xmm3
  __m128i v8; // [rsp+0h] [rbp-20h] BYREF
  __m128i v9; // [rsp+10h] [rbp-10h]

  v1 = *a1;
  v2 = *(a1 - 4);
  v8 = _mm_loadu_si128((const __m128i *)a1);
  v9 = _mm_loadu_si128((const __m128i *)a1 + 1);
  if ( v1 < v2 )
  {
    v3 = a1 - 4;
    do
    {
      v3[4] = v2;
      v4 = _mm_loadu_si128((const __m128i *)(v3 + 1));
      a1 = v3;
      v3 -= 4;
      v5 = v3[7];
      *(__m128i *)(v3 + 9) = v4;
      v3[11] = v5;
      v2 = *v3;
    }
    while ( v1 < *v3 );
  }
  result = v9.m128i_i64[1];
  v7 = _mm_loadu_si128((const __m128i *)&v8.m128i_u64[1]);
  *a1 = v1;
  a1[3] = result;
  *(__m128i *)(a1 + 1) = v7;
  return result;
}
