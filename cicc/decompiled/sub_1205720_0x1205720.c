// Function: sub_1205720
// Address: 0x1205720
//
__int64 __fastcall sub_1205720(__m128i *a1)
{
  unsigned int v1; // ecx
  unsigned int v2; // edx
  __m128i *v3; // rax
  __m128i v4; // xmm0
  __int64 v5; // rdx
  __int64 result; // rax
  __m128i v7; // xmm2
  __m128i v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9; // [rsp+10h] [rbp-10h]

  v1 = a1->m128i_i32[0] & 6;
  v2 = a1[-2].m128i_i32[2] & 6;
  v9 = a1[1].m128i_i64[0];
  v8 = _mm_loadu_si128(a1);
  if ( v1 < v2 )
  {
    v3 = (__m128i *)((char *)a1 - 24);
    do
    {
      v4 = _mm_loadu_si128(v3);
      v5 = v3[1].m128i_i64[0];
      a1 = v3;
      v3 = (__m128i *)((char *)v3 - 24);
      v3[3] = v4;
      v3[4].m128i_i64[0] = v5;
    }
    while ( (v3->m128i_i32[0] & 6u) > v1 );
  }
  result = v9;
  v7 = _mm_loadu_si128(&v8);
  a1[1].m128i_i64[0] = v9;
  *a1 = v7;
  return result;
}
