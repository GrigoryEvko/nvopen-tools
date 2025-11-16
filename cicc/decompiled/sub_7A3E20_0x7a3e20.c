// Function: sub_7A3E20
// Address: 0x7a3e20
//
__int64 __fastcall sub_7A3E20(const __m128i **a1)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rdi
  const __m128i *v5; // r15
  __m128i *v6; // rax
  __m128i *v7; // r13
  const __m128i *v8; // rdx
  __m128i *v9; // rsi
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = (__int64)a1[1];
  if ( v2 <= 1 )
  {
    v4 = 48;
    v3 = 2;
  }
  else
  {
    v3 = v2 + (v2 >> 1) + 1;
    v4 = 24 * v3;
  }
  v5 = *a1;
  v11 = (__int64)a1[2];
  v6 = (__m128i *)sub_823970(v4);
  v7 = v6;
  if ( v11 > 0 )
  {
    v8 = v5;
    v9 = (__m128i *)((char *)v6 + 24 * v11);
    do
    {
      if ( v6 )
      {
        *v6 = _mm_loadu_si128(v8);
        v6[1].m128i_i64[0] = v8[1].m128i_i64[0];
      }
      v6 = (__m128i *)((char *)v6 + 24);
      v8 = (const __m128i *)((char *)v8 + 24);
    }
    while ( v9 != v6 );
  }
  result = sub_823A00(v5, 24 * v2);
  *a1 = v7;
  a1[1] = (const __m128i *)v3;
  return result;
}
