// Function: sub_3985DD0
// Address: 0x3985dd0
//
__int64 __fastcall sub_3985DD0(__m128i *src, const __m128i *a2)
{
  const __m128i *v3; // r15
  __m128i v4; // xmm1
  __int64 result; // rax
  __m128i v6; // xmm3
  unsigned __int64 v7; // r12
  const __m128i *v8; // rdi
  _BYTE v9[8]; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v10; // [rsp+18h] [rbp-68h]
  __m128i v11; // [rsp+30h] [rbp-50h] BYREF
  __m128i v12[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( src != a2 && a2 != &src[2] )
  {
    v3 = src + 2;
    do
    {
      while ( 1 )
      {
        sub_15B1350(
          (__int64)v9,
          *(unsigned __int64 **)(v3->m128i_i64[0] + 24),
          *(unsigned __int64 **)(v3->m128i_i64[0] + 32));
        v7 = v10;
        result = sub_15B1350(
                   (__int64)&v11,
                   *(unsigned __int64 **)(src->m128i_i64[0] + 24),
                   *(unsigned __int64 **)(src->m128i_i64[0] + 32));
        if ( v7 < v11.m128i_i64[1] )
          break;
        v8 = v3;
        v3 += 2;
        result = (__int64)sub_39856E0(v8);
        if ( a2 == v3 )
          return result;
      }
      v4 = _mm_loadu_si128(v3 + 1);
      v11 = _mm_loadu_si128(v3);
      v12[0] = v4;
      if ( src != v3 )
        result = (__int64)memmove(&src[2], src, (char *)v3 - (char *)src);
      v6 = _mm_loadu_si128(v12);
      v3 += 2;
      *src = _mm_loadu_si128(&v11);
      src[1] = v6;
    }
    while ( a2 != v3 );
  }
  return result;
}
