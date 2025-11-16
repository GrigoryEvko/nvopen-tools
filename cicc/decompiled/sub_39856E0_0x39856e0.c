// Function: sub_39856E0
// Address: 0x39856e0
//
__m128i *__fastcall sub_39856E0(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __int64 v2; // r15
  __m128i v3; // xmm1
  unsigned __int64 v4; // r12
  __m128i v6; // xmm5
  __m128i *v7; // [rsp+8h] [rbp-98h]
  char v8[8]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v9; // [rsp+18h] [rbp-88h]
  char v10[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v11; // [rsp+38h] [rbp-68h]
  __m128i v12; // [rsp+50h] [rbp-50h] BYREF
  __m128i v13[4]; // [rsp+60h] [rbp-40h] BYREF

  v1 = a1;
  v2 = a1->m128i_i64[0];
  v12 = _mm_loadu_si128(a1);
  v13[0] = _mm_loadu_si128(a1 + 1);
  while ( 1 )
  {
    v7 = (__m128i *)v1;
    v1 -= 2;
    sub_15B1350((__int64)v8, *(unsigned __int64 **)(v2 + 24), *(unsigned __int64 **)(v2 + 32));
    v4 = v9;
    sub_15B1350(
      (__int64)v10,
      *(unsigned __int64 **)(v1->m128i_i64[0] + 24),
      *(unsigned __int64 **)(v1->m128i_i64[0] + 32));
    if ( v4 >= v11 )
      break;
    v3 = _mm_loadu_si128(v1 + 1);
    v1[2] = _mm_loadu_si128(v1);
    v1[3] = v3;
  }
  v6 = _mm_loadu_si128(v13);
  *v7 = _mm_loadu_si128(&v12);
  v7[1] = v6;
  return v7;
}
