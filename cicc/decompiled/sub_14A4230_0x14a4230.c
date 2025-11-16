// Function: sub_14A4230
// Address: 0x14a4230
//
__m128i *__fastcall sub_14A4230(__m128i *a1)
{
  void (__fastcall *v1)(__m128i *, __m128i *, __int64); // rax
  __int64 v2; // rdx
  __m128i v3; // xmm1
  __m128i v4; // xmm0
  __int64 v5; // rax
  __m128i *v6; // rax
  __m128i *v7; // r12
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v10)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v11; // [rsp+18h] [rbp-18h]

  v1 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a1[1].m128i_i64[0];
  v2 = v11;
  a1[1].m128i_i64[0] = 0;
  v3 = _mm_loadu_si128(&v9);
  v4 = _mm_loadu_si128(a1);
  v10 = v1;
  v5 = a1[1].m128i_i64[1];
  a1[1].m128i_i64[1] = v2;
  *a1 = v3;
  v11 = v5;
  v9 = v4;
  v6 = (__m128i *)sub_22077B0(208);
  v7 = v6;
  if ( v6 )
    sub_14A3F50(v6, &v9);
  if ( v10 )
    v10(&v9, &v9, 3);
  return v7;
}
