// Function: sub_CF6D20
// Address: 0xcf6d20
//
__m128i *__fastcall sub_CF6D20(__m128i *a1, __int8 a2)
{
  void (__fastcall *v2)(__m128i *, __m128i *, __int64); // rax
  __int64 v3; // rdx
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i *v7; // rax
  __m128i *v8; // r12
  __m128i v10; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v11)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v12; // [rsp+18h] [rbp-18h]

  v2 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a1[1].m128i_i64[0];
  v3 = v12;
  a1[1].m128i_i64[0] = 0;
  v4 = _mm_loadu_si128(&v10);
  v5 = _mm_loadu_si128(a1);
  v11 = v2;
  v6 = a1[1].m128i_i64[1];
  a1[1].m128i_i64[1] = v3;
  *a1 = v4;
  v12 = v6;
  v10 = v5;
  v7 = (__m128i *)sub_22077B0(216);
  v8 = v7;
  if ( v7 )
    sub_CF6B40(v7, &v10, a2);
  if ( v11 )
    v11(&v10, &v10, 3);
  return v8;
}
