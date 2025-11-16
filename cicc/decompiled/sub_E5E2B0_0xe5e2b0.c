// Function: sub_E5E2B0
// Address: 0xe5e2b0
//
__m128i *__fastcall sub_E5E2B0(__m128i *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int8 v7; // bl
  __int64 v8; // rax
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  char v12; // [rsp+1Fh] [rbp-59h] BYREF
  __int64 v13; // [rsp+20h] [rbp-58h] BYREF
  __m128i v14; // [rsp+28h] [rbp-50h] BYREF
  __m128i v15; // [rsp+38h] [rbp-40h] BYREF

  v14 = 0u;
  v15.m128i_i64[0] = 0;
  v15.m128i_i32[2] = 0;
  v7 = sub_E5C4E0(a2, a4, a3, &v14, a5, &v13, &v12);
  if ( !v7 )
    (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64 *, __int64 *, __int64 *, __int64, __int64, __int64, __int64))(**(_QWORD **)(a2 + 24) + 32LL))(
      *(_QWORD *)(a2 + 24),
      a2,
      a3,
      a4,
      &v13,
      &v13,
      v14.m128i_i64[0],
      v14.m128i_i64[1],
      v15.m128i_i64[0],
      v15.m128i_i64[1]);
  v8 = v13;
  v9 = _mm_loadu_si128(&v14);
  a1->m128i_i8[0] = v7;
  v10 = _mm_loadu_si128(&v15);
  a1->m128i_i64[1] = v8;
  a1[1] = v9;
  a1[2] = v10;
  return a1;
}
