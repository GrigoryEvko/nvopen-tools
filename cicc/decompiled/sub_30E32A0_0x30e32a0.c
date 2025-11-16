// Function: sub_30E32A0
// Address: 0x30e32a0
//
__int64 __fastcall sub_30E32A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 i; // r12
  __int64 v8; // r14
  bool v9; // zf
  __int64 *v10; // rcx
  __int64 v11; // rsi
  char v12; // al
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 (__fastcall *v15)(__m128i *, __m128i *, __int64); // rax
  __int64 v16; // rdx
  __m128i v17; // xmm1
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 result; // rax
  __int64 v24; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+38h] [rbp-58h] BYREF
  __m128i v26; // [rsp+40h] [rbp-50h] BYREF
  __int64 (__fastcall *v27)(__m128i *, __m128i *, __int64); // [rsp+50h] [rbp-40h]
  __int64 v28; // [rsp+58h] [rbp-38h]

  v5 = a1;
  v24 = (a3 - 1) / 2;
  if ( a2 >= v24 )
  {
    v8 = a2;
    v14 = a3;
    if ( (a3 & 1) != 0 )
      goto LABEL_9;
    goto LABEL_12;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = *(_QWORD *)(a5 + 16) == 0;
    v10 = (__int64 *)(v5 + 16 * (i + 1));
    v11 = *v10;
    v26.m128i_i64[0] = *(v10 - 1);
    v25 = v11;
    if ( v9 )
      sub_4263D6(a1, v11, a3);
    a1 = a5;
    v12 = (*(__int64 (__fastcall **)(__int64, __int64 *, __m128i *))(a5 + 24))(a5, &v25, &v26);
    v13 = (_QWORD *)(v5 + 16 * (i + 1));
    if ( v12 )
    {
      --v8;
      v13 = (_QWORD *)(v5 + 8 * v8);
    }
    *(_QWORD *)(v5 + 8 * i) = *v13;
    if ( v8 >= v24 )
      break;
  }
  v14 = a3;
  if ( (a3 & 1) == 0 )
  {
LABEL_12:
    if ( (v14 - 2) / 2 == v8 )
    {
      *(_QWORD *)(v5 + 8 * v8) = *(_QWORD *)(v5 + 8 * (2 * v8 + 2) - 8);
      v8 = 2 * v8 + 1;
    }
  }
LABEL_9:
  v15 = *(__int64 (__fastcall **)(__m128i *, __m128i *, __int64))(a5 + 16);
  v16 = v28;
  v17 = _mm_loadu_si128(&v26);
  *(_QWORD *)(a5 + 16) = 0;
  v27 = v15;
  v18 = *(_QWORD *)(a5 + 24);
  *(_QWORD *)(a5 + 24) = v16;
  v19 = _mm_loadu_si128((const __m128i *)a5);
  *(__m128i *)a5 = v17;
  v28 = v18;
  v26 = v19;
  sub_30E31D0(v5, v8, a2, a4, (__int64)&v26);
  result = (__int64)v27;
  if ( v27 )
    return v27(&v26, &v26, 3);
  return result;
}
