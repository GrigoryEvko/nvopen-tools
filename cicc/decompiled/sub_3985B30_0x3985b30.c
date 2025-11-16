// Function: sub_3985B30
// Address: 0x3985b30
//
__int64 __fastcall sub_3985B30(__int64 a1, __int64 a2, __int64 a3, unsigned __int32 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 i; // r12
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r9
  __m128i *v11; // r14
  unsigned __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 j; // r12
  __m128i *v17; // r13
  const __m128i *v18; // rbx
  __int64 v20; // rcx
  __int64 v23; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+28h] [rbp-98h]
  __int64 v26; // [rsp+38h] [rbp-88h]
  unsigned __int64 v27; // [rsp+38h] [rbp-88h]
  char v28[8]; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v29; // [rsp+58h] [rbp-68h]
  char v30[8]; // [rsp+70h] [rbp-50h] BYREF
  unsigned __int64 v31; // [rsp+78h] [rbp-48h]

  v5 = a1;
  v6 = a5;
  v25 = (a3 - 1) / 2;
  v23 = a3 & 1;
  if ( a2 >= v25 )
  {
    v11 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_14;
    v13 = a2;
    goto LABEL_17;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = 32 * (i + 1);
    v10 = a1 + v9 - 16;
    v11 = (__m128i *)(a1 + v9);
    v26 = v10;
    sub_15B1350(
      (__int64)v30,
      *(unsigned __int64 **)(v11->m128i_i64[1] + 24),
      *(unsigned __int64 **)(v11->m128i_i64[1] + 32));
    v12 = v31;
    sub_15B1350(
      (__int64)v28,
      *(unsigned __int64 **)(*(_QWORD *)(v26 + 8) + 24LL),
      *(unsigned __int64 **)(*(_QWORD *)(v26 + 8) + 32LL));
    if ( v12 < v29 )
    {
      --v8;
      v11 = (__m128i *)(a1 + 16 * v8);
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v11);
    if ( v8 >= v25 )
      break;
  }
  v5 = a1;
  v13 = v8;
  v6 = a5;
  if ( !v23 )
  {
LABEL_17:
    if ( (a3 - 2) / 2 == v13 )
    {
      v20 = v13 + 1;
      v13 = 2 * (v13 + 1) - 1;
      *v11 = _mm_loadu_si128((const __m128i *)(v5 + 32 * v20 - 16));
      v11 = (__m128i *)(v5 + 16 * v13);
    }
  }
  if ( v13 > a2 )
  {
    v14 = v5;
    v15 = v6;
    for ( j = (v13 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v17 = (__m128i *)(v14 + 16 * v13);
      v18 = (const __m128i *)(v14 + 16 * j);
      sub_15B1350(
        (__int64)v30,
        *(unsigned __int64 **)(v18->m128i_i64[1] + 24),
        *(unsigned __int64 **)(v18->m128i_i64[1] + 32));
      v27 = v31;
      sub_15B1350((__int64)v28, *(unsigned __int64 **)(v15 + 24), *(unsigned __int64 **)(v15 + 32));
      if ( v27 >= v29 )
      {
        v11 = v17;
        v6 = v15;
        goto LABEL_14;
      }
      *v17 = _mm_loadu_si128(v18);
      v13 = j;
      if ( a2 >= j )
        break;
    }
    v11 = (__m128i *)(v14 + 16 * j);
    v6 = v15;
  }
LABEL_14:
  v11->m128i_i64[1] = v6;
  v11->m128i_i32[0] = a4;
  return a4;
}
