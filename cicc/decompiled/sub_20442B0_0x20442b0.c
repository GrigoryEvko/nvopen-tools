// Function: sub_20442B0
// Address: 0x20442b0
//
__int64 __fastcall sub_20442B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        __int128 a8,
        __int64 a9)
{
  __int64 i; // r15
  __int64 v10; // r12
  __m128i *v11; // r14
  int v12; // eax
  bool v13; // sf
  __m128i *v14; // rax
  __int64 v15; // r13
  const __m128i *v16; // r15
  __m128i v17; // xmm5
  __int64 result; // rax
  __m128i v19; // xmm4
  const __m128i *v20; // rax
  __int64 v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  __m128i v25; // [rsp+30h] [rbp-60h] BYREF
  __m128i v26; // [rsp+40h] [rbp-50h] BYREF
  __int64 v27; // [rsp+50h] [rbp-40h]

  v24 = (a3 - 1) / 2;
  v23 = a7.m128i_i64[1];
  v22 = a3 & 1;
  if ( a2 >= v24 )
  {
    v11 = (__m128i *)(a1 + 40 * a2);
    if ( (a3 & 1) != 0 )
    {
      v25 = _mm_loadu_si128(&a7);
      v27 = a9;
      v26 = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_15;
    }
    v10 = a2;
    goto LABEL_18;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v11 = (__m128i *)(a1 + 80 * (i + 1));
    v12 = sub_16AEA10(v11->m128i_i64[1] + 24, *(_QWORD *)(a1 + 40 * (v10 - 1) + 8) + 24LL);
    v13 = v12 < 0;
    if ( v12 < 0 )
      v11 = (__m128i *)(a1 + 40 * (v10 - 1));
    v14 = (__m128i *)(a1 + 40 * i);
    if ( v13 )
      --v10;
    *v14 = _mm_loadu_si128(v11);
    v14[1] = _mm_loadu_si128(v11 + 1);
    v14[2].m128i_i32[0] = v11[2].m128i_i32[0];
    if ( v10 >= v24 )
      break;
  }
  if ( !v22 )
  {
LABEL_18:
    if ( (a3 - 2) / 2 == v10 )
    {
      v10 = 2 * v10 + 1;
      v20 = (const __m128i *)(a1 + 40 * v10);
      *v11 = _mm_loadu_si128(v20);
      v11[1] = _mm_loadu_si128(v20 + 1);
      v11[2].m128i_i32[0] = v20[2].m128i_i32[0];
      v11 = (__m128i *)v20;
    }
  }
  v27 = a9;
  v25 = _mm_loadu_si128(&a7);
  v26 = _mm_loadu_si128((const __m128i *)&a8);
  v15 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v16 = (const __m128i *)(a1 + 40 * v15);
      v11 = (__m128i *)(a1 + 40 * v10);
      if ( (int)sub_16AEA10(v16->m128i_i64[1] + 24, v23 + 24) >= 0 )
        break;
      v10 = v15;
      *v11 = _mm_loadu_si128(v16);
      v11[1] = _mm_loadu_si128(v16 + 1);
      v11[2].m128i_i32[0] = v16[2].m128i_i32[0];
      if ( a2 >= v15 )
      {
        v11 = (__m128i *)(a1 + 40 * v15);
        break;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_15:
  v17 = _mm_loadu_si128(&v26);
  v25.m128i_i64[1] = v23;
  result = (unsigned int)v27;
  v19 = _mm_loadu_si128(&v25);
  v11[1] = v17;
  v11[2].m128i_i32[0] = result;
  *v11 = v19;
  return result;
}
