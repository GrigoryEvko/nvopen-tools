// Function: sub_1205570
// Address: 0x1205570
//
__int64 __fastcall sub_1205570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v9; // r14
  __int64 v11; // r12
  __int64 v12; // r10
  __int64 i; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __m128i *v16; // rcx
  __int64 v17; // rsi
  __m128i *v18; // rdx
  __int64 v19; // rcx
  const __m128i *v20; // rsi
  __int64 result; // rax
  __m128i v22; // xmm2
  const __m128i *v23; // rcx
  __m128i v24; // [rsp+0h] [rbp-40h] BYREF
  __int64 v25; // [rsp+10h] [rbp-30h]

  v9 = a3 & 1;
  v11 = (a3 - 1) / 2;
  v12 = a7;
  if ( a2 >= v11 )
  {
    v18 = (__m128i *)(a1 + 24 * a2);
    if ( v9 )
    {
      v24 = _mm_loadu_si128((const __m128i *)&a7);
      v25 = a8;
      goto LABEL_13;
    }
    v15 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v15 )
  {
    v14 = i + 1;
    v15 = 2 * v14;
    v16 = (__m128i *)(a1 + 24 * i);
    v17 = 2 * v14 - 1;
    v18 = (__m128i *)(a1 + 48 * v14);
    if ( (v18->m128i_i32[0] & 6u) < (*(_DWORD *)(a1 + 24 * v17) & 6u) )
    {
      v18 = (__m128i *)(a1 + 24 * v17);
      v15 = v17;
    }
    *v16 = _mm_loadu_si128(v18);
    v16[1].m128i_i64[0] = v18[1].m128i_i64[0];
    if ( v15 >= v11 )
      break;
  }
  if ( !v9 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v15 )
    {
      v15 = 2 * v15 + 1;
      v23 = (const __m128i *)(a1 + 24 * v15);
      *v18 = _mm_loadu_si128(v23);
      v18[1].m128i_i64[0] = v23[1].m128i_i64[0];
      v18 = (__m128i *)v23;
    }
  }
  v25 = a8;
  v24 = _mm_loadu_si128((const __m128i *)&a7);
  v19 = (v15 - 1) / 2;
  if ( v15 > a2 )
  {
    while ( 1 )
    {
      v20 = (const __m128i *)(a1 + 24 * v19);
      v18 = (__m128i *)(a1 + 24 * v15);
      if ( (v20->m128i_i32[0] & 6u) >= ((unsigned __int8)v12 & 6u) )
        break;
      *v18 = _mm_loadu_si128(v20);
      v18[1].m128i_i64[0] = v20[1].m128i_i64[0];
      v15 = v19;
      if ( a2 >= v19 )
      {
        v18 = (__m128i *)(a1 + 24 * v19);
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
LABEL_13:
  result = v25;
  v24.m128i_i64[0] = v12;
  v22 = _mm_loadu_si128(&v24);
  v18[1].m128i_i64[0] = v25;
  *v18 = v22;
  return result;
}
