// Function: sub_1939300
// Address: 0x1939300
//
__int64 __fastcall sub_1939300(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r12
  __int64 i; // r13
  __int64 v12; // rbx
  __m128i *v13; // r15
  __m128i *v14; // rdx
  __int64 v15; // r12
  const __m128i *v16; // r13
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v21; // [rsp+30h] [rbp-40h]

  v10 = (a3 - 1) / 2;
  v21 = a3 & 1;
  if ( a2 >= v10 )
  {
    v13 = (__m128i *)(a1 + 32 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v12 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v12 )
  {
    v12 = 2 * (i + 1);
    v13 = (__m128i *)(a1 + ((i + 1) << 6));
    if ( (int)sub_16AEA10(v13->m128i_i64[1] + 24, v13[-2].m128i_i64[1] + 24) < 0 )
    {
      --v12;
      v13 = (__m128i *)(a1 + 32 * v12);
    }
    v14 = (__m128i *)(a1 + 32 * i);
    *v14 = _mm_loadu_si128(v13);
    v14[1] = _mm_loadu_si128(v13 + 1);
    if ( v12 >= v10 )
      break;
  }
  if ( !v21 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v12 )
    {
      v18 = v12 + 1;
      v12 = 2 * (v12 + 1) - 1;
      v19 = a1 + (v18 << 6);
      *v13 = _mm_loadu_si128((const __m128i *)(v19 - 32));
      v13[1] = _mm_loadu_si128((const __m128i *)(v19 - 16));
      v13 = (__m128i *)(a1 + 32 * v12);
    }
  }
  v15 = (v12 - 1) / 2;
  if ( v12 > a2 )
  {
    while ( 1 )
    {
      v13 = (__m128i *)(a1 + 32 * v12);
      v16 = (const __m128i *)(a1 + 32 * v15);
      if ( (int)sub_16AEA10(v16->m128i_i64[1] + 24, a8 + 24) >= 0 )
        break;
      v12 = v15;
      *v13 = _mm_loadu_si128(v16);
      v13[1] = _mm_loadu_si128(v16 + 1);
      if ( a2 >= v15 )
      {
        v13 = (__m128i *)(a1 + 32 * v15);
        break;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_13:
  v13->m128i_i64[0] = a7;
  v13->m128i_i64[1] = a8;
  v13[1].m128i_i64[0] = a9;
  v13[1].m128i_i64[1] = a10;
  return a10;
}
