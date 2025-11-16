// Function: sub_3712350
// Address: 0x3712350
//
__int64 __fastcall sub_3712350(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *, unsigned __int64 *),
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v9; // r14
  __int64 i; // r15
  __int64 v11; // r13
  __m128i *v12; // r12
  __m128i *v13; // rax
  __int64 v14; // r14
  const __m128i *v15; // r15
  __m128i v16; // xmm5
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v21; // [rsp+8h] [rbp-78h]
  __m128i v23; // [rsp+20h] [rbp-60h] BYREF
  __m128i v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+40h] [rbp-40h]

  v9 = (a3 - 1) / 2;
  v21 = a3 & 1;
  if ( a2 >= v9 )
  {
    v12 = (__m128i *)(a1 + 40 * a2);
    if ( (a3 & 1) != 0 )
    {
      v23 = _mm_loadu_si128((const __m128i *)&a7);
      v25 = a9;
      v24 = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_13;
    }
    v11 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    v12 = (__m128i *)(a1 + 80 * (i + 1));
    if ( a4(v12, &v12[-3].m128i_u64[1]) )
    {
      --v11;
      v12 = (__m128i *)(a1 + 40 * v11);
    }
    v13 = (__m128i *)(a1 + 40 * i);
    *v13 = _mm_loadu_si128(v12);
    v13[1] = _mm_loadu_si128(v12 + 1);
    v13[2].m128i_i16[0] = v12[2].m128i_i16[0];
    if ( v11 >= v9 )
      break;
  }
  if ( !v21 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v11 )
    {
      v18 = 10 * (v11 + 1);
      v11 = 2 * (v11 + 1) - 1;
      v19 = a1 + 8 * v18;
      *v12 = _mm_loadu_si128((const __m128i *)(v19 - 40));
      v12[1] = _mm_loadu_si128((const __m128i *)(v19 - 24));
      v12[2].m128i_i16[0] = *(_WORD *)(v19 - 8);
      v12 = (__m128i *)(a1 + 40 * v11);
    }
  }
  v25 = a9;
  v23 = _mm_loadu_si128((const __m128i *)&a7);
  v24 = _mm_loadu_si128((const __m128i *)&a8);
  v14 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      v15 = (const __m128i *)(a1 + 40 * v14);
      v12 = (__m128i *)(a1 + 40 * v11);
      if ( !a4((__m128i *)v15, (unsigned __int64 *)&v23) )
        break;
      v11 = v14;
      *v12 = _mm_loadu_si128(v15);
      v12[1] = _mm_loadu_si128(v15 + 1);
      v12[2].m128i_i16[0] = v15[2].m128i_i16[0];
      if ( a2 >= v14 )
      {
        v12 = (__m128i *)(a1 + 40 * v14);
        break;
      }
      v14 = (v14 - 1) / 2;
    }
  }
LABEL_13:
  v16 = _mm_loadu_si128(&v24);
  result = (unsigned __int16)v25;
  *v12 = _mm_loadu_si128(&v23);
  v12[2].m128i_i16[0] = result;
  v12[1] = v16;
  return result;
}
