// Function: sub_24FD590
// Address: 0x24fd590
//
__int64 *__fastcall sub_24FD590(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int64 v8; // r11
  __int64 v10; // r12
  __int64 v11; // r9
  __int64 i; // rdx
  __int64 v13; // rcx
  __int64 *result; // rax
  __int64 v15; // r10
  __int64 v16; // rdx
  __int64 v17; // r10
  __int64 *v18; // rdx
  __int64 v19; // rdx
  __m128i v20; // xmm2
  __int64 v21; // rdx
  __int64 v22; // rdx
  __m128i v23; // [rsp+0h] [rbp-30h] BYREF
  __m128i v24; // [rsp+10h] [rbp-20h]

  v8 = (a3 - 1) / 2;
  v10 = a3 & 1;
  v11 = a7;
  if ( a2 >= v8 )
  {
    result = (__int64 *)(a1 + 32 * a2);
    if ( (a3 & 1) != 0 )
    {
      v23 = _mm_loadu_si128((const __m128i *)&a7);
      v24 = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_13;
    }
    v13 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v13 )
  {
    v13 = 2 * (i + 1);
    result = (__int64 *)(a1 + ((i + 1) << 6));
    v15 = *result;
    if ( *result < *(result - 4) )
    {
      --v13;
      result = (__int64 *)(a1 + 32 * v13);
      v15 = *result;
    }
    v16 = a1 + 32 * i;
    *(_QWORD *)v16 = v15;
    *(__m128i *)(v16 + 8) = _mm_loadu_si128((const __m128i *)(result + 1));
    *(_QWORD *)(v16 + 24) = result[3];
    if ( v13 >= v8 )
      break;
  }
  if ( !v10 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v13 )
    {
      v21 = v13 + 1;
      v13 = 2 * (v13 + 1) - 1;
      v22 = a1 + (v21 << 6) - 32;
      *result = *(_QWORD *)v22;
      *(__m128i *)(result + 1) = _mm_loadu_si128((const __m128i *)(v22 + 8));
      result[3] = *(_QWORD *)(v22 + 24);
      result = (__int64 *)(a1 + 32 * v13);
    }
  }
  v23 = _mm_loadu_si128((const __m128i *)&a7);
  v24 = _mm_loadu_si128((const __m128i *)&a8);
  v17 = (v13 - 1) / 2;
  if ( v13 > a2 )
  {
    while ( 1 )
    {
      result = (__int64 *)(a1 + 32 * v13);
      v18 = (__int64 *)(a1 + 32 * v17);
      if ( *v18 >= v11 )
        break;
      *result = *v18;
      *(__m128i *)(result + 1) = _mm_loadu_si128((const __m128i *)(v18 + 1));
      result[3] = v18[3];
      v13 = v17;
      if ( a2 >= v17 )
      {
        result = (__int64 *)(a1 + 32 * v17);
        break;
      }
      v17 = (v17 - 1) / 2;
    }
  }
LABEL_13:
  v19 = v24.m128i_i64[1];
  v20 = _mm_loadu_si128((const __m128i *)&v23.m128i_u64[1]);
  *result = v11;
  result[3] = v19;
  *(__m128i *)(result + 1) = v20;
  return result;
}
