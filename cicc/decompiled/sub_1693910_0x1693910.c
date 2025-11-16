// Function: sub_1693910
// Address: 0x1693910
//
__int64 __fastcall sub_1693910(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v9; // r13
  __int64 v11; // rbx
  unsigned __int64 v12; // r10
  __int64 i; // rcx
  __int64 result; // rax
  __int64 *v15; // r11
  unsigned __int64 *v16; // rdx
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rcx
  unsigned __int64 *v20; // rsi
  __int64 *v21; // rcx
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h]

  v9 = a3 & 1;
  v11 = (a3 - 1) / 2;
  v12 = a7;
  if ( a2 >= v11 )
  {
    v16 = (unsigned __int64 *)(a1 + 24 * a2);
    if ( v9 )
    {
      result = a8;
      v22 = _mm_loadu_si128((const __m128i *)&a7);
      v23 = a8;
      goto LABEL_13;
    }
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1) - 1;
    v15 = (__int64 *)(a1 + 48 * (i + 1));
    v16 = (unsigned __int64 *)(a1 + 24 * result);
    v17 = *v16;
    if ( *v15 >= *v16 )
    {
      v17 = *v15;
      v16 = (unsigned __int64 *)(a1 + 48 * (i + 1));
      result = 2 * (i + 1);
    }
    v18 = a1 + 24 * i;
    *(_QWORD *)v18 = v17;
    *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)(v16 + 1));
    if ( result >= v11 )
      break;
  }
  if ( !v9 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      v21 = (__int64 *)(a1 + 24 * result);
      *v16 = *v21;
      *(__m128i *)(v16 + 1) = _mm_loadu_si128((const __m128i *)(v21 + 1));
      v16 = (unsigned __int64 *)v21;
    }
  }
  v23 = a8;
  v22 = _mm_loadu_si128((const __m128i *)&a7);
  v19 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      result *= 3;
      v20 = (unsigned __int64 *)(a1 + 24 * v19);
      v16 = (unsigned __int64 *)(a1 + 8 * result);
      if ( *v20 >= v12 )
        break;
      *v16 = *v20;
      *(__m128i *)(v16 + 1) = _mm_loadu_si128((const __m128i *)(v20 + 1));
      result = v19;
      if ( a2 >= v19 )
      {
        v16 = (unsigned __int64 *)(a1 + 24 * v19);
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
LABEL_13:
  *v16 = v12;
  *(__m128i *)(v16 + 1) = _mm_loadu_si128((const __m128i *)&v22.m128i_u64[1]);
  return result;
}
