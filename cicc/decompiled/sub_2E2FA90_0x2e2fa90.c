// Function: sub_2E2FA90
// Address: 0x2e2fa90
//
__int64 __fastcall sub_2E2FA90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v9; // r11
  __int64 v10; // r12
  unsigned int v12; // r9d
  __int64 i; // rcx
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rcx
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h]

  v9 = (a3 - 1) / 2;
  v10 = a3 & 1;
  v12 = a7;
  if ( a2 >= v9 )
  {
    v18 = a1 + 24 * a2;
    if ( v10 )
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
    v14 = i + 1;
    result = 2 * v14;
    v16 = a1 + 24 * i;
    v17 = 2 * v14 - 1;
    v18 = a1 + 48 * v14;
    if ( *(_DWORD *)v18 < *(_DWORD *)(a1 + 24 * v17) )
    {
      v18 = a1 + 24 * v17;
      result = v17;
    }
    *(_DWORD *)v16 = *(_DWORD *)v18;
    *(__m128i *)(v16 + 8) = _mm_loadu_si128((const __m128i *)(v18 + 8));
    if ( result >= v9 )
      break;
  }
  if ( !v10 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      v21 = a1 + 24 * result;
      *(_DWORD *)v18 = *(_DWORD *)v21;
      *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)(v21 + 8));
      v18 = v21;
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
      v20 = a1 + 24 * v19;
      v18 = a1 + 8 * result;
      if ( *(_DWORD *)v20 >= v12 )
        break;
      *(_DWORD *)v18 = *(_DWORD *)v20;
      *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)(v20 + 8));
      result = v19;
      if ( a2 >= v19 )
      {
        v18 = a1 + 24 * v19;
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
LABEL_13:
  *(_DWORD *)v18 = v12;
  *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)&v22.m128i_u64[1]);
  return result;
}
