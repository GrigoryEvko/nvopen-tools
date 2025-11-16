// Function: sub_B24D50
// Address: 0xb24d50
//
char __fastcall sub_B24D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __int64 v7; // r14
  __int64 i; // r15
  __int64 v9; // r12
  __m128i *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r8
  __int64 v18; // [rsp+20h] [rbp-60h]
  __m128i v19; // [rsp+30h] [rbp-50h] BYREF
  __m128i v20; // [rsp+40h] [rbp-40h] BYREF

  v7 = (a3 - 1) / 2;
  v18 = a3 & 1;
  if ( a2 >= v7 )
  {
    v10 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
    {
      v20.m128i_i64[0] = a4;
      LOBYTE(v11) = a5;
      v20.m128i_i64[1] = a5;
      goto LABEL_13;
    }
    v9 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = (__m128i *)(a1 + 32 * (i + 1));
    if ( sub_B1DED0((__int64)&a7, v10->m128i_i64, v10[-1].m128i_i64) )
    {
      --v9;
      v10 = (__m128i *)(a1 + 16 * v9);
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v10);
    if ( v9 >= v7 )
      break;
  }
  if ( !v18 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v9 )
    {
      v13 = v9 + 1;
      v9 = 2 * (v9 + 1) - 1;
      *v10 = _mm_loadu_si128((const __m128i *)(a1 + 32 * v13 - 16));
      v10 = (__m128i *)(a1 + 16 * v9);
    }
  }
  v20.m128i_i64[0] = a4;
  v19 = _mm_loadu_si128((const __m128i *)&a7);
  v20.m128i_i64[1] = a5;
  LOBYTE(v11) = v9 - 1;
  v12 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      LOBYTE(v11) = sub_B1DED0((__int64)&v19, (__int64 *)(a1 + 16 * v12), v20.m128i_i64);
      v10 = (__m128i *)(a1 + 16 * v9);
      if ( !(_BYTE)v11 )
        break;
      v9 = v12;
      *v10 = _mm_loadu_si128((const __m128i *)(a1 + 16 * v12));
      v11 = (v12 - 1) / 2;
      if ( a2 >= v12 )
      {
        v10 = (__m128i *)(a1 + 16 * v12);
        break;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = _mm_loadu_si128(&v20);
  return v11;
}
