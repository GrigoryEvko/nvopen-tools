// Function: sub_F1D1B0
// Address: 0xf1d1b0
//
__int64 __fastcall sub_F1D1B0(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r13
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  const __m128i *v12; // rax
  const __m128i *v13; // rcx
  __m128i *v14; // r13
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __int64 v17; // rax
  _BYTE v18[256]; // [rsp+0h] [rbp-100h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v3 = v6;
    if ( (unsigned int)v6 <= 0x40 )
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v3 = 64;
        v8 = 3584;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 56LL * v7;
        sub_F1D050(a1, v4, v4 + v10);
        return sub_C7D6A0(v4, v10, 8);
      }
      v12 = (const __m128i *)(a1 + 16);
      v13 = (const __m128i *)(a1 + 240);
      v3 = 64;
      goto LABEL_10;
    }
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 56LL * (unsigned int)v6;
      goto LABEL_5;
    }
  }
  v12 = (const __m128i *)(a1 + 16);
  v13 = (const __m128i *)(a1 + 240);
LABEL_10:
  v14 = (__m128i *)v18;
  do
  {
    if ( v12->m128i_i64[0] == -4096 )
    {
      if ( !v12->m128i_i64[1] && !v12[2].m128i_i8[0] )
      {
LABEL_22:
        if ( !v12[2].m128i_i64[1] )
          goto LABEL_16;
      }
    }
    else if ( v12->m128i_i64[0] == -8192
           && !v12->m128i_i64[1]
           && v12[2].m128i_i8[0]
           && !v12[1].m128i_i64[0]
           && !v12[1].m128i_i64[1] )
    {
      goto LABEL_22;
    }
    if ( v14 )
    {
      v15 = _mm_loadu_si128(v12 + 1);
      v16 = _mm_loadu_si128(v12 + 2);
      *v14 = _mm_loadu_si128(v12);
      v14[1] = v15;
      v14[2] = v16;
    }
    v14 = (__m128i *)((char *)v14 + 56);
    v14[-1].m128i_i64[1] = v12[3].m128i_i64[0];
LABEL_16:
    v12 = (const __m128i *)((char *)v12 + 56);
  }
  while ( v12 != v13 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v17 = sub_C7D670(56LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v17;
  }
  return sub_F1D050(a1, (__int64)v18, (__int64)v14);
}
