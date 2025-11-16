// Function: sub_1362490
// Address: 0x1362490
//
__int64 __fastcall sub_1362490(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  int v7; // ebx
  __int64 v8; // rdi
  const __m128i *v9; // rax
  __m128i *v10; // r13
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE v18[736]; // [rsp+0h] [rbp-2E0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = *(_QWORD *)(a1 + 16);
    v15 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 16);
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
  v7 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v8 = 88LL * (unsigned int)v6;
    if ( v4 )
      goto LABEL_5;
    v15 = *(_DWORD *)(a1 + 24);
    goto LABEL_25;
  }
  if ( !v4 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    v7 = 64;
    v8 = 5632;
LABEL_25:
    v16 = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v16;
LABEL_21:
    sub_1362280(a1, v5, v5 + 88LL * v15);
    return j___libc_free_0(v5);
  }
  v8 = 5632;
  v7 = 64;
LABEL_5:
  v9 = (const __m128i *)(a1 + 16);
  v10 = (__m128i *)v18;
  do
  {
    if ( v9->m128i_i64[0] == -8 )
    {
      if ( !v9->m128i_i64[1]
        && !v9[1].m128i_i64[0]
        && !v9[1].m128i_i64[1]
        && !v9[2].m128i_i64[0]
        && v9[2].m128i_i64[1] == -8 )
      {
        goto LABEL_31;
      }
    }
    else if ( v9->m128i_i64[0] == -16
           && !v9->m128i_i64[1]
           && !v9[1].m128i_i64[0]
           && !v9[1].m128i_i64[1]
           && !v9[2].m128i_i64[0]
           && v9[2].m128i_i64[1] == -16 )
    {
LABEL_31:
      if ( !v9[3].m128i_i64[0] && !v9[3].m128i_i64[1] && !v9[4].m128i_i64[0] && !v9[4].m128i_i64[1] )
        goto LABEL_11;
    }
    if ( v10 )
    {
      v11 = _mm_loadu_si128(v9 + 1);
      v12 = _mm_loadu_si128(v9 + 2);
      v13 = _mm_loadu_si128(v9 + 3);
      v14 = _mm_loadu_si128(v9 + 4);
      *v10 = _mm_loadu_si128(v9);
      v10[1] = v11;
      v10[2] = v12;
      v10[3] = v13;
      v10[4] = v14;
    }
    v10 = (__m128i *)((char *)v10 + 88);
    v10[-1].m128i_i8[8] = v9[5].m128i_i8[0];
LABEL_11:
    v9 = (const __m128i *)((char *)v9 + 88);
  }
  while ( (const __m128i *)(a1 + 720) != v9 );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v17 = sub_22077B0(v8);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v17;
  return sub_1362280(a1, (__int64)v18, (__int64)v10);
}
