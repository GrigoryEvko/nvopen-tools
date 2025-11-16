// Function: sub_F3E3C0
// Address: 0xf3e3c0
//
__int64 __fastcall sub_F3E3C0(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // rdi
  __int64 v8; // rax
  bool v9; // zf
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 i; // rdx
  __int64 v14; // r14
  __int64 v15; // rsi
  __m128i *v16; // rax
  const __m128i *v18; // r14
  __m128i *v19; // r13
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __int64 v22; // rax
  __int64 v23; // [rsp+18h] [rbp-138h]
  const __m128i *v24; // [rsp+18h] [rbp-138h]
  __int64 v25[3]; // [rsp+20h] [rbp-130h] BYREF
  char v26; // [rsp+38h] [rbp-118h]
  __int64 v27; // [rsp+40h] [rbp-110h]
  __m128i *v28; // [rsp+50h] [rbp-100h] BYREF
  __int64 v29; // [rsp+58h] [rbp-F8h]
  __int64 v30; // [rsp+60h] [rbp-F0h]
  __int64 v31; // [rsp+68h] [rbp-E8h]
  __int64 v32; // [rsp+70h] [rbp-E0h]
  _QWORD v33[26]; // [rsp+80h] [rbp-D0h] BYREF

  v3 = a2;
  if ( a2 > 4 )
  {
    v4 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v3 = v4;
    if ( (unsigned int)v4 <= 0x40 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
      {
        v5 = *(_QWORD *)(a1 + 16);
        v6 = *(_DWORD *)(a1 + 24);
        v3 = 64;
        v7 = 2560;
        goto LABEL_5;
      }
      v18 = (const __m128i *)(a1 + 16);
      v25[0] = 0;
      v3 = 64;
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = 0;
      v31 = 1;
      v32 = 0;
      v24 = (const __m128i *)(a1 + 176);
LABEL_23:
      v19 = (__m128i *)v33;
      do
      {
        if ( !sub_F34140((__int64)v18, (__int64)v25) && !sub_F34140((__int64)v18, (__int64)&v28) )
        {
          if ( v19 )
          {
            v20 = _mm_loadu_si128(v18);
            v21 = _mm_loadu_si128(v18 + 1);
            v19[2].m128i_i64[0] = v18[2].m128i_i64[0];
            *v19 = v20;
            v19[1] = v21;
          }
          v19 = (__m128i *)((char *)v19 + 40);
        }
        v18 = (const __m128i *)((char *)v18 + 40);
      }
      while ( v24 != v18 );
      if ( v3 > 4 )
      {
        *(_BYTE *)(a1 + 8) &= ~1u;
        v22 = sub_C7D670(40LL * v3, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v22;
      }
      return sub_F3E2B0(a1, (__int64)v33, (__int64)v19);
    }
  }
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v18 = (const __m128i *)(a1 + 16);
    v25[0] = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 1;
    v32 = 0;
    v24 = (const __m128i *)(a1 + 176);
    goto LABEL_23;
  }
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  if ( v3 <= 4 )
  {
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_9;
  }
  v7 = 40LL * v3;
LABEL_5:
  v8 = sub_C7D670(v7, 8);
  *(_DWORD *)(a1 + 24) = v3;
  *(_QWORD *)(a1 + 16) = v8;
LABEL_9:
  v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v23 = 40LL * v6;
  v10 = v5 + v23;
  if ( v9 )
  {
    v11 = *(_QWORD *)(a1 + 16);
    v12 = 40LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v11 = a1 + 16;
    v12 = 160;
  }
  for ( i = v11 + v12; i != v11; v11 += 40 )
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = 0;
      *(_BYTE *)(v11 + 24) = 0;
      *(_QWORD *)(v11 + 32) = 0;
    }
  }
  memset(v33, 0, 24);
  v33[3] = 1;
  v33[4] = 0;
  if ( v10 != v5 )
  {
    v14 = v5;
    do
    {
      while ( !*(_QWORD *)v14 && !*(_BYTE *)(v14 + 24) && !*(_QWORD *)(v14 + 32) || sub_F34140(v14, (__int64)v33) )
      {
        v14 += 40;
        if ( v10 == v14 )
          return sub_C7D6A0(v5, v23, 8);
      }
      v15 = v14;
      v14 += 40;
      sub_F38D60(a1, v15, (__int64 *)&v28);
      v16 = v28;
      *v28 = _mm_loadu_si128((const __m128i *)(v14 - 40));
      v16[1] = _mm_loadu_si128((const __m128i *)(v14 - 24));
      v16[2].m128i_i64[0] = *(_QWORD *)(v14 - 8);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v10 != v14 );
  }
  return sub_C7D6A0(v5, v23, 8);
}
