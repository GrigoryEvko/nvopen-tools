// Function: sub_F3B390
// Address: 0xf3b390
//
__int64 __fastcall sub_F3B390(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // r12d
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v11; // r13
  __m128i *v12; // r12
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-218h]
  _QWORD v23[3]; // [rsp+10h] [rbp-210h] BYREF
  char v24; // [rsp+28h] [rbp-1F8h]
  __int64 v25; // [rsp+30h] [rbp-1F0h]
  __int64 v26; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 v27; // [rsp+48h] [rbp-1D8h]
  __int64 v28; // [rsp+50h] [rbp-1D0h]
  __int64 v29; // [rsp+58h] [rbp-1C8h]
  __int64 v30; // [rsp+60h] [rbp-1C0h]
  _BYTE v31[432]; // [rsp+70h] [rbp-1B0h] BYREF

  v3 = a2;
  if ( a2 <= 4
    || (v4 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
           + 1,
        v3 = v4,
        (unsigned int)v4 > 0x40) )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v5 = *(_QWORD *)(a1 + 16);
      v6 = *(_DWORD *)(a1 + 24);
      if ( v3 <= 4 )
      {
        *(_BYTE *)(a1 + 8) |= 1u;
        goto LABEL_9;
      }
      v7 = 96LL * v3;
LABEL_5:
      v8 = sub_C7D670(v7, 8);
      *(_DWORD *)(a1 + 24) = v3;
      *(_QWORD *)(a1 + 16) = v8;
LABEL_9:
      v9 = 96LL * v6;
      sub_F3B210(a1, v5, v5 + v9);
      return sub_C7D6A0(v5, v9, 8);
    }
    v11 = a1 + 16;
    v23[0] = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 1;
    v30 = 0;
    v22 = a1 + 400;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v5 = *(_QWORD *)(a1 + 16);
      v6 = *(_DWORD *)(a1 + 24);
      v3 = 64;
      v7 = 6144;
      goto LABEL_5;
    }
    v11 = a1 + 16;
    v23[0] = 0;
    v3 = 64;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 1;
    v30 = 0;
    v22 = a1 + 400;
  }
  v12 = (__m128i *)v31;
  do
  {
    if ( !sub_F34140(v11, (__int64)v23) )
    {
      v13 = &v26;
      if ( !sub_F34140(v11, (__int64)&v26) )
      {
        if ( v12 )
        {
          v18 = _mm_loadu_si128((const __m128i *)v11);
          v19 = _mm_loadu_si128((const __m128i *)(v11 + 16));
          v12[2].m128i_i64[0] = *(_QWORD *)(v11 + 32);
          *v12 = v18;
          v12[1] = v19;
        }
        v12[2].m128i_i64[1] = (__int64)&v12[3].m128i_i64[1];
        v12[3].m128i_i64[0] = 0x400000000LL;
        if ( *(_DWORD *)(v11 + 48) )
        {
          v13 = (__int64 *)(v11 + 40);
          sub_F334E0((__int64)&v12[2].m128i_i64[1], (char **)(v11 + 40), v14, v15, v16, v17);
        }
        v20 = *(_QWORD *)(v11 + 40);
        v12 += 6;
        v12[-1].m128i_i64[1] = *(_QWORD *)(v11 + 88);
        if ( v20 != v11 + 56 )
          _libc_free(v20, v13);
      }
    }
    v11 += 96;
  }
  while ( v11 != v22 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v21 = sub_C7D670(96LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v21;
  }
  return sub_F3B210(a1, (__int64)v31, (__int64)v12);
}
