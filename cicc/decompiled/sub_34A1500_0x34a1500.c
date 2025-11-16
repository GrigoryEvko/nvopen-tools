// Function: sub_34A1500
// Address: 0x34a1500
//
void __fastcall sub_34A1500(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v8; // r13
  char v9; // dl
  unsigned __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r15
  _BYTE *v16; // r8
  __m128i *v17; // r13
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  int v20; // eax
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  _BYTE *v23; // [rsp+8h] [rbp-278h]
  _BYTE *v24; // [rsp+8h] [rbp-278h]
  _BYTE *v25; // [rsp+8h] [rbp-278h]
  _BYTE v26[624]; // [rsp+10h] [rbp-270h] BYREF

  v6 = a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
LABEL_9:
    v14 = a1 + 16;
    v15 = a1 + 592;
    goto LABEL_10;
  }
  a4 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
  v10 = (a4
       | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
         | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
         | (a2 - 1)
         | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
       | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
       | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
       | (a2 - 1)
       | ((unsigned __int64)(a2 - 1) >> 1))
      + 1;
  v6 = v10;
  if ( (unsigned int)v10 > 0x40 )
  {
    if ( !v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 72LL * (unsigned int)v10;
      goto LABEL_5;
    }
    goto LABEL_9;
  }
  if ( !v9 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    v6 = 64;
    v12 = 4608;
LABEL_5:
    v13 = sub_C7D670(v12, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
    sub_34A13A0(a1, v8, v8 + 72LL * v11);
    sub_C7D6A0(v8, 72LL * v11, 8);
    return;
  }
  v14 = a1 + 16;
  v15 = a1 + 592;
  v6 = 64;
LABEL_10:
  v16 = v26;
  v17 = (__m128i *)v26;
  do
  {
    if ( *(_QWORD *)v14
      || *(_BYTE *)(v14 + 24) && (*(_QWORD *)(v14 + 8) || *(_QWORD *)(v14 + 16))
      || *(_QWORD *)(v14 + 32) )
    {
      if ( v17 )
      {
        v18 = _mm_loadu_si128((const __m128i *)v14);
        v19 = _mm_loadu_si128((const __m128i *)(v14 + 16));
        v17[2].m128i_i64[0] = *(_QWORD *)(v14 + 32);
        *v17 = v18;
        v17[1] = v19;
      }
      v17[3].m128i_i64[0] = 0x200000000LL;
      v20 = *(_DWORD *)(v14 + 48);
      v17[2].m128i_i64[1] = (__int64)&v17[3].m128i_i64[1];
      if ( v20 )
      {
        v24 = v16;
        sub_349D9E0(
          (__int64)&v17[2].m128i_i64[1],
          (char **)(v14 + 40),
          (__int64)&v17[3].m128i_i64[1],
          a4,
          (__int64)v16,
          a6);
        v16 = v24;
      }
      v21 = *(_QWORD *)(v14 + 40);
      v17 = (__m128i *)((char *)v17 + 72);
      if ( v21 != v14 + 56 )
      {
        v23 = v16;
        _libc_free(v21);
        v16 = v23;
      }
    }
    v14 += 72;
  }
  while ( v14 != v15 );
  if ( v6 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v25 = v16;
    v22 = sub_C7D670(72LL * v6, 8);
    *(_DWORD *)(a1 + 24) = v6;
    v16 = v25;
    *(_QWORD *)(a1 + 16) = v22;
  }
  sub_34A13A0(a1, (__int64)v16, (__int64)v17);
}
