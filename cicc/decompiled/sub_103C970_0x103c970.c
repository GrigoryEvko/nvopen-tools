// Function: sub_103C970
// Address: 0x103c970
//
__int64 __fastcall sub_103C970(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // r12
  _QWORD *v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // r12
  _QWORD *v16; // r14
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r15
  unsigned __int64 *v19; // rdi
  unsigned __int64 v20; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // r9
  _QWORD *v24; // r8
  __int64 v25; // r10
  __int64 i; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx

  if ( *(_DWORD *)(a1 + 80) )
  {
    v22 = *(_QWORD **)(a1 + 72);
    v23 = &v22[2 * *(unsigned int *)(a1 + 88)];
    if ( v22 != v23 )
    {
      while ( 1 )
      {
        v24 = v22;
        if ( *v22 != -4096 && *v22 != -8192 )
          break;
        v22 += 2;
        if ( v23 == v22 )
          goto LABEL_2;
      }
      while ( v23 != v24 )
      {
        v25 = v24[1];
        for ( i = *(_QWORD *)(v25 + 8); v25 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          v27 = 32LL * (*(_DWORD *)(i - 28) & 0x7FFFFFF);
          if ( (*(_BYTE *)(i - 25) & 0x40) != 0 )
          {
            v28 = *(_QWORD *)(i - 40);
            v29 = v28 + v27;
          }
          else
          {
            v29 = i - 32;
            v28 = i - 32 - v27;
          }
          for ( ; v29 != v28; v28 += 32 )
          {
            if ( *(_QWORD *)v28 )
            {
              v30 = *(_QWORD *)(v28 + 8);
              **(_QWORD **)(v28 + 16) = v30;
              if ( v30 )
                *(_QWORD *)(v30 + 16) = *(_QWORD *)(v28 + 16);
            }
            *(_QWORD *)v28 = 0;
          }
        }
        v24 += 2;
        if ( v24 == v23 )
          break;
        while ( *v24 == -8192 || *v24 == -4096 )
        {
          v24 += 2;
          if ( v23 == v24 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  v2 = *(_QWORD *)(a1 + 344);
  if ( v2 )
    j_j___libc_free_0(v2, 24);
  v3 = *(_QWORD *)(a1 + 336);
  if ( v3 )
    j_j___libc_free_0(v3, 24);
  v4 = *(_QWORD *)(a1 + 328);
  if ( v4 )
  {
    v5 = 56LL * *(unsigned int *)(v4 + 2384);
    sub_C7D6A0(*(_QWORD *)(v4 + 2368), v5, 8);
    v6 = *(_QWORD *)(v4 + 40);
    if ( v6 != v4 + 56 )
      _libc_free(v6, v5);
    j_j___libc_free_0(v4, 2400);
  }
  v7 = 16LL * *(unsigned int *)(a1 + 320);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), v7, 8);
  if ( !*(_BYTE *)(a1 + 164) )
    _libc_free(*(_QWORD *)(a1 + 144), v7);
  v8 = *(_QWORD *)(a1 + 128);
  if ( v8 )
    sub_BD72D0(v8, v7);
  v9 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 104);
    v11 = &v10[2 * (unsigned int)v9];
    do
    {
      if ( *v10 != -4096 && *v10 != -8192 )
      {
        v12 = v10[1];
        if ( v12 )
          j_j___libc_free_0(v12, 16);
      }
      v10 += 2;
    }
    while ( v11 != v10 );
    v9 = *(unsigned int *)(a1 + 120);
  }
  v13 = 16 * v9;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 16 * v9, 8);
  v14 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 72);
    v16 = &v15[2 * (unsigned int)v14];
    do
    {
      if ( *v15 != -8192 && *v15 != -4096 )
      {
        v17 = (unsigned __int64 *)v15[1];
        if ( v17 )
        {
          v18 = (unsigned __int64 *)v17[1];
          while ( v17 != v18 )
          {
            v19 = v18;
            v18 = (unsigned __int64 *)v18[1];
            v20 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
            *v18 = v20 | *v18 & 7;
            *(_QWORD *)(v20 + 8) = v18;
            *v19 &= 7u;
            v19 -= 4;
            v19[5] = 0;
            sub_BD72D0((__int64)v19, v13);
          }
          v13 = 16;
          j_j___libc_free_0(v17, 16);
        }
      }
      v15 += 2;
    }
    while ( v16 != v15 );
    v14 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16 * v14, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 56), 8);
}
