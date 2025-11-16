// Function: sub_B21EE0
// Address: 0xb21ee0
//
_QWORD *__fastcall sub_B21EE0(__int64 a1, unsigned int a2)
{
  unsigned int v3; // r12d
  char v4; // r14
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r14
  bool v10; // zf
  __int64 v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v16; // rdi
  __int64 v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v21; // rsi
  __int64 v22; // rbx
  _QWORD *v23; // r15
  __int64 v24; // rax
  int v25; // edi
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // [rsp+0h] [rbp-160h]
  __int64 v31; // [rsp+0h] [rbp-160h]
  __int64 v32; // [rsp+8h] [rbp-158h]
  _QWORD v33[42]; // [rsp+10h] [rbp-150h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 8);
  v32 = *(_QWORD *)(a1 + 16);
  if ( a2 <= 4 )
  {
    if ( (v4 & 1) == 0 )
    {
      v6 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) = v4 | 1;
      goto LABEL_6;
    }
    v22 = a1 + 16;
    v31 = a1 + 304;
  }
  else
  {
    v5 = sub_AF1560(a2 - 1);
    v3 = v5;
    if ( v5 > 0x40 )
    {
      v22 = a1 + 16;
      v31 = a1 + 304;
      if ( (v4 & 1) == 0 )
      {
        v6 = *(_DWORD *)(a1 + 24);
        v7 = 72LL * v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( (v4 & 1) == 0 )
      {
        v6 = *(_DWORD *)(a1 + 24);
        v7 = 4608;
        v3 = 64;
LABEL_5:
        v8 = sub_C7D670(v7, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v8;
LABEL_6:
        v9 = 72LL * v6;
        v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v11 = v32 + v9;
        if ( v10 )
        {
          v12 = *(_QWORD **)(a1 + 16);
          v13 = 9LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v12 = (_QWORD *)(a1 + 16);
          v13 = 36;
        }
        for ( i = &v12[v13]; i != v12; v12 += 9 )
        {
          if ( v12 )
            *v12 = -4096;
        }
        for ( j = v32; v11 != j; j += 72 )
        {
          if ( *(_QWORD *)j != -4096 && *(_QWORD *)j != -8192 )
          {
            v21 = j;
            sub_B1BD90(a1, (__int64 *)j, v33);
            *(_QWORD *)v33[0] = *(_QWORD *)j;
            v18 = (_QWORD *)v33[0];
            v19 = v33[0] + 24LL;
            *(_QWORD *)(v33[0] + 16LL) = 0x200000000LL;
            v18[1] = v19;
            if ( *(_DWORD *)(j + 16) )
            {
              v21 = j + 8;
              v30 = v18;
              sub_B187A0((__int64)(v18 + 1), (char **)(j + 8));
              v18 = v30;
            }
            v18[6] = 0x200000000LL;
            v18[5] = v18 + 7;
            if ( *(_DWORD *)(j + 48) )
            {
              v21 = j + 40;
              sub_B187A0((__int64)(v18 + 5), (char **)(j + 40));
            }
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v16 = *(_QWORD *)(j + 40);
            if ( v16 != j + 56 )
              _libc_free(v16, v21);
            v17 = *(_QWORD *)(j + 8);
            if ( v17 != j + 24 )
              _libc_free(v17, v21);
          }
        }
        return (_QWORD *)sub_C7D6A0(v32, v9, 8);
      }
      v22 = a1 + 16;
      v3 = 64;
      v31 = a1 + 304;
    }
  }
  v23 = v33;
  do
  {
    v24 = *(_QWORD *)v22;
    if ( *(_QWORD *)v22 != -4096 && v24 != -8192 )
    {
      if ( v23 )
        *v23 = v24;
      v25 = *(_DWORD *)(v22 + 16);
      v23[2] = 0x200000000LL;
      v23[1] = v23 + 3;
      if ( v25 )
        sub_B187A0((__int64)(v23 + 1), (char **)(v22 + 8));
      v26 = *(unsigned int *)(v22 + 48);
      v23[6] = 0x200000000LL;
      v23[5] = v23 + 7;
      if ( (_DWORD)v26 )
      {
        v26 = v22 + 40;
        sub_B187A0((__int64)(v23 + 5), (char **)(v22 + 40));
      }
      v27 = *(_QWORD *)(v22 + 40);
      v23 += 9;
      if ( v27 != v22 + 56 )
        _libc_free(v27, v26);
      v28 = *(_QWORD *)(v22 + 8);
      if ( v28 != v22 + 24 )
        _libc_free(v28, v26);
    }
    v22 += 72;
  }
  while ( v22 != v31 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v29 = sub_C7D670(72LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v29;
  }
  return sub_B21CE0(a1, (__int64)v33, (__int64)v23);
}
