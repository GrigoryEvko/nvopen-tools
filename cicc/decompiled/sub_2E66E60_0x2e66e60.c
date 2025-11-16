// Function: sub_2E66E60
// Address: 0x2e66e60
//
void __fastcall sub_2E66E60(__int64 a1, unsigned int a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 v8; // r15
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v20; // rdi
  int v21; // esi
  int v22; // r11d
  __int64 *v23; // r10
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r9
  __int64 v27; // rcx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  int v31; // esi
  __int64 v32; // rbx
  _QWORD *v33; // r15
  __int64 v34; // rax
  int v35; // edi
  int v36; // esi
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  __int64 *v40; // [rsp+0h] [rbp-160h]
  __int64 v41; // [rsp+8h] [rbp-158h]
  __int64 v42; // [rsp+8h] [rbp-158h]
  _BYTE v43[336]; // [rsp+10h] [rbp-150h] BYREF

  v7 = a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !(_BYTE)v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v32 = a1 + 16;
    v42 = a1 + 304;
    goto LABEL_35;
  }
  a3 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
  v10 = (a3
       | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
         | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
         | (a2 - 1)
         | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
       | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
       | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
       | (a2 - 1)
       | ((unsigned __int64)(a2 - 1) >> 1))
      + 1;
  v7 = v10;
  if ( (unsigned int)v10 > 0x40 )
  {
    v32 = a1 + 16;
    v42 = a1 + 304;
    if ( !(_BYTE)v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 72LL * (unsigned int)v10;
      goto LABEL_5;
    }
LABEL_35:
    v33 = v43;
    do
    {
      v34 = *(_QWORD *)v32;
      if ( *(_QWORD *)v32 != -4096 && v34 != -8192 )
      {
        if ( v33 )
          *v33 = v34;
        v35 = *(_DWORD *)(v32 + 16);
        v33[2] = 0x200000000LL;
        v33[1] = v33 + 3;
        if ( v35 )
          sub_2E64060((__int64)(v33 + 1), (char **)(v32 + 8), a3, v9, a5, a6);
        v36 = *(_DWORD *)(v32 + 48);
        v33[6] = 0x200000000LL;
        v33[5] = v33 + 7;
        if ( v36 )
          sub_2E64060((__int64)(v33 + 5), (char **)(v32 + 40), a3, v9, a5, a6);
        v37 = *(_QWORD *)(v32 + 40);
        v33 += 9;
        if ( v37 != v32 + 56 )
          _libc_free(v37);
        v38 = *(_QWORD *)(v32 + 8);
        if ( v38 != v32 + 24 )
          _libc_free(v38);
      }
      v32 += 72;
    }
    while ( v32 != v42 );
    if ( v7 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v39 = sub_C7D670(72LL * v7, 8);
      *(_DWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 16) = v39;
    }
    sub_2E66C60(a1, (__int64)v43, (__int64)v33);
    return;
  }
  if ( (_BYTE)v9 )
  {
    v32 = a1 + 16;
    v7 = 64;
    v42 = a1 + 304;
    goto LABEL_35;
  }
  v11 = *(_DWORD *)(a1 + 24);
  v7 = 64;
  v12 = 4608;
LABEL_5:
  v13 = sub_C7D670(v12, 8);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v13;
LABEL_6:
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v41 = 72LL * v11;
  v15 = v8 + v41;
  if ( v14 )
  {
    v16 = *(_QWORD **)(a1 + 16);
    v17 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v16 = (_QWORD *)(a1 + 16);
    v17 = 36;
  }
  for ( i = &v16[v17]; i != v16; v16 += 9 )
  {
    if ( v16 )
      *v16 = -4096;
  }
  for ( j = v8; v15 != j; j += 72 )
  {
    v30 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -8192 && v30 != -4096 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v20 = a1 + 16;
        v21 = 3;
      }
      else
      {
        v31 = *(_DWORD *)(a1 + 24);
        v20 = *(_QWORD *)(a1 + 16);
        if ( !v31 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v21 = v31 - 1;
      }
      v22 = 1;
      v23 = 0;
      v24 = v21 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v25 = (__int64 *)(v20 + 72LL * v24);
      v26 = *v25;
      if ( v30 != *v25 )
      {
        while ( v26 != -4096 )
        {
          if ( !v23 && v26 == -8192 )
            v23 = v25;
          a5 = (unsigned int)(v22 + 1);
          v24 = v21 & (v22 + v24);
          v25 = (__int64 *)(v20 + 72LL * v24);
          v26 = *v25;
          if ( v30 == *v25 )
            goto LABEL_16;
          ++v22;
        }
        if ( v23 )
          v25 = v23;
      }
LABEL_16:
      *v25 = v30;
      v25[1] = (__int64)(v25 + 3);
      v25[2] = 0x200000000LL;
      v27 = *(unsigned int *)(j + 16);
      if ( (_DWORD)v27 )
      {
        v40 = v25;
        sub_2E64060((__int64)(v25 + 1), (char **)(j + 8), (__int64)v25, v27, a5, v26);
        v25 = v40;
      }
      v25[6] = 0x200000000LL;
      v25[5] = (__int64)(v25 + 7);
      if ( *(_DWORD *)(j + 48) )
        sub_2E64060((__int64)(v25 + 5), (char **)(j + 40), (__int64)v25, v27, a5, v26);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v28 = *(_QWORD *)(j + 40);
      if ( v28 != j + 56 )
        _libc_free(v28);
      v29 = *(_QWORD *)(j + 8);
      if ( v29 != j + 24 )
        _libc_free(v29);
    }
  }
  sub_C7D6A0(v8, v41, 8);
}
