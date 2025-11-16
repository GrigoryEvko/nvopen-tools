// Function: sub_D5FDE0
// Address: 0xd5fde0
//
__int64 __fastcall sub_D5FDE0(__int64 a1, unsigned int a2)
{
  __int64 v4; // r13
  char v5; // si
  unsigned __int64 v6; // rdx
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  bool v11; // zf
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v17; // rdi
  int v18; // esi
  int v19; // r10d
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // r8
  int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rax
  int v28; // edx
  __int64 v30; // rax
  __int64 v31; // rcx
  _QWORD *v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  _BYTE v35[368]; // [rsp+0h] [rbp-170h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v30 = a1 + 16;
    v31 = a1 + 336;
    goto LABEL_33;
  }
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
  a2 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v30 = a1 + 16;
    v31 = a1 + 336;
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 40LL * (unsigned int)v6;
      goto LABEL_5;
    }
LABEL_33:
    v32 = v35;
    do
    {
      v33 = *(_QWORD *)v30;
      if ( *(_QWORD *)v30 != -4096 && v33 != -8192 )
      {
        if ( v32 )
          *v32 = v33;
        v32 += 5;
        *((_DWORD *)v32 - 6) = *(_DWORD *)(v30 + 16);
        *(v32 - 4) = *(_QWORD *)(v30 + 8);
        *((_DWORD *)v32 - 2) = *(_DWORD *)(v30 + 32);
        *(v32 - 2) = *(_QWORD *)(v30 + 24);
      }
      v30 += 40;
    }
    while ( v30 != v31 );
    if ( a2 > 8 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v34 = sub_C7D670(40LL * a2, 8);
      *(_DWORD *)(a1 + 24) = a2;
      *(_QWORD *)(a1 + 16) = v34;
    }
    return sub_D5FC10(a1, (__int64)v35, (__int64)v32);
  }
  if ( v5 )
  {
    v30 = a1 + 16;
    v31 = a1 + 336;
    a2 = 64;
    goto LABEL_33;
  }
  v7 = *(_DWORD *)(a1 + 24);
  a2 = 64;
  v8 = 2560;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v10 = 40LL * v7;
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v12 = v4 + v10;
  if ( v11 )
  {
    v13 = *(_QWORD **)(a1 + 16);
    v14 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v13 = (_QWORD *)(a1 + 16);
    v14 = 40;
  }
  for ( i = &v13[v14]; i != v13; v13 += 5 )
  {
    if ( v13 )
      *v13 = -4096;
  }
  for ( j = v4; v12 != j; j += 40 )
  {
    v27 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -8192 && v27 != -4096 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v17 = a1 + 16;
        v18 = 7;
      }
      else
      {
        v28 = *(_DWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        if ( !v28 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v18 = v28 - 1;
      }
      v19 = 1;
      v20 = 0;
      v21 = v18 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v22 = v17 + 40LL * v21;
      v23 = *(_QWORD *)v22;
      if ( v27 != *(_QWORD *)v22 )
      {
        while ( v23 != -4096 )
        {
          if ( !v20 && v23 == -8192 )
            v20 = v22;
          v21 = v18 & (v19 + v21);
          v22 = v17 + 40LL * v21;
          v23 = *(_QWORD *)v22;
          if ( v27 == *(_QWORD *)v22 )
            goto LABEL_18;
          ++v19;
        }
        if ( v20 )
          v22 = v20;
      }
LABEL_18:
      *(_QWORD *)v22 = v27;
      *(_DWORD *)(v22 + 16) = *(_DWORD *)(j + 16);
      *(_QWORD *)(v22 + 8) = *(_QWORD *)(j + 8);
      v24 = *(_DWORD *)(j + 32);
      *(_DWORD *)(j + 16) = 0;
      *(_DWORD *)(v22 + 32) = v24;
      *(_QWORD *)(v22 + 24) = *(_QWORD *)(j + 24);
      *(_DWORD *)(j + 32) = 0;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( *(_DWORD *)(j + 32) > 0x40u )
      {
        v25 = *(_QWORD *)(j + 24);
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
      if ( *(_DWORD *)(j + 16) > 0x40u )
      {
        v26 = *(_QWORD *)(j + 8);
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
    }
  }
  return sub_C7D6A0(v4, v10, 8);
}
