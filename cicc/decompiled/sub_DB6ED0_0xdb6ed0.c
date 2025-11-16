// Function: sub_DB6ED0
// Address: 0xdb6ed0
//
_QWORD *__fastcall sub_DB6ED0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r12
  bool v13; // zf
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v19; // rdi
  int v20; // esi
  int v21; // r10d
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rdx
  int v29; // esi
  __int64 *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v35; // [rsp+8h] [rbp-118h]
  __int64 v36[34]; // [rsp+10h] [rbp-110h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = a1 + 16;
    v11 = a1 + 240;
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    goto LABEL_26;
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
  v2 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v10 = a1 + 16;
    v11 = a1 + 240;
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 56LL * (unsigned int)v6;
      goto LABEL_5;
    }
LABEL_26:
    v30 = v36;
    do
    {
      v31 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != -4096 && v31 != -8192 )
      {
        if ( v30 )
          *v30 = v31;
        v32 = *(_QWORD *)(v10 + 16);
        ++*(_QWORD *)(v10 + 8);
        v30[1] = 1;
        v30 += 7;
        *(v30 - 5) = v32;
        LODWORD(v32) = *(_DWORD *)(v10 + 24);
        *(_QWORD *)(v10 + 16) = 0;
        *((_DWORD *)v30 - 8) = v32;
        LODWORD(v32) = *(_DWORD *)(v10 + 28);
        *(_DWORD *)(v10 + 24) = 0;
        *((_DWORD *)v30 - 7) = v32;
        LODWORD(v32) = *(_DWORD *)(v10 + 32);
        *(_DWORD *)(v10 + 28) = 0;
        *((_DWORD *)v30 - 6) = v32;
        LOBYTE(v32) = *(_BYTE *)(v10 + 40);
        *(_DWORD *)(v10 + 32) = 0;
        *((_BYTE *)v30 - 16) = v32;
        v35 = v11;
        *((_BYTE *)v30 - 15) = *(_BYTE *)(v10 + 41);
        *(v30 - 1) = *(_QWORD *)(v10 + 48);
        sub_C7D6A0(0, 0, 8);
        v11 = v35;
      }
      v10 += 56;
    }
    while ( v10 != v11 );
    if ( v2 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v33 = sub_C7D670(56LL * v2, 8);
      *(_DWORD *)(a1 + 24) = v2;
      *(_QWORD *)(a1 + 16) = v33;
    }
    return sub_DB6CC0(a1, v36, v30);
  }
  if ( v5 )
  {
    v10 = a1 + 16;
    v11 = a1 + 240;
    v2 = 64;
    goto LABEL_26;
  }
  v7 = *(_DWORD *)(a1 + 24);
  v2 = 64;
  v8 = 3584;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v12 = 56LL * v7;
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v14 = v4 + v12;
  if ( v13 )
  {
    v15 = *(_QWORD **)(a1 + 16);
    v16 = 7LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v15 = (_QWORD *)(a1 + 16);
    v16 = 28;
  }
  for ( i = &v15[v16]; i != v15; v15 += 7 )
  {
    if ( v15 )
      *v15 = -4096;
  }
  for ( j = v4; v14 != j; j += 56 )
  {
    v28 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -8192 && v28 != -4096 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v19 = a1 + 16;
        v20 = 3;
      }
      else
      {
        v29 = *(_DWORD *)(a1 + 24);
        v19 = *(_QWORD *)(a1 + 16);
        if ( !v29 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v20 = v29 - 1;
      }
      v21 = 1;
      v22 = 0;
      v23 = v20 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v24 = v19 + 56LL * v23;
      v25 = *(_QWORD *)v24;
      if ( v28 != *(_QWORD *)v24 )
      {
        while ( v25 != -4096 )
        {
          if ( !v22 && v25 == -8192 )
            v22 = v24;
          v23 = v20 & (v21 + v23);
          v24 = v19 + 56LL * v23;
          v25 = *(_QWORD *)v24;
          if ( v28 == *(_QWORD *)v24 )
            goto LABEL_18;
          ++v21;
        }
        if ( v22 )
          v24 = v22;
      }
LABEL_18:
      *(_QWORD *)(v24 + 24) = 0;
      *(_QWORD *)(v24 + 16) = 0;
      *(_DWORD *)(v24 + 32) = 0;
      *(_QWORD *)v24 = v28;
      *(_QWORD *)(v24 + 8) = 1;
      v26 = *(_QWORD *)(j + 16);
      ++*(_QWORD *)(j + 8);
      v27 = *(_QWORD *)(v24 + 16);
      *(_QWORD *)(v24 + 16) = v26;
      LODWORD(v26) = *(_DWORD *)(j + 24);
      *(_QWORD *)(j + 16) = v27;
      LODWORD(v27) = *(_DWORD *)(v24 + 24);
      *(_DWORD *)(v24 + 24) = v26;
      LODWORD(v26) = *(_DWORD *)(j + 28);
      *(_DWORD *)(j + 24) = v27;
      LODWORD(v27) = *(_DWORD *)(v24 + 28);
      *(_DWORD *)(v24 + 28) = v26;
      LODWORD(v26) = *(_DWORD *)(j + 32);
      *(_DWORD *)(j + 28) = v27;
      LODWORD(v27) = *(_DWORD *)(v24 + 32);
      *(_DWORD *)(v24 + 32) = v26;
      *(_DWORD *)(j + 32) = v27;
      *(_BYTE *)(v24 + 40) = *(_BYTE *)(j + 40);
      *(_BYTE *)(v24 + 41) = *(_BYTE *)(j + 41);
      *(_QWORD *)(v24 + 48) = *(_QWORD *)(j + 48);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      sub_C7D6A0(*(_QWORD *)(j + 16), 16LL * *(unsigned int *)(j + 32), 8);
    }
  }
  return (_QWORD *)sub_C7D6A0(v4, v12, 8);
}
