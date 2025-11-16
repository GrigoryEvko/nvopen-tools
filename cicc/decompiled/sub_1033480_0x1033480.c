// Function: sub_1033480
// Address: 0x1033480
//
__int64 __fastcall sub_1033480(__int64 a1, unsigned int a2)
{
  unsigned int v3; // r13d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // r13
  bool v13; // zf
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 j; // r14
  __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // esi
  int v22; // r10d
  _QWORD *v23; // r9
  unsigned int v24; // ecx
  _QWORD *v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rdi
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  _QWORD *v31; // r12
  __int64 v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // rax
  int v36; // edx
  _BYTE v37[400]; // [rsp+10h] [rbp-190h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = a1 + 16;
    v11 = a1 + 368;
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    goto LABEL_29;
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
  v3 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v10 = a1 + 16;
    v11 = a1 + 368;
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 88LL * (unsigned int)v6;
      goto LABEL_5;
    }
LABEL_29:
    v31 = v37;
    do
    {
      v32 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != -4096 && v32 != -8192 )
      {
        if ( v31 )
          *v31 = v32;
        v31[1] = 0;
        v33 = (__int64)(v31 + 1);
        v34 = v31 + 3;
        v31 += 11;
        *(v31 - 9) = 1;
        do
        {
          if ( v34 )
            *v34 = -4096;
          v34 += 2;
        }
        while ( v34 != v31 );
        sub_1033120(v33, v10 + 8);
        if ( (*(_BYTE *)(v10 + 16) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v10 + 24), 16LL * *(unsigned int *)(v10 + 32), 8);
      }
      v10 += 88;
    }
    while ( v10 != v11 );
    if ( v3 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v35 = sub_C7D670(88LL * v3, 8);
      *(_DWORD *)(a1 + 24) = v3;
      *(_QWORD *)(a1 + 16) = v35;
    }
    return sub_10332A0(a1, (__int64)v37, (__int64)v31);
  }
  if ( v5 )
  {
    v10 = a1 + 16;
    v11 = a1 + 368;
    v3 = 64;
    goto LABEL_29;
  }
  v7 = *(_DWORD *)(a1 + 24);
  v3 = 64;
  v8 = 5632;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v3;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v12 = 88LL * v7;
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v14 = v4 + v12;
  if ( v13 )
  {
    v15 = *(_QWORD **)(a1 + 16);
    v16 = 11LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v15 = (_QWORD *)(a1 + 16);
    v16 = 44;
  }
  for ( i = &v15[v16]; i != v15; v15 += 11 )
  {
    if ( v15 )
      *v15 = -4096;
  }
  for ( j = v4; v14 != j; j += 88 )
  {
    v19 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -8192 && v19 != -4096 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v20 = a1 + 16;
        v21 = 3;
      }
      else
      {
        v36 = *(_DWORD *)(a1 + 24);
        v20 = *(_QWORD *)(a1 + 16);
        if ( !v36 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v21 = v36 - 1;
      }
      v22 = 1;
      v23 = 0;
      v24 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v25 = (_QWORD *)(v20 + 88LL * v24);
      v26 = *v25;
      if ( *v25 != v19 )
      {
        while ( v26 != -4096 )
        {
          if ( !v23 && v26 == -8192 )
            v23 = v25;
          v24 = v21 & (v22 + v24);
          v25 = (_QWORD *)(v20 + 88LL * v24);
          v26 = *v25;
          if ( v19 == *v25 )
            goto LABEL_20;
          ++v22;
        }
        if ( v23 )
          v25 = v23;
      }
LABEL_20:
      *v25 = v19;
      v27 = (__int64)(v25 + 1);
      v28 = v25 + 3;
      v29 = v25 + 11;
      *(v29 - 10) = 0;
      *(v29 - 9) = 1;
      do
      {
        if ( v28 )
          *v28 = -4096;
        v28 += 2;
      }
      while ( v28 != v29 );
      sub_1033120(v27, j + 8);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( (*(_BYTE *)(j + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(j + 24), 16LL * *(unsigned int *)(j + 32), 8);
    }
  }
  return sub_C7D6A0(v4, v12, 8);
}
