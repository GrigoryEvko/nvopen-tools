// Function: sub_2ED9250
// Address: 0x2ed9250
//
void __fastcall sub_2ED9250(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  char v4; // cl
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  bool v11; // zf
  __int64 v12; // r15
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  __int64 j; // rbx
  int v17; // eax
  __int64 v18; // rdi
  int v19; // esi
  int v20; // r10d
  __int64 v21; // r9
  unsigned int v22; // ecx
  __int64 v23; // rdx
  int v24; // r8d
  __int64 v25; // rax
  unsigned __int64 *v26; // rax
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // rsi
  int *v30; // r13
  __int64 v31; // rax
  unsigned __int64 *v32; // [rsp+8h] [rbp-78h]
  int v33[28]; // [rsp+10h] [rbp-70h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v28 = a1 + 16;
    v29 = a1 + 80;
    goto LABEL_32;
  }
  v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v3 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    v28 = a1 + 16;
    v29 = a1 + 80;
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v8 = 16LL * (unsigned int)v5;
      goto LABEL_5;
    }
LABEL_32:
    v30 = v33;
    do
    {
      if ( *(_DWORD *)v28 <= 0xFFFFFFFD )
      {
        if ( v30 )
          *v30 = *(_DWORD *)v28;
        v30 += 4;
        *((_QWORD *)v30 - 1) = *(_QWORD *)(v28 + 8);
      }
      v28 += 16;
    }
    while ( v28 != v29 );
    if ( v3 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v31 = sub_C7D670(16LL * v3, 8);
      *(_DWORD *)(a1 + 24) = v3;
      *(_QWORD *)(a1 + 16) = v31;
    }
    sub_2ED90B0(a1, v33, v30);
    return;
  }
  if ( v4 )
  {
    v28 = a1 + 16;
    v29 = a1 + 80;
    v3 = 64;
    goto LABEL_32;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(unsigned int *)(a1 + 24);
  v3 = 64;
  v8 = 1024;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v3;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
  v10 = 16 * v7;
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v12 = v6 + v10;
  if ( v11 )
  {
    v13 = *(_DWORD **)(a1 + 16);
    v14 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v13 = (_DWORD *)(a1 + 16);
    v14 = 16;
  }
  for ( i = &v13[v14]; i != v13; v13 += 4 )
  {
    if ( v13 )
      *v13 = -1;
  }
  for ( j = v6; v12 != j; j += 16 )
  {
    while ( 1 )
    {
      v17 = *(_DWORD *)j;
      if ( *(_DWORD *)j <= 0xFFFFFFFD )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v18 = a1 + 16;
          v19 = 3;
        }
        else
        {
          v27 = *(_DWORD *)(a1 + 24);
          v18 = *(_QWORD *)(a1 + 16);
          if ( !v27 )
          {
            MEMORY[0] = *(_DWORD *)j;
            BUG();
          }
          v19 = v27 - 1;
        }
        v20 = 1;
        v21 = 0;
        v22 = v19 & (37 * v17);
        v23 = v18 + 16LL * v22;
        v24 = *(_DWORD *)v23;
        if ( v17 != *(_DWORD *)v23 )
        {
          while ( v24 != -1 )
          {
            if ( v24 == -2 && !v21 )
              v21 = v23;
            v22 = v19 & (v20 + v22);
            v23 = v18 + 16LL * v22;
            v24 = *(_DWORD *)v23;
            if ( v17 == *(_DWORD *)v23 )
              goto LABEL_19;
            ++v20;
          }
          if ( v21 )
            v23 = v21;
        }
LABEL_19:
        *(_DWORD *)v23 = *(_DWORD *)j;
        *(_QWORD *)(v23 + 8) = *(_QWORD *)(j + 8);
        *(_QWORD *)(j + 8) = 0;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v25 = *(_QWORD *)(j + 8);
        if ( v25 )
        {
          if ( (v25 & 2) != 0 )
          {
            v26 = (unsigned __int64 *)(v25 & 0xFFFFFFFFFFFFFFFCLL);
            if ( v26 )
              break;
          }
        }
      }
      j += 16;
      if ( v12 == j )
        goto LABEL_25;
    }
    if ( (unsigned __int64 *)*v26 != v26 + 2 )
    {
      v32 = v26;
      _libc_free(*v26);
      v26 = v32;
    }
    j_j___libc_free_0((unsigned __int64)v26);
  }
LABEL_25:
  sub_C7D6A0(v6, v10, 8);
}
