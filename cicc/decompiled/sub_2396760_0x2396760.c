// Function: sub_2396760
// Address: 0x2396760
//
void __fastcall sub_2396760(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  bool v11; // zf
  __int64 *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v17; // rsi
  int v18; // edi
  int v19; // r10d
  _QWORD *v20; // r9
  unsigned int v21; // ecx
  _QWORD *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  unsigned __int64 *v25; // rax
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // r13
  __int64 *v29; // r13
  __int64 *v30; // r15
  __int64 *v31; // r8
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 *v36; // [rsp+8h] [rbp-58h]
  __int64 *v37; // [rsp+8h] [rbp-58h]
  __int64 *v38; // [rsp+8h] [rbp-58h]
  _BYTE v39[80]; // [rsp+10h] [rbp-50h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 2 )
  {
    if ( !v5 )
    {
      v28 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      v9 = 16 * v28;
      v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v12 = (__int64 *)(v4 + v9);
      if ( !v11 )
        goto LABEL_6;
LABEL_30:
      v13 = *(_QWORD **)(a1 + 16);
      v14 = 2LL * *(unsigned int *)(a1 + 24);
      goto LABEL_7;
    }
    v29 = (__int64 *)(a1 + 16);
    v30 = (__int64 *)(a1 + 48);
    goto LABEL_32;
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
  if ( (unsigned int)v6 <= 0x40 )
  {
    if ( !v5 )
    {
      v7 = *(unsigned int *)(a1 + 24);
      v3 = 64;
      v8 = 1024;
      goto LABEL_5;
    }
    v29 = (__int64 *)(a1 + 16);
    v30 = (__int64 *)(a1 + 48);
    v3 = 64;
LABEL_32:
    v31 = (__int64 *)v39;
    v32 = (__int64 *)v39;
    do
    {
      v33 = *v29;
      if ( *v29 != -4096 && v33 != -8192 )
      {
        if ( v32 )
          *v32 = v33;
        v34 = v29[1];
        v37 = v31;
        v32 += 2;
        v29[1] = 0;
        *(v32 - 1) = v34;
        sub_2396550(v29 + 1);
        v31 = v37;
      }
      v29 += 2;
    }
    while ( v29 != v30 );
    if ( v3 > 2 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v38 = v31;
      v35 = sub_C7D670(16LL * v3, 8);
      *(_DWORD *)(a1 + 24) = v3;
      v31 = v38;
      *(_QWORD *)(a1 + 16) = v35;
    }
    sub_23965B0(a1, v31, v32);
    return;
  }
  v29 = (__int64 *)(a1 + 16);
  v30 = (__int64 *)(a1 + 48);
  if ( v5 )
    goto LABEL_32;
  v7 = *(unsigned int *)(a1 + 24);
  v8 = 16LL * (unsigned int)v6;
LABEL_5:
  v9 = 16 * v7;
  v10 = sub_C7D670(v8, 8);
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v12 = (__int64 *)(v4 + v9);
  *(_QWORD *)(a1 + 16) = v10;
  *(_DWORD *)(a1 + 24) = v3;
  if ( v11 )
    goto LABEL_30;
LABEL_6:
  v13 = (_QWORD *)(a1 + 16);
  v14 = 4;
LABEL_7:
  for ( i = &v13[v14]; i != v13; v13 += 2 )
  {
    if ( v13 )
      *v13 = -4096;
  }
  for ( j = (__int64 *)v4; v12 != j; j += 2 )
  {
    v26 = *j;
    if ( *j != -4096 && v26 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v17 = a1 + 16;
        v18 = 1;
      }
      else
      {
        v27 = *(_DWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        if ( !v27 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v18 = v27 - 1;
      }
      v19 = 1;
      v20 = 0;
      v21 = v18 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v22 = (_QWORD *)(v17 + 16LL * v21);
      v23 = *v22;
      if ( v26 != *v22 )
      {
        while ( v23 != -4096 )
        {
          if ( !v20 && v23 == -8192 )
            v20 = v22;
          v21 = v18 & (v19 + v21);
          v22 = (_QWORD *)(v17 + 16LL * v21);
          v23 = *v22;
          if ( v26 == *v22 )
            goto LABEL_15;
          ++v19;
        }
        if ( v20 )
          v22 = v20;
      }
LABEL_15:
      *v22 = v26;
      v22[1] = j[1];
      j[1] = 0;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v24 = j[1];
      if ( v24 )
      {
        if ( (v24 & 4) != 0 )
        {
          v25 = (unsigned __int64 *)(v24 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v25 )
          {
            if ( (unsigned __int64 *)*v25 != v25 + 2 )
            {
              v36 = v25;
              _libc_free(*v25);
              v25 = v36;
            }
            j_j___libc_free_0((unsigned __int64)v25);
          }
        }
      }
    }
  }
  sub_C7D6A0(v4, v9, 8);
}
