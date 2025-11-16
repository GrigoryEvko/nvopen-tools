// Function: sub_279C8E0
// Address: 0x279c8e0
//
__int64 __fastcall sub_279C8E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  char v9; // cl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  __int64 v18; // r14
  int v19; // edx
  unsigned int v20; // edi
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // rcx
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rdi
  int v33; // edx
  int v34; // edx

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 3;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v11 = v16 - 1;
  }
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = v10 + 16LL * v12;
  a5 = *(_QWORD *)v13;
  if ( v8 == *(_QWORD *)v13 )
  {
LABEL_4:
    v14 = *(unsigned int *)(v13 + 8);
    return *(_QWORD *)(a1 + 80) + 16 * v14 + 8;
  }
  a6 = 1;
  v18 = 0;
  while ( a5 != -4096 )
  {
    if ( !v18 && a5 == -8192 )
      v18 = v13;
    v12 = v11 & (a6 + v12);
    v13 = v10 + 16LL * v12;
    a5 = *(_QWORD *)v13;
    if ( v8 == *(_QWORD *)v13 )
      goto LABEL_4;
    a6 = (unsigned int)(a6 + 1);
  }
  if ( !v18 )
    v18 = v13;
  v17 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 12;
  v16 = 4;
LABEL_10:
  if ( v20 <= 4 * v19 )
  {
    sub_BB64D0(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v24 = a1 + 16;
      v25 = 3;
    }
    else
    {
      v33 = *(_DWORD *)(a1 + 24);
      v24 = *(_QWORD *)(a1 + 16);
      if ( !v33 )
        goto LABEL_56;
      v25 = v33 - 1;
    }
    v26 = v25 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v18 = v24 + 16LL * v26;
    v27 = *(_QWORD *)v18;
    if ( v8 != *(_QWORD *)v18 )
    {
      a5 = 1;
      v32 = 0;
      while ( v27 != -4096 )
      {
        if ( !v32 && v27 == -8192 )
          v32 = v18;
        a6 = (unsigned int)(a5 + 1);
        v26 = v25 & (a5 + v26);
        v18 = v24 + 16LL * v26;
        v27 = *(_QWORD *)v18;
        if ( v8 == *(_QWORD *)v18 )
          goto LABEL_26;
        a5 = (unsigned int)a6;
      }
      goto LABEL_32;
    }
LABEL_26:
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 12) - v19 <= v16 >> 3 )
  {
    sub_BB64D0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v28 = a1 + 16;
      v29 = 3;
      goto LABEL_29;
    }
    v34 = *(_DWORD *)(a1 + 24);
    v28 = *(_QWORD *)(a1 + 16);
    if ( v34 )
    {
      v29 = v34 - 1;
LABEL_29:
      v30 = v29 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v18 = v28 + 16LL * v30;
      v31 = *(_QWORD *)v18;
      if ( v8 != *(_QWORD *)v18 )
      {
        a5 = 1;
        v32 = 0;
        while ( v31 != -4096 )
        {
          if ( !v32 && v31 == -8192 )
            v32 = v18;
          a6 = (unsigned int)(a5 + 1);
          v30 = v29 & (a5 + v30);
          v18 = v28 + 16LL * v30;
          v31 = *(_QWORD *)v18;
          if ( v8 == *(_QWORD *)v18 )
            goto LABEL_26;
          a5 = (unsigned int)a6;
        }
LABEL_32:
        if ( v32 )
          v18 = v32;
        goto LABEL_26;
      }
      goto LABEL_26;
    }
LABEL_56:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *(_QWORD *)v18 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v18 = v8;
  *(_DWORD *)(v18 + 8) = 0;
  v21 = *(unsigned int *)(a1 + 88);
  v22 = *a2;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v21 + 1, 0x10u, a5, a6);
    v21 = *(unsigned int *)(a1 + 88);
  }
  v23 = (_QWORD *)(*(_QWORD *)(a1 + 80) + 16 * v21);
  *v23 = v22;
  v23[1] = 0;
  v14 = *(unsigned int *)(a1 + 88);
  *(_DWORD *)(a1 + 88) = v14 + 1;
  *(_DWORD *)(v18 + 8) = v14;
  return *(_QWORD *)(a1 + 80) + 16 * v14 + 8;
}
