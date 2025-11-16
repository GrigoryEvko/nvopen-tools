// Function: sub_B86C20
// Address: 0xb86c20
//
_QWORD *__fastcall sub_B86C20(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  char v5; // di
  __int64 v6; // r13
  int v7; // edi
  __int64 v8; // r8
  int v9; // esi
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  unsigned int v14; // esi
  unsigned int v15; // edx
  _QWORD *v16; // r9
  int v17; // eax
  unsigned int v18; // r8d
  __int64 v19; // rdi
  int v20; // r10d
  __int64 v21; // rsi
  int v22; // ecx
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rsi
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // rax
  int v29; // r8d
  _QWORD *v30; // rdi
  int v31; // ecx
  int v32; // ecx
  int v33; // r8d

  (*(void (__fastcall **)(_QWORD *))(*a2 + 152LL))(a2);
  v4 = *(unsigned int *)(a1 + 264);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 268) )
  {
    sub_C8D5F0(a1 + 256, a1 + 272, v4 + 1, 8);
    v4 = *(unsigned int *)(a1 + 264);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v4) = a2;
  v5 = *(_BYTE *)(a1 + 408);
  ++*(_DWORD *)(a1 + 264);
  v6 = a2[2];
  v7 = v5 & 1;
  if ( v7 )
  {
    v8 = a1 + 416;
    v9 = 7;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 424);
    v8 = *(_QWORD *)(a1 + 416);
    if ( !v14 )
    {
      v15 = *(_DWORD *)(a1 + 408);
      ++*(_QWORD *)(a1 + 400);
      v16 = 0;
      v17 = (v15 >> 1) + 1;
LABEL_10:
      v18 = 3 * v14;
      goto LABEL_11;
    }
    v9 = v14 - 1;
  }
  v10 = v9 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v6 == *v11 )
  {
LABEL_6:
    v11[1] = a2;
    return v11 + 1;
  }
  v20 = 1;
  v16 = 0;
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v16 )
      v16 = v11;
    v10 = v9 & (v20 + v10);
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
      goto LABEL_6;
    ++v20;
  }
  v15 = *(_DWORD *)(a1 + 408);
  v18 = 24;
  v14 = 8;
  if ( !v16 )
    v16 = v11;
  ++*(_QWORD *)(a1 + 400);
  v17 = (v15 >> 1) + 1;
  if ( !(_BYTE)v7 )
  {
    v14 = *(_DWORD *)(a1 + 424);
    goto LABEL_10;
  }
LABEL_11:
  v19 = a1 + 400;
  if ( v18 <= 4 * v17 )
  {
    sub_B867E0(v19, 2 * v14);
    if ( (*(_BYTE *)(a1 + 408) & 1) != 0 )
    {
      v21 = a1 + 416;
      v22 = 7;
    }
    else
    {
      v31 = *(_DWORD *)(a1 + 424);
      v21 = *(_QWORD *)(a1 + 416);
      if ( !v31 )
        goto LABEL_54;
      v22 = v31 - 1;
    }
    v23 = v22 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v16 = (_QWORD *)(v21 + 16LL * v23);
    v24 = *v16;
    if ( v6 != *v16 )
    {
      v33 = 1;
      v30 = 0;
      while ( v24 != -4096 )
      {
        if ( !v30 && v24 == -8192 )
          v30 = v16;
        v23 = v22 & (v33 + v23);
        v16 = (_QWORD *)(v21 + 16LL * v23);
        v24 = *v16;
        if ( v6 == *v16 )
          goto LABEL_25;
        ++v33;
      }
      goto LABEL_31;
    }
LABEL_25:
    v15 = *(_DWORD *)(a1 + 408);
    goto LABEL_13;
  }
  if ( v14 - *(_DWORD *)(a1 + 412) - v17 <= v14 >> 3 )
  {
    sub_B867E0(v19, v14);
    if ( (*(_BYTE *)(a1 + 408) & 1) != 0 )
    {
      v25 = a1 + 416;
      v26 = 7;
      goto LABEL_28;
    }
    v32 = *(_DWORD *)(a1 + 424);
    v25 = *(_QWORD *)(a1 + 416);
    if ( v32 )
    {
      v26 = v32 - 1;
LABEL_28:
      v27 = v26 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v16 = (_QWORD *)(v25 + 16LL * v27);
      v28 = *v16;
      if ( v6 != *v16 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v30 )
            v30 = v16;
          v27 = v26 & (v29 + v27);
          v16 = (_QWORD *)(v25 + 16LL * v27);
          v28 = *v16;
          if ( v6 == *v16 )
            goto LABEL_25;
          ++v29;
        }
LABEL_31:
        if ( v30 )
          v16 = v30;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_54:
    *(_DWORD *)(a1 + 408) = (2 * (*(_DWORD *)(a1 + 408) >> 1) + 2) | *(_DWORD *)(a1 + 408) & 1;
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 408) = (2 * (v15 >> 1) + 2) | v15 & 1;
  if ( *v16 != -4096 )
    --*(_DWORD *)(a1 + 412);
  *v16 = v6;
  v16[1] = 0;
  v16[1] = a2;
  return v16 + 1;
}
