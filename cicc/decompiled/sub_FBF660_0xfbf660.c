// Function: sub_FBF660
// Address: 0xfbf660
//
_QWORD *__fastcall sub_FBF660(__int64 a1, __int64 *a2)
{
  int v4; // edi
  __int64 v5; // r9
  int v6; // esi
  unsigned int v7; // eax
  _QWORD *v8; // rcx
  __int64 v9; // r10
  unsigned int v11; // esi
  unsigned int v12; // eax
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 v17; // rcx
  __int64 v18; // rax
  _QWORD *v19; // r8
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  int v22; // r11d
  __int64 v23; // r8
  int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // r8
  int v28; // esi
  unsigned int v29; // eax
  __int64 v30; // rcx
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // ecx
  int v34; // esi
  int v35; // r10d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v13 = 0;
      v14 = (v12 >> 1) + 1;
LABEL_8:
      v15 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 248LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  v22 = 1;
  v13 = 0;
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v13 )
      v13 = v8;
    v7 = v6 & (v22 + v7);
    v8 = (_QWORD *)(v5 + 248LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v22;
  }
  v12 = *(_DWORD *)(a1 + 8);
  v15 = 12;
  v11 = 4;
  if ( !v13 )
    v13 = v8;
  ++*(_QWORD *)a1;
  v14 = (v12 >> 1) + 1;
  if ( !(_BYTE)v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * (int)v14 >= (unsigned int)v15 )
  {
    sub_FBF310(a1, 2 * v11, (unsigned __int64)v13, v14, v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v23 = a1 + 16;
      v24 = 3;
    }
    else
    {
      v33 = *(_DWORD *)(a1 + 24);
      v23 = *(_QWORD *)(a1 + 16);
      if ( !v33 )
        goto LABEL_56;
      v24 = v33 - 1;
    }
    v25 = v24 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v13 = (_QWORD *)(v23 + 248LL * v25);
    v26 = *v13;
    if ( *v13 != *a2 )
    {
      v35 = 1;
      v32 = 0;
      while ( v26 != -4096 )
      {
        if ( !v32 && v26 == -8192 )
          v32 = v13;
        v25 = v24 & (v35 + v25);
        v13 = (_QWORD *)(v23 + 248LL * v25);
        v26 = *v13;
        if ( *a2 == *v13 )
          goto LABEL_27;
        ++v35;
      }
      goto LABEL_33;
    }
LABEL_27:
    v12 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  v16 = v11 - *(_DWORD *)(a1 + 12) - v14;
  v17 = v11 >> 3;
  if ( v16 <= (unsigned int)v17 )
  {
    sub_FBF310(a1, v11, (unsigned __int64)v13, v17, v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v27 = a1 + 16;
      v28 = 3;
      goto LABEL_30;
    }
    v34 = *(_DWORD *)(a1 + 24);
    v27 = *(_QWORD *)(a1 + 16);
    if ( v34 )
    {
      v28 = v34 - 1;
LABEL_30:
      v29 = v28 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v13 = (_QWORD *)(v27 + 248LL * v29);
      v30 = *v13;
      if ( *v13 != *a2 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v32 )
            v32 = v13;
          v29 = v28 & (v31 + v29);
          v13 = (_QWORD *)(v27 + 248LL * v29);
          v30 = *v13;
          if ( *a2 == *v13 )
            goto LABEL_27;
          ++v31;
        }
LABEL_33:
        if ( v32 )
          v13 = v32;
        goto LABEL_27;
      }
      goto LABEL_27;
    }
LABEL_56:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v18 = *a2;
  v19 = v13 + 1;
  v13[1] = 0;
  v13[2] = 1;
  *v13 = v18;
  v20 = v13 + 3;
  v21 = v13 + 31;
  do
  {
    if ( v20 )
      *v20 = -4096;
    v20 += 7;
  }
  while ( v20 != v21 );
  return v19;
}
