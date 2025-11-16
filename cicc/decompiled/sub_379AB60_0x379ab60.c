// Function: sub_379AB60
// Address: 0x379ab60
//
__int64 __fastcall sub_379AB60(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v4; // r12d
  char v5; // r8
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // edx
  _DWORD *v9; // rax
  int v10; // r9d
  int *v11; // r12
  __int64 v12; // r8
  int v13; // ecx
  unsigned int v14; // edi
  __int64 v15; // rax
  int v16; // r9d
  unsigned int v18; // esi
  __int64 v19; // rax
  unsigned int v20; // eax
  _DWORD *v21; // rcx
  int v22; // edx
  unsigned int v23; // r9d
  __int64 v24; // rax
  int v25; // eax
  int v26; // r10d
  __int64 v27; // rsi
  int v28; // eax
  unsigned int v29; // edx
  int v30; // edi
  __int64 v31; // rsi
  int v32; // edx
  unsigned int v33; // eax
  int v34; // edi
  int v35; // r9d
  _DWORD *v36; // r8
  int v37; // eax
  int v38; // edx
  int v39; // r10d
  int v40; // r9d

  v4 = sub_375D5B0(a1, a2, a3);
  v5 = *(_BYTE *)(a1 + 1456) & 1;
  if ( v5 )
  {
    v6 = a1 + 1464;
    v7 = 7;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 1472);
    v6 = *(_QWORD *)(a1 + 1464);
    if ( !v18 )
    {
      v20 = *(_DWORD *)(a1 + 1456);
      ++*(_QWORD *)(a1 + 1448);
      v21 = 0;
      v22 = (v20 >> 1) + 1;
LABEL_14:
      v23 = 3 * v18;
      goto LABEL_15;
    }
    v7 = v18 - 1;
  }
  v8 = v7 & (37 * v4);
  v9 = (_DWORD *)(v6 + 8LL * v8);
  v10 = *v9;
  if ( v4 == *v9 )
  {
LABEL_4:
    v11 = v9 + 1;
    goto LABEL_5;
  }
  v26 = 1;
  v21 = 0;
  while ( v10 != -1 )
  {
    if ( !v21 && v10 == -2 )
      v21 = v9;
    v8 = v7 & (v26 + v8);
    v9 = (_DWORD *)(v6 + 8LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
      goto LABEL_4;
    ++v26;
  }
  v23 = 24;
  v18 = 8;
  if ( !v21 )
    v21 = v9;
  v20 = *(_DWORD *)(a1 + 1456);
  ++*(_QWORD *)(a1 + 1448);
  v22 = (v20 >> 1) + 1;
  if ( !v5 )
  {
    v18 = *(_DWORD *)(a1 + 1472);
    goto LABEL_14;
  }
LABEL_15:
  if ( v23 <= 4 * v22 )
  {
    sub_375BDE0(a1 + 1448, 2 * v18);
    if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
    {
      v27 = a1 + 1464;
      v28 = 7;
    }
    else
    {
      v37 = *(_DWORD *)(a1 + 1472);
      v27 = *(_QWORD *)(a1 + 1464);
      if ( !v37 )
        goto LABEL_67;
      v28 = v37 - 1;
    }
    v29 = v28 & (37 * v4);
    v21 = (_DWORD *)(v27 + 8LL * v29);
    v30 = *v21;
    if ( v4 != *v21 )
    {
      v40 = 1;
      v36 = 0;
      while ( v30 != -1 )
      {
        if ( v30 == -2 && !v36 )
          v36 = v21;
        v29 = v28 & (v40 + v29);
        v21 = (_DWORD *)(v27 + 8LL * v29);
        v30 = *v21;
        if ( v4 == *v21 )
          goto LABEL_36;
        ++v40;
      }
      goto LABEL_42;
    }
LABEL_36:
    v20 = *(_DWORD *)(a1 + 1456);
    goto LABEL_17;
  }
  if ( v18 - *(_DWORD *)(a1 + 1460) - v22 <= v18 >> 3 )
  {
    sub_375BDE0(a1 + 1448, v18);
    if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
    {
      v31 = a1 + 1464;
      v32 = 7;
      goto LABEL_39;
    }
    v38 = *(_DWORD *)(a1 + 1472);
    v31 = *(_QWORD *)(a1 + 1464);
    if ( v38 )
    {
      v32 = v38 - 1;
LABEL_39:
      v33 = v32 & (37 * v4);
      v21 = (_DWORD *)(v31 + 8LL * v33);
      v34 = *v21;
      if ( v4 != *v21 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -1 )
        {
          if ( !v36 && v34 == -2 )
            v36 = v21;
          v33 = v32 & (v35 + v33);
          v21 = (_DWORD *)(v31 + 8LL * v33);
          v34 = *v21;
          if ( v4 == *v21 )
            goto LABEL_36;
          ++v35;
        }
LABEL_42:
        if ( v36 )
          v21 = v36;
        goto LABEL_36;
      }
      goto LABEL_36;
    }
LABEL_67:
    *(_DWORD *)(a1 + 1456) = (2 * (*(_DWORD *)(a1 + 1456) >> 1) + 2) | *(_DWORD *)(a1 + 1456) & 1;
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 1456) = (2 * (v20 >> 1) + 2) | v20 & 1;
  if ( *v21 != -1 )
    --*(_DWORD *)(a1 + 1460);
  *v21 = v4;
  v11 = v21 + 1;
  v21[1] = 0;
LABEL_5:
  sub_37593F0(a1, v11);
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v12 = a1 + 520;
    v13 = 7;
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 528);
    v12 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v19 )
      goto LABEL_21;
    v13 = v19 - 1;
  }
  v14 = v13 & (37 * *v11);
  v15 = v12 + 24LL * v14;
  v16 = *(_DWORD *)v15;
  if ( *v11 == *(_DWORD *)v15 )
    return *(_QWORD *)(v15 + 8);
  v25 = 1;
  while ( v16 != -1 )
  {
    v39 = v25 + 1;
    v14 = v13 & (v25 + v14);
    v15 = v12 + 24LL * v14;
    v16 = *(_DWORD *)v15;
    if ( *v11 == *(_DWORD *)v15 )
      return *(_QWORD *)(v15 + 8);
    v25 = v39;
  }
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v24 = 192;
    return *(_QWORD *)(v12 + v24 + 8);
  }
  v19 = *(unsigned int *)(a1 + 528);
LABEL_21:
  v24 = 24 * v19;
  return *(_QWORD *)(v12 + v24 + 8);
}
