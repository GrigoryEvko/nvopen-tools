// Function: sub_11702B0
// Address: 0x11702b0
//
__int64 __fastcall sub_11702B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  int v7; // edi
  __int64 v8; // r9
  int v9; // esi
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // r12
  char v15; // di
  __int64 v16; // r8
  int v17; // esi
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // r9
  _DWORD *v21; // rdx
  __int64 result; // rax
  unsigned int v23; // esi
  unsigned int v24; // esi
  unsigned int v25; // eax
  _QWORD *v26; // rdx
  int v27; // ecx
  unsigned int v28; // r8d
  _DWORD *v29; // rdx
  unsigned int v30; // edx
  _QWORD *v31; // r8
  int v32; // eax
  unsigned int v33; // r9d
  int v34; // r10d
  int v35; // r11d
  __int64 v36; // rsi
  int v37; // ecx
  unsigned int v38; // eax
  __int64 v39; // rdi
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rsi
  int v45; // ecx
  unsigned int v46; // edx
  __int64 v47; // rax
  int v48; // r9d
  _QWORD *v49; // rdi
  __int64 v50; // rsi
  int v51; // ecx
  unsigned int v52; // eax
  __int64 v53; // rdi
  int v54; // r9d
  _QWORD *v55; // r8
  int v56; // ecx
  int v57; // ecx
  int v58; // ecx
  int v59; // ecx
  int v60; // r9d
  int v61; // r9d

  v5 = *a1;
  v7 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v7 )
  {
    v8 = v5 + 16;
    v9 = 7;
  }
  else
  {
    v23 = *(_DWORD *)(v5 + 24);
    v8 = *(_QWORD *)(v5 + 16);
    if ( !v23 )
    {
      v30 = *(_DWORD *)(v5 + 8);
      ++*(_QWORD *)v5;
      v31 = 0;
      v32 = (v30 >> 1) + 1;
LABEL_21:
      v33 = 3 * v23;
      goto LABEL_22;
    }
    v9 = v23 - 1;
  }
  v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( a2 == *v11 )
  {
LABEL_4:
    v13 = v11 + 1;
    goto LABEL_5;
  }
  v35 = 1;
  v31 = 0;
  while ( v12 != -4096 )
  {
    if ( !v31 && v12 == -8192 )
      v31 = v11;
    v10 = v9 & (v35 + v10);
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      goto LABEL_4;
    ++v35;
  }
  v30 = *(_DWORD *)(v5 + 8);
  v33 = 24;
  v23 = 8;
  if ( !v31 )
    v31 = v11;
  ++*(_QWORD *)v5;
  v32 = (v30 >> 1) + 1;
  if ( !(_BYTE)v7 )
  {
    v23 = *(_DWORD *)(v5 + 24);
    goto LABEL_21;
  }
LABEL_22:
  if ( 4 * v32 >= v33 )
  {
    sub_116FE70(v5, 2 * v23);
    if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
    {
      v40 = v5 + 16;
      v41 = 7;
    }
    else
    {
      v56 = *(_DWORD *)(v5 + 24);
      v40 = *(_QWORD *)(v5 + 16);
      if ( !v56 )
        goto LABEL_103;
      v41 = v56 - 1;
    }
    v42 = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v31 = (_QWORD *)(v40 + 16LL * v42);
    v43 = *v31;
    if ( *v31 != a2 )
    {
      v60 = 1;
      v49 = 0;
      while ( v43 != -4096 )
      {
        if ( !v49 && v43 == -8192 )
          v49 = v31;
        v42 = v41 & (v60 + v42);
        v31 = (_QWORD *)(v40 + 16LL * v42);
        v43 = *v31;
        if ( *v31 == a2 )
          goto LABEL_46;
        ++v60;
      }
      goto LABEL_52;
    }
LABEL_46:
    v30 = *(_DWORD *)(v5 + 8);
    goto LABEL_24;
  }
  if ( v23 - *(_DWORD *)(v5 + 12) - v32 <= v23 >> 3 )
  {
    sub_116FE70(v5, v23);
    if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
    {
      v44 = v5 + 16;
      v45 = 7;
      goto LABEL_49;
    }
    v59 = *(_DWORD *)(v5 + 24);
    v44 = *(_QWORD *)(v5 + 16);
    if ( v59 )
    {
      v45 = v59 - 1;
LABEL_49:
      v46 = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = (_QWORD *)(v44 + 16LL * v46);
      v47 = *v31;
      if ( *v31 != a2 )
      {
        v48 = 1;
        v49 = 0;
        while ( v47 != -4096 )
        {
          if ( v47 == -8192 && !v49 )
            v49 = v31;
          v46 = v45 & (v48 + v46);
          v31 = (_QWORD *)(v44 + 16LL * v46);
          v47 = *v31;
          if ( *v31 == a2 )
            goto LABEL_46;
          ++v48;
        }
LABEL_52:
        if ( v49 )
          v31 = v49;
        goto LABEL_46;
      }
      goto LABEL_46;
    }
LABEL_103:
    *(_DWORD *)(v5 + 8) = (2 * (*(_DWORD *)(v5 + 8) >> 1) + 2) | *(_DWORD *)(v5 + 8) & 1;
    BUG();
  }
LABEL_24:
  *(_DWORD *)(v5 + 8) = (2 * (v30 >> 1) + 2) | v30 & 1;
  if ( *v31 != -4096 )
    --*(_DWORD *)(v5 + 12);
  *v31 = a2;
  v13 = v31 + 1;
  v31[1] = 0;
LABEL_5:
  *v13 = a3;
  v14 = a1[1];
  v15 = *(_BYTE *)(v14 + 8) & 1;
  if ( v15 )
  {
    v16 = v14 + 16;
    v17 = 7;
  }
  else
  {
    v24 = *(_DWORD *)(v14 + 24);
    v16 = *(_QWORD *)(v14 + 16);
    if ( !v24 )
    {
      v25 = *(_DWORD *)(v14 + 8);
      ++*(_QWORD *)v14;
      v26 = 0;
      v27 = (v25 >> 1) + 1;
LABEL_14:
      v28 = 3 * v24;
      goto LABEL_15;
    }
    v17 = v24 - 1;
  }
  v18 = v17 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = v16 + 16LL * v18;
  v20 = *(_QWORD *)v19;
  if ( *(_QWORD *)v19 == a3 )
  {
LABEL_8:
    v21 = (_DWORD *)(v19 + 8);
    result = (unsigned int)(*(_DWORD *)(v19 + 8) + 1);
    *v21 = result;
    return result;
  }
  v34 = 1;
  v26 = 0;
  while ( v20 != -4096 )
  {
    if ( !v26 && v20 == -8192 )
      v26 = (_QWORD *)v19;
    v18 = v17 & (v34 + v18);
    v19 = v16 + 16LL * v18;
    v20 = *(_QWORD *)v19;
    if ( *(_QWORD *)v19 == a3 )
      goto LABEL_8;
    ++v34;
  }
  v28 = 24;
  v24 = 8;
  if ( !v26 )
    v26 = (_QWORD *)v19;
  v25 = *(_DWORD *)(v14 + 8);
  ++*(_QWORD *)v14;
  v27 = (v25 >> 1) + 1;
  if ( !v15 )
  {
    v24 = *(_DWORD *)(v14 + 24);
    goto LABEL_14;
  }
LABEL_15:
  if ( 4 * v27 >= v28 )
  {
    sub_FB9E50(v14, 2 * v24);
    if ( (*(_BYTE *)(v14 + 8) & 1) != 0 )
    {
      v36 = v14 + 16;
      v37 = 7;
    }
    else
    {
      v57 = *(_DWORD *)(v14 + 24);
      v36 = *(_QWORD *)(v14 + 16);
      if ( !v57 )
        goto LABEL_104;
      v37 = v57 - 1;
    }
    v38 = v37 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v26 = (_QWORD *)(v36 + 16LL * v38);
    v39 = *v26;
    if ( *v26 != a3 )
    {
      v61 = 1;
      v55 = 0;
      while ( v39 != -4096 )
      {
        if ( !v55 && v39 == -8192 )
          v55 = v26;
        v38 = v37 & (v61 + v38);
        v26 = (_QWORD *)(v36 + 16LL * v38);
        v39 = *v26;
        if ( *v26 == a3 )
          goto LABEL_42;
        ++v61;
      }
      goto LABEL_59;
    }
LABEL_42:
    v25 = *(_DWORD *)(v14 + 8);
    goto LABEL_17;
  }
  if ( v24 - *(_DWORD *)(v14 + 12) - v27 <= v24 >> 3 )
  {
    sub_FB9E50(v14, v24);
    if ( (*(_BYTE *)(v14 + 8) & 1) != 0 )
    {
      v50 = v14 + 16;
      v51 = 7;
      goto LABEL_56;
    }
    v58 = *(_DWORD *)(v14 + 24);
    v50 = *(_QWORD *)(v14 + 16);
    if ( v58 )
    {
      v51 = v58 - 1;
LABEL_56:
      v52 = v51 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v26 = (_QWORD *)(v50 + 16LL * v52);
      v53 = *v26;
      if ( *v26 != a3 )
      {
        v54 = 1;
        v55 = 0;
        while ( v53 != -4096 )
        {
          if ( !v55 && v53 == -8192 )
            v55 = v26;
          v52 = v51 & (v54 + v52);
          v26 = (_QWORD *)(v50 + 16LL * v52);
          v53 = *v26;
          if ( *v26 == a3 )
            goto LABEL_42;
          ++v54;
        }
LABEL_59:
        if ( v55 )
          v26 = v55;
        goto LABEL_42;
      }
      goto LABEL_42;
    }
LABEL_104:
    *(_DWORD *)(v14 + 8) = (2 * (*(_DWORD *)(v14 + 8) >> 1) + 2) | *(_DWORD *)(v14 + 8) & 1;
    BUG();
  }
LABEL_17:
  *(_DWORD *)(v14 + 8) = (2 * (v25 >> 1) + 2) | v25 & 1;
  if ( *v26 != -4096 )
    --*(_DWORD *)(v14 + 12);
  *v26 = a3;
  v29 = v26 + 1;
  *v29 = 0;
  *v29 = 1;
  return 1;
}
