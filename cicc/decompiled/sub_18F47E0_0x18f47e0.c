// Function: sub_18F47E0
// Address: 0x18f47e0
//
bool __fastcall sub_18F47E0(__int64 a1, __int64 *a2)
{
  char v2; // al
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 *v6; // rax
  const char *v7; // r14
  size_t v8; // rdx
  size_t v9; // r12
  __int64 v10; // r13
  char v11; // al
  int v12; // ebx
  int v13; // ebx
  __int64 v15; // rax
  unsigned int v16; // ecx
  const void *v17; // rsi
  __int64 v18; // rax
  const void *v19; // rsi
  __int64 v20; // rax
  const void *v21; // rsi
  __int64 v22; // rax
  const void *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned int v27; // esi
  int *v28; // rax
  int v29; // edi
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned int v32; // edi
  int *v33; // rax
  int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned int v37; // esi
  int *v38; // rax
  int v39; // edi
  __int64 v40; // rdx
  __int64 v41; // rsi
  unsigned int v42; // ecx
  int *v43; // rax
  int v44; // edi
  int v45; // eax
  int v46; // r9d
  int v47; // eax
  int v48; // r9d
  int v49; // eax
  int v50; // r9d
  int v51; // eax
  int v52; // r9d

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 55 )
    return 1;
  if ( v2 == 78 )
  {
    v15 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v15 + 16) || (*(_BYTE *)(v15 + 33) & 0x20) == 0 )
    {
      v3 = a1 | 4;
      goto LABEL_4;
    }
    v16 = *(_DWORD *)(v15 + 36) - 109;
    if ( v16 <= 0x1D )
      return ((1LL << v16) & 0x3F000081) != 0;
    return 0;
  }
  v3 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  if ( v2 != 29 )
    return 0;
LABEL_4:
  v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v5 = (__int64 *)(v4 - 24);
  v6 = (__int64 *)(v4 - 72);
  if ( (v3 & 4) != 0 )
    v6 = v5;
  if ( *(_BYTE *)(*v6 + 16) )
    return 0;
  v7 = sub_1649960(*v6);
  v9 = v8;
  v10 = *a2;
  v11 = *(_BYTE *)(*a2 + 92) & 3;
  if ( !v11 )
    goto LABEL_9;
  if ( v11 != 3 )
  {
    v25 = *(unsigned int *)(v10 + 136);
    v26 = *(_QWORD *)(v10 + 120);
    if ( (_DWORD)v25 )
    {
      v27 = ((_WORD)v25 - 1) & 0x3530;
      v28 = (int *)(v26 + 40LL * (((_WORD)v25 - 1) & 0x3530));
      v29 = *v28;
      if ( *v28 == 368 )
      {
LABEL_46:
        v17 = (const void *)*((_QWORD *)v28 + 1);
        v18 = *((_QWORD *)v28 + 2);
        goto LABEL_22;
      }
      v49 = 1;
      while ( v29 != -1 )
      {
        v50 = v49 + 1;
        v27 = (v25 - 1) & (v49 + v27);
        v28 = (int *)(v26 + 40LL * v27);
        v29 = *v28;
        if ( *v28 == 368 )
          goto LABEL_46;
        v49 = v50;
      }
    }
    v28 = (int *)(v26 + 40 * v25);
    goto LABEL_46;
  }
  v17 = (const void *)qword_4F9B700[736];
  v18 = qword_4F9B700[737];
LABEL_22:
  if ( v9 == v18 && (!v9 || !memcmp(v7, v17, v9)) )
    return 1;
LABEL_9:
  v12 = *(unsigned __int8 *)(v10 + 93);
  if ( !(v12 >> 6) )
    goto LABEL_10;
  if ( v12 >> 6 != 3 )
  {
    v30 = *(unsigned int *)(v10 + 136);
    v31 = *(_QWORD *)(v10 + 120);
    if ( (_DWORD)v30 )
    {
      v32 = ((_WORD)v30 - 1) & 0x3633;
      v33 = (int *)(v31 + 40LL * (((_WORD)v30 - 1) & 0x3633));
      v34 = *v33;
      if ( *v33 == 375 )
      {
LABEL_49:
        v21 = (const void *)*((_QWORD *)v33 + 1);
        v22 = *((_QWORD *)v33 + 2);
        goto LABEL_34;
      }
      v51 = 1;
      while ( v34 != -1 )
      {
        v52 = v51 + 1;
        v32 = (v30 - 1) & (v51 + v32);
        v33 = (int *)(v31 + 40LL * v32);
        v34 = *v33;
        if ( *v33 == 375 )
          goto LABEL_49;
        v51 = v52;
      }
    }
    v33 = (int *)(v31 + 40 * v30);
    goto LABEL_49;
  }
  v21 = (const void *)qword_4F9B700[750];
  v22 = qword_4F9B700[751];
LABEL_34:
  if ( v9 == v22 && (!v9 || !memcmp(v7, v21, v9)) )
    return 1;
LABEL_10:
  if ( (*(_BYTE *)(v10 + 91) & 3) == 0 )
    goto LABEL_11;
  if ( (*(_BYTE *)(v10 + 91) & 3) != 3 )
  {
    v35 = *(unsigned int *)(v10 + 136);
    v36 = *(_QWORD *)(v10 + 120);
    if ( (_DWORD)v35 )
    {
      v37 = ((_WORD)v35 - 1) & 0x349C;
      v38 = (int *)(v36 + 40LL * (((_WORD)v35 - 1) & 0x349C));
      v39 = *v38;
      if ( *v38 == 364 )
      {
LABEL_52:
        v23 = (const void *)*((_QWORD *)v38 + 1);
        v24 = *((_QWORD *)v38 + 2);
        goto LABEL_40;
      }
      v45 = 1;
      while ( v39 != -1 )
      {
        v46 = v45 + 1;
        v37 = (v35 - 1) & (v45 + v37);
        v38 = (int *)(v36 + 40LL * v37);
        v39 = *v38;
        if ( *v38 == 364 )
          goto LABEL_52;
        v45 = v46;
      }
    }
    v38 = (int *)(v36 + 40 * v35);
    goto LABEL_52;
  }
  v23 = (const void *)qword_4F9B700[728];
  v24 = qword_4F9B700[729];
LABEL_40:
  if ( v9 != v24 || v9 && memcmp(v7, v23, v9) )
  {
LABEL_11:
    v13 = (v12 >> 2) & 3;
    if ( !v13 )
      return 0;
    if ( v13 == 3 )
    {
      v19 = (const void *)qword_4F9B700[746];
      v20 = qword_4F9B700[747];
      return v9 == v20 && (!v9 || !memcmp(v7, v19, v9));
    }
    v40 = *(unsigned int *)(v10 + 136);
    v41 = *(_QWORD *)(v10 + 120);
    if ( (_DWORD)v40 )
    {
      v42 = ((_WORD)v40 - 1) & 0x35E9;
      v43 = (int *)(v41 + 40LL * (((_WORD)v40 - 1) & 0x35E9));
      v44 = *v43;
      if ( *v43 == 373 )
      {
LABEL_55:
        v19 = (const void *)*((_QWORD *)v43 + 1);
        v20 = *((_QWORD *)v43 + 2);
        return v9 == v20 && (!v9 || !memcmp(v7, v19, v9));
      }
      v47 = 1;
      while ( v44 != -1 )
      {
        v48 = v47 + 1;
        v42 = (v40 - 1) & (v47 + v42);
        v43 = (int *)(v41 + 40LL * v42);
        v44 = *v43;
        if ( *v43 == 373 )
          goto LABEL_55;
        v47 = v48;
      }
    }
    v43 = (int *)(v41 + 40 * v40);
    goto LABEL_55;
  }
  return 1;
}
