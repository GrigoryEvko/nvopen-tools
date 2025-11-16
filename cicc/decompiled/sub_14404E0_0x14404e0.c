// Function: sub_14404E0
// Address: 0x14404e0
//
int *__fastcall sub_14404E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  unsigned int v5; // esi
  unsigned int v6; // ecx
  __int64 v7; // rdi
  unsigned int v8; // r15d
  unsigned int v9; // edx
  __int64 **v10; // r12
  __int64 *v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r8
  int v14; // ecx
  __int64 v15; // r9
  unsigned int v16; // edx
  int *v17; // rax
  int v18; // edi
  __int64 *v20; // r8
  __int64 v21; // r11
  int i; // r9d
  int v23; // eax
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // edx
  int v27; // eax
  __int64 *v28; // rsi
  int v29; // r9d
  __int64 **v30; // r8
  __int64 **v31; // r8
  int v32; // r9d
  int v33; // eax
  int v34; // eax
  int v35; // esi
  int v36; // edx
  __int64 v37; // r9
  int v38; // ecx
  unsigned int v39; // edi
  int v40; // r8d
  int v41; // r11d
  int *v42; // r10
  int v43; // r13d
  int *v44; // r11
  int v45; // ecx
  int v46; // edx
  int v47; // eax
  int v48; // esi
  int v49; // edx
  __int64 v50; // r9
  int v51; // r11d
  unsigned int v52; // edi
  int v53; // r8d
  int v54; // eax
  int v55; // edx
  __int64 v56; // rsi
  int v57; // r8d
  unsigned int v58; // r15d
  __int64 **v59; // rdi
  __int64 *v60; // rcx
  unsigned __int64 v61[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v62[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a1 + 8;
  v5 = *(_DWORD *)(a1 + 32);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = *(_QWORD *)(a1 + 16);
    v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v9 = (v5 - 1) & v8;
    v10 = (__int64 **)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    v20 = *v10;
    LODWORD(v21) = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    for ( i = 1; ; ++i )
    {
      if ( v20 == (__int64 *)-8LL )
        goto LABEL_8;
      v21 = v6 & ((_DWORD)v21 + i);
      v20 = *(__int64 **)(v7 + 16 * v21);
      if ( v20 == a2 )
        break;
    }
    v31 = (__int64 **)(v7 + 16LL * (v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  }
  else
  {
LABEL_8:
    v61[0] = (unsigned __int64)v62;
    v61[1] = 0x800000000LL;
    sub_143E570(a1, a2, (__int64)v61);
    if ( (_BYTE *)v61[0] != v62 )
      _libc_free(v61[0]);
    v5 = *(_DWORD *)(a1 + 32);
    if ( !v5 )
    {
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_12;
    }
    v6 = v5 - 1;
    v7 = *(_QWORD *)(a1 + 16);
    v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v9 = v8 & (v5 - 1);
    v31 = (__int64 **)(v7 + 16LL * v9);
    v11 = *v31;
    if ( a2 == *v31 )
    {
      v10 = (__int64 **)(v7 + 16LL * (v8 & (v5 - 1)));
LABEL_3:
      v12 = *(_DWORD *)(a1 + 64);
      v13 = a1 + 40;
      if ( v12 )
        goto LABEL_4;
LABEL_29:
      ++*(_QWORD *)(a1 + 40);
      goto LABEL_30;
    }
  }
  v32 = 1;
  v10 = 0;
  while ( v11 != (__int64 *)-8LL )
  {
    if ( v11 != (__int64 *)-16LL || v10 )
      v31 = v10;
    v9 = v6 & (v32 + v9);
    v10 = (__int64 **)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v32;
    v10 = v31;
    v31 = (__int64 **)(v7 + 16LL * v9);
  }
  v33 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
    v10 = v31;
  ++*(_QWORD *)(a1 + 8);
  v27 = v33 + 1;
  if ( 4 * v27 >= 3 * v5 )
  {
LABEL_12:
    sub_143D960(v2, 2 * v5);
    v23 = *(_DWORD *)(a1 + 32);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 16);
      v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(a1 + 24) + 1;
      v10 = (__int64 **)(v25 + 16LL * v26);
      v28 = *v10;
      if ( *v10 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != (__int64 *)-8LL )
        {
          if ( v28 == (__int64 *)-16LL && !v30 )
            v30 = v10;
          v26 = v24 & (v29 + v26);
          v10 = (__int64 **)(v25 + 16LL * v26);
          v28 = *v10;
          if ( *v10 == a2 )
            goto LABEL_26;
          ++v29;
        }
        if ( v30 )
          v10 = v30;
      }
      goto LABEL_26;
    }
LABEL_90:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  if ( v5 - (v27 + *(_DWORD *)(a1 + 28)) > v5 >> 3 )
    goto LABEL_26;
  sub_143D960(v2, v5);
  v54 = *(_DWORD *)(a1 + 32);
  if ( !v54 )
    goto LABEL_90;
  v55 = v54 - 1;
  v56 = *(_QWORD *)(a1 + 16);
  v57 = 1;
  v58 = (v54 - 1) & v8;
  v59 = 0;
  v27 = *(_DWORD *)(a1 + 24) + 1;
  v10 = (__int64 **)(v56 + 16LL * v58);
  v60 = *v10;
  if ( *v10 != a2 )
  {
    while ( v60 != (__int64 *)-8LL )
    {
      if ( v60 == (__int64 *)-16LL && !v59 )
        v59 = v10;
      v58 = v55 & (v57 + v58);
      v10 = (__int64 **)(v56 + 16LL * v58);
      v60 = *v10;
      if ( *v10 == a2 )
        goto LABEL_26;
      ++v57;
    }
    if ( v59 )
      v10 = v59;
  }
LABEL_26:
  *(_DWORD *)(a1 + 24) = v27;
  if ( *v10 != (__int64 *)-8LL )
    --*(_DWORD *)(a1 + 28);
  *v10 = a2;
  v13 = a1 + 40;
  *((_DWORD *)v10 + 2) = 0;
  v12 = *(_DWORD *)(a1 + 64);
  if ( !v12 )
    goto LABEL_29;
LABEL_4:
  v14 = *((_DWORD *)v10 + 2);
  v15 = *(_QWORD *)(a1 + 48);
  v16 = (v12 - 1) & (37 * v14);
  v17 = (int *)(v15 + 80LL * v16);
  v18 = *v17;
  if ( *v17 == v14 )
    return v17 + 2;
  v43 = 1;
  v44 = 0;
  while ( v18 != -1 )
  {
    if ( v18 == -2 && !v44 )
      v44 = v17;
    v16 = (v12 - 1) & (v43 + v16);
    v17 = (int *)(v15 + 80LL * v16);
    v18 = *v17;
    if ( v14 == *v17 )
      return v17 + 2;
    ++v43;
  }
  v45 = *(_DWORD *)(a1 + 56);
  if ( v44 )
    v17 = v44;
  ++*(_QWORD *)(a1 + 40);
  v38 = v45 + 1;
  if ( 4 * v38 < 3 * v12 )
  {
    if ( v12 - *(_DWORD *)(a1 + 60) - v38 > v12 >> 3 )
      goto LABEL_43;
    sub_143E380(v13, v12);
    v47 = *(_DWORD *)(a1 + 64);
    if ( v47 )
    {
      v48 = *((_DWORD *)v10 + 2);
      v49 = v47 - 1;
      v50 = *(_QWORD *)(a1 + 48);
      v42 = 0;
      v51 = 1;
      v38 = *(_DWORD *)(a1 + 56) + 1;
      v52 = (v47 - 1) & (37 * v48);
      v17 = (int *)(v50 + 80LL * v52);
      v53 = *v17;
      if ( *v17 == v48 )
        goto LABEL_43;
      while ( v53 != -1 )
      {
        if ( !v42 && v53 == -2 )
          v42 = v17;
        v52 = v49 & (v51 + v52);
        v17 = (int *)(v50 + 80LL * v52);
        v53 = *v17;
        if ( v48 == *v17 )
          goto LABEL_43;
        ++v51;
      }
LABEL_34:
      if ( v42 )
        v17 = v42;
      goto LABEL_43;
    }
LABEL_89:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_30:
  sub_143E380(v13, 2 * v12);
  v34 = *(_DWORD *)(a1 + 64);
  if ( !v34 )
    goto LABEL_89;
  v35 = *((_DWORD *)v10 + 2);
  v36 = v34 - 1;
  v37 = *(_QWORD *)(a1 + 48);
  v38 = *(_DWORD *)(a1 + 56) + 1;
  v39 = (v34 - 1) & (37 * v35);
  v17 = (int *)(v37 + 80LL * v39);
  v40 = *v17;
  if ( *v17 != v35 )
  {
    v41 = 1;
    v42 = 0;
    while ( v40 != -1 )
    {
      if ( !v42 && v40 == -2 )
        v42 = v17;
      v39 = v36 & (v41 + v39);
      v17 = (int *)(v37 + 80LL * v39);
      v40 = *v17;
      if ( v35 == *v17 )
        goto LABEL_43;
      ++v41;
    }
    goto LABEL_34;
  }
LABEL_43:
  *(_DWORD *)(a1 + 56) = v38;
  if ( *v17 != -1 )
    --*(_DWORD *)(a1 + 60);
  v46 = *((_DWORD *)v10 + 2);
  *((_QWORD *)v17 + 1) = 0;
  *((_QWORD *)v17 + 4) = 4;
  *v17 = v46;
  *((_QWORD *)v17 + 2) = v17 + 12;
  *((_QWORD *)v17 + 3) = v17 + 12;
  v17[10] = 0;
  return v17 + 2;
}
