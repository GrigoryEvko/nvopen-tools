// Function: sub_1D0CB80
// Address: 0x1d0cb80
//
void __fastcall sub_1D0CB80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rbx
  int v11; // r9d
  __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  _DWORD *v14; // rsi
  _DWORD *v15; // rdx
  _DWORD *v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r13
  unsigned int *v20; // r14
  unsigned int v21; // edi
  __int64 i; // r15
  unsigned int v23; // ecx
  __int64 v24; // rax
  _BOOL4 v25; // ebx
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r12
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 v31; // rax
  _BOOL4 v32; // r11d
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rcx
  unsigned __int64 v36; // rax
  __int64 j; // rax
  int v38; // eax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rax
  __int64 k; // rax
  int v43; // eax
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // ecx
  __int64 v48; // rax
  _BOOL4 v49; // r12d
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 m; // r12
  __int64 v53; // rax
  __int64 v54; // r9
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+20h] [rbp-50h]
  __int64 v65; // [rsp+20h] [rbp-50h]
  __int64 v66; // [rsp+20h] [rbp-50h]
  __int64 v67; // [rsp+20h] [rbp-50h]
  unsigned int v68; // [rsp+28h] [rbp-48h]
  _BOOL4 v69; // [rsp+28h] [rbp-48h]
  __int64 v70; // [rsp+28h] [rbp-48h]
  __int64 v71; // [rsp+28h] [rbp-48h]
  __int64 v72; // [rsp+28h] [rbp-48h]
  __int64 v73; // [rsp+30h] [rbp-40h]
  __int64 v74; // [rsp+30h] [rbp-40h]
  __int64 v75; // [rsp+30h] [rbp-40h]
  __int64 v76; // [rsp+30h] [rbp-40h]
  __int64 v77; // [rsp+30h] [rbp-40h]

  v8 = a5;
  v9 = a1;
  v10 = *(unsigned int *)(a1 + 64);
  if ( !(_DWORD)v10 )
    goto LABEL_2;
  if ( *(_QWORD *)(a6 + 88) )
  {
    v28 = *(_QWORD *)(a6 + 64);
    v29 = a6 + 56;
    if ( !v28 )
    {
      v28 = a6 + 56;
      if ( v29 == *(_QWORD *)(a6 + 72) )
      {
        v32 = 1;
LABEL_37:
        v60 = a6;
        v64 = a4;
        v69 = v32;
        v74 = v29;
        v33 = sub_22077B0(40);
        *(_DWORD *)(v33 + 32) = v10;
        sub_220F040(v69, v33, v28, v74);
        LODWORD(a6) = v60;
        a4 = v64;
        ++*(_QWORD *)(v60 + 88);
        goto LABEL_38;
      }
      goto LABEL_73;
    }
    while ( 1 )
    {
      v30 = *(_DWORD *)(v28 + 32);
      v31 = *(_QWORD *)(v28 + 24);
      if ( (unsigned int)v10 < v30 )
        v31 = *(_QWORD *)(v28 + 16);
      if ( !v31 )
        break;
      v28 = v31;
    }
    if ( (unsigned int)v10 < v30 )
    {
      if ( *(_QWORD *)(a6 + 72) != v28 )
      {
LABEL_73:
        v75 = a4;
        v66 = a6;
        v71 = a6 + 56;
        v51 = sub_220EF80(v28);
        a4 = v75;
        if ( (unsigned int)v10 <= *(_DWORD *)(v51 + 32) )
          goto LABEL_2;
        v29 = v71;
        a6 = v66;
        if ( !v28 )
          goto LABEL_2;
      }
    }
    else if ( (unsigned int)v10 <= v30 )
    {
      goto LABEL_2;
    }
    v32 = 1;
    if ( v29 != v28 )
      v32 = (unsigned int)v10 < *(_DWORD *)(v28 + 32);
    goto LABEL_37;
  }
  v13 = *(unsigned int *)(a6 + 8);
  v14 = *(_DWORD **)a6;
  v15 = (_DWORD *)(*(_QWORD *)a6 + 4 * v13);
  if ( *(_DWORD **)a6 != v15 )
  {
    v16 = *(_DWORD **)a6;
    while ( (_DWORD)v10 != *v16 )
    {
      if ( v15 == ++v16 )
        goto LABEL_12;
    }
    if ( v15 != v16 )
    {
LABEL_2:
      if ( (*(_BYTE *)(a1 + 26) & 1) != 0 )
      {
        v11 = 0;
        v12 = *(_QWORD *)(a2 + 648);
        goto LABEL_5;
      }
      return;
    }
  }
LABEL_12:
  if ( v13 <= 7 )
  {
    if ( *(_DWORD *)(a6 + 8) >= *(_DWORD *)(a6 + 12) )
    {
      v72 = a4;
      v76 = a6;
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 4, a5, a6);
      a6 = v76;
      a4 = v72;
      v15 = (_DWORD *)(*(_QWORD *)v76 + 4LL * *(unsigned int *)(v76 + 8));
    }
    *v15 = v10;
    ++*(_DWORD *)(a6 + 8);
    goto LABEL_38;
  }
  v17 = a6 + 56;
  v59 = a3;
  v73 = a6 + 56;
  v58 = a4;
  v68 = *(_DWORD *)(a1 + 64);
  v18 = *(_QWORD *)(a6 + 64);
  v19 = a6;
  while ( 1 )
  {
    v20 = &v14[v13 - 1];
    if ( v18 )
    {
      v21 = *v20;
      for ( i = v18; ; i = v24 )
      {
        v23 = *(_DWORD *)(i + 32);
        v24 = *(_QWORD *)(i + 24);
        if ( v21 < v23 )
          v24 = *(_QWORD *)(i + 16);
        if ( !v24 )
          break;
      }
      if ( v21 >= v23 )
      {
        if ( v21 <= v23 )
          goto LABEL_25;
LABEL_22:
        v25 = 1;
        if ( v17 != i )
          v25 = *v20 < *(_DWORD *)(i + 32);
LABEL_24:
        v26 = sub_22077B0(40);
        *(_DWORD *)(v26 + 32) = *v20;
        sub_220F040(v25, v26, i, v17);
        ++*(_QWORD *)(v19 + 88);
        v18 = *(_QWORD *)(v19 + 64);
        goto LABEL_25;
      }
      if ( i == *(_QWORD *)(v19 + 72) )
        goto LABEL_22;
    }
    else
    {
      i = v17;
      if ( v17 == *(_QWORD *)(v19 + 72) )
      {
        v25 = 1;
        goto LABEL_24;
      }
    }
    if ( *(_DWORD *)(sub_220EF80(i) + 32) < *v20 )
      goto LABEL_22;
LABEL_25:
    v27 = *(_DWORD *)(v19 + 8) - 1;
    *(_DWORD *)(v19 + 8) = v27;
    if ( !v27 )
      break;
    v14 = *(_DWORD **)v19;
    v13 = v27;
  }
  v46 = v18;
  a6 = v19;
  v10 = v68;
  a3 = v59;
  a4 = v58;
  v8 = a5;
  v9 = a1;
  if ( v46 )
  {
    while ( 1 )
    {
      v47 = *(_DWORD *)(v46 + 32);
      v48 = *(_QWORD *)(v46 + 24);
      if ( v68 < v47 )
        v48 = *(_QWORD *)(v46 + 16);
      if ( !v48 )
        break;
      v46 = v48;
    }
    if ( v68 < v47 )
    {
      if ( v46 != *(_QWORD *)(a6 + 72) )
        goto LABEL_88;
    }
    else if ( v68 <= v47 )
    {
      goto LABEL_38;
    }
    goto LABEL_69;
  }
  v46 = v73;
  if ( v73 == *(_QWORD *)(a6 + 72) )
  {
    v46 = v73;
    v49 = 1;
    goto LABEL_71;
  }
LABEL_88:
  v62 = a6;
  v67 = v46;
  v56 = sub_220EF80(v46);
  a4 = v58;
  if ( v68 > *(_DWORD *)(v56 + 32) )
  {
    v46 = v67;
    a6 = v62;
    if ( v67 )
    {
LABEL_69:
      v49 = 1;
      if ( v73 != v46 )
        v49 = v68 < *(_DWORD *)(v46 + 32);
LABEL_71:
      v61 = a6;
      v65 = a4;
      v70 = v46;
      v50 = sub_22077B0(40);
      *(_DWORD *)(v50 + 32) = v10;
      sub_220F040(v49, v50, v70, v73);
      LODWORD(a6) = v61;
      a4 = v65;
      ++*(_QWORD *)(v61 + 88);
    }
  }
LABEL_38:
  v34 = *(_QWORD *)(a3 + 40);
  v35 = *(_QWORD **)(a3 + 48);
  if ( *(_QWORD **)(v34 + 32) == v35 )
    goto LABEL_55;
  v36 = *(_QWORD *)(v34 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v36 )
    goto LABEL_98;
  if ( (*(_QWORD *)v36 & 4) == 0 && (*(_BYTE *)(v36 + 46) & 4) != 0 )
  {
    for ( j = *(_QWORD *)v36; ; j = *(_QWORD *)v36 )
    {
      v36 = j & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v36 + 46) & 4) == 0 )
        break;
    }
  }
  v38 = **(unsigned __int16 **)(v36 + 16);
  if ( v38 == 45 || !v38 )
  {
LABEL_55:
    v44 = *(unsigned int *)(v8 + 8);
    if ( (unsigned int)v44 >= *(_DWORD *)(v8 + 12) )
    {
      sub_16CD150(v8, (const void *)(v8 + 16), 0, 16, a5, a6);
      v44 = *(unsigned int *)(v8 + 8);
    }
    v45 = (_QWORD *)(*(_QWORD *)v8 + 16 * v44);
    *v45 = v10;
    v45[1] = 0;
    ++*(_DWORD *)(v8 + 8);
    return;
  }
  v39 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v39 )
LABEL_98:
    BUG();
  v40 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
  v41 = v40;
  if ( ((*(__int64 *)v39 >> 2) & 1) == 0 && (*(_BYTE *)(v39 + 46) & 4) != 0 )
  {
    for ( k = *(_QWORD *)v39; ; k = *(_QWORD *)v41 )
    {
      v41 = k & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v41 + 46) & 4) == 0 )
        break;
    }
  }
  v43 = **(unsigned __int16 **)(v41 + 16);
  if ( !v43 || v43 == 45 )
    goto LABEL_55;
  if ( ((*(__int64 *)v39 >> 2) & 1) == 0 && (*(_BYTE *)(v39 + 46) & 4) != 0 )
  {
    for ( m = *(_QWORD *)v39; ; m = *(_QWORD *)v40 )
    {
      v40 = m & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v40 + 46) & 4) == 0 )
        break;
    }
  }
  v53 = *(unsigned int *)(v8 + 8);
  v54 = (unsigned int)v10;
  if ( (unsigned int)v53 >= *(_DWORD *)(v8 + 12) )
  {
    v77 = a4;
    sub_16CD150(v8, (const void *)(v8 + 16), 0, 16, a5, v10);
    v53 = *(unsigned int *)(v8 + 8);
    v54 = (unsigned int)v10;
    a4 = v77;
  }
  v55 = (_QWORD *)(*(_QWORD *)v8 + 16 * v53);
  *v55 = v54;
  v55[1] = v40;
  ++*(_DWORD *)(v8 + 8);
  if ( (*(_BYTE *)(v9 + 26) & 1) != 0 )
  {
    v11 = v10;
    v12 = *(_QWORD *)(a2 + 648);
LABEL_5:
    sub_1D0C4D0(v9, v12, a3, v8, a4, v11);
  }
}
