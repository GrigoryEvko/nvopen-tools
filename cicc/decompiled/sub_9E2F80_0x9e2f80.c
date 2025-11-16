// Function: sub_9E2F80
// Address: 0x9e2f80
//
__int64 __fastcall sub_9E2F80(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  int v5; // r12d
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rsi
  int v10; // r10d
  unsigned int i; // eax
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned int v14; // r14d
  _BYTE *v16; // rsi
  __int64 v17; // r10
  unsigned int v18; // esi
  __int64 v19; // r8
  __int64 *v20; // r11
  int v21; // r15d
  unsigned int j; // ecx
  __int64 *v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // ecx
  unsigned int v26; // esi
  __int64 v27; // r8
  _DWORD *v28; // rdx
  int v29; // edi
  __int64 v30; // rax
  _DWORD *v31; // r8
  unsigned __int64 v32; // rdx
  __int64 v33; // r9
  size_t v34; // rdx
  int v35; // eax
  __int64 v36; // r11
  unsigned int v37; // esi
  int v38; // edx
  _DWORD *v39; // rax
  int v40; // r9d
  int v41; // ecx
  int v42; // ecx
  int v43; // esi
  int v44; // esi
  __int64 v45; // rcx
  int v46; // r9d
  __int64 *v47; // rdi
  unsigned int v48; // edx
  __int64 v49; // r8
  unsigned int v50; // edx
  int v51; // eax
  int v52; // edx
  __int64 v53; // rdi
  unsigned int v54; // r15d
  int k; // r8d
  __int64 *v56; // rsi
  __int64 v57; // rcx
  unsigned int v58; // r15d
  int v59; // edx
  int v60; // eax
  __int64 v61; // r9
  _DWORD *v62; // rsi
  int v63; // edi
  unsigned int v64; // r11d
  int v65; // r8d
  int v66; // r8d
  _DWORD *v67; // rdi
  __int64 v68; // [rsp+8h] [rbp-58h]
  __int64 v69; // [rsp+8h] [rbp-58h]
  int v70; // [rsp+10h] [rbp-50h]
  __int64 v71; // [rsp+10h] [rbp-50h]
  int v72; // [rsp+10h] [rbp-50h]
  int v73; // [rsp+10h] [rbp-50h]
  int v74; // [rsp+10h] [rbp-50h]
  unsigned int v75; // [rsp+18h] [rbp-48h]
  int v76; // [rsp+18h] [rbp-48h]
  int v77; // [rsp+18h] [rbp-48h]
  __int64 v78; // [rsp+18h] [rbp-48h]
  int v79; // [rsp+18h] [rbp-48h]
  _DWORD *v80; // [rsp+18h] [rbp-48h]
  int n; // [rsp+20h] [rbp-40h]
  size_t na; // [rsp+20h] [rbp-40h]
  size_t nb; // [rsp+20h] [rbp-40h]
  size_t ng; // [rsp+20h] [rbp-40h]
  int nc; // [rsp+20h] [rbp-40h]
  size_t nh; // [rsp+20h] [rbp-40h]
  int nd; // [rsp+20h] [rbp-40h]
  int ne; // [rsp+20h] [rbp-40h]
  int nf; // [rsp+20h] [rbp-40h]
  __int64 v90[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = -1;
  v90[0] = a2;
  if ( a4 )
    v5 = *a3;
  v7 = *(unsigned int *)(a1 + 608);
  v8 = v90[0];
  v9 = *(_QWORD *)(a1 + 592);
  if ( (_DWORD)v7 )
  {
    v10 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(37 * v5) | ((unsigned __int64)((LODWORD(v90[0]) >> 9) ^ (LODWORD(v90[0]) >> 4)) << 32))) >> 31)
             ^ (756364221 * v5)); ; i = (v7 - 1) & v13 )
    {
      v12 = v9 + 24LL * i;
      if ( v90[0] == *(_QWORD *)v12 && v5 == *(_DWORD *)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_DWORD *)(v12 + 8) == -1 )
        goto LABEL_13;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != v9 + 24 * v7 )
      return *(unsigned int *)(v12 + 16);
  }
LABEL_13:
  v16 = *(_BYTE **)(a1 + 536);
  v17 = (__int64)&v16[-*(_QWORD *)(a1 + 528)] >> 3;
  v14 = v17;
  if ( v16 == *(_BYTE **)(a1 + 544) )
  {
    v78 = a4;
    nh = (__int64)(*(_QWORD *)(a1 + 536) - *(_QWORD *)(a1 + 528)) >> 3;
    sub_918210(a1 + 528, v16, v90);
    a4 = v78;
    LODWORD(v17) = nh;
  }
  else
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = v90[0];
      v16 = *(_BYTE **)(a1 + 536);
    }
    *(_QWORD *)(a1 + 536) = v16 + 8;
  }
  if ( !a4 )
    goto LABEL_18;
  v26 = *(_DWORD *)(a1 + 576);
  na = a1 + 552;
  if ( !v26 )
  {
    ++*(_QWORD *)(a1 + 552);
    goto LABEL_36;
  }
  v27 = *(_QWORD *)(a1 + 560);
  v75 = (v26 - 1) & (37 * v17);
  v28 = (_DWORD *)(v27 + 32LL * v75);
  v29 = *v28;
  if ( (_DWORD)v17 != *v28 )
  {
    v72 = 1;
    v39 = 0;
    while ( v29 != -1 )
    {
      if ( v29 == -2 && !v39 )
        v39 = v28;
      v75 = (v26 - 1) & (v72 + v75);
      v28 = (_DWORD *)(v27 + 32LL * v75);
      v29 = *v28;
      if ( v14 == *v28 )
        goto LABEL_29;
      ++v72;
    }
    if ( !v39 )
      v39 = v28;
    v59 = *(_DWORD *)(a1 + 568);
    ++*(_QWORD *)(a1 + 552);
    v38 = v59 + 1;
    if ( 4 * v38 < 3 * v26 )
    {
      if ( v26 - *(_DWORD *)(a1 + 572) - v38 > v26 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(a1 + 568) = v38;
        if ( *v39 != -1 )
          --*(_DWORD *)(a1 + 572);
        *v39 = v17;
        v31 = v39 + 2;
        *((_QWORD *)v39 + 1) = v39 + 6;
        v32 = 1;
        *((_QWORD *)v39 + 2) = 0x100000000LL;
        v30 = 0;
        goto LABEL_30;
      }
      v68 = a4;
      v73 = 37 * v17;
      v79 = v17;
      sub_9D85E0(na, v26);
      v60 = *(_DWORD *)(a1 + 576);
      if ( v60 )
      {
        v61 = *(_QWORD *)(a1 + 560);
        v62 = 0;
        nf = v60 - 1;
        LODWORD(v17) = v79;
        v63 = 1;
        v64 = (v60 - 1) & v73;
        v38 = *(_DWORD *)(a1 + 568) + 1;
        a4 = v68;
        v39 = (_DWORD *)(v61 + 32LL * v64);
        v65 = *v39;
        if ( v79 != *v39 )
        {
          while ( v65 != -1 )
          {
            if ( v65 == -2 && !v62 )
              v62 = v39;
            v64 = nf & (v63 + v64);
            v39 = (_DWORD *)(v61 + 32LL * v64);
            v65 = *v39;
            if ( v14 == *v39 )
              goto LABEL_38;
            ++v63;
          }
          if ( v62 )
            v39 = v62;
        }
        goto LABEL_38;
      }
LABEL_117:
      ++*(_DWORD *)(a1 + 568);
      BUG();
    }
LABEL_36:
    v71 = a4;
    v77 = v17;
    sub_9D85E0(na, 2 * v26);
    v35 = *(_DWORD *)(a1 + 576);
    if ( v35 )
    {
      LODWORD(v17) = v77;
      v36 = *(_QWORD *)(a1 + 560);
      nc = v35 - 1;
      v37 = (v35 - 1) & (37 * v77);
      v38 = *(_DWORD *)(a1 + 568) + 1;
      a4 = v71;
      v39 = (_DWORD *)(v36 + 32LL * v37);
      v40 = *v39;
      if ( v77 != *v39 )
      {
        v66 = 1;
        v67 = 0;
        while ( v40 != -1 )
        {
          if ( v40 == -2 && !v67 )
            v67 = v39;
          v37 = nc & (v66 + v37);
          v39 = (_DWORD *)(v36 + 32LL * v37);
          v40 = *v39;
          if ( v14 == *v39 )
            goto LABEL_38;
          ++v66;
        }
        if ( v67 )
          v39 = v67;
      }
      goto LABEL_38;
    }
    goto LABEL_117;
  }
LABEL_29:
  v30 = (unsigned int)v28[4];
  v31 = v28 + 2;
  v32 = (unsigned int)v28[5];
LABEL_30:
  nb = 4 * a4;
  v33 = (4 * a4) >> 2;
  if ( v32 < v30 + v33 )
  {
    v69 = (4 * a4) >> 2;
    v74 = v17;
    v80 = v31;
    sub_C8D5F0(v31, v31 + 4, v30 + v33, 4);
    v31 = v80;
    LODWORD(v33) = v69;
    LODWORD(v17) = v74;
    v30 = (unsigned int)v80[2];
  }
  if ( nb )
  {
    v70 = v33;
    v76 = v17;
    v34 = nb;
    ng = (size_t)v31;
    memcpy((void *)(*(_QWORD *)v31 + 4 * v30), a3, v34);
    v31 = (_DWORD *)ng;
    LODWORD(v33) = v70;
    LODWORD(v17) = v76;
    LODWORD(v30) = *(_DWORD *)(ng + 8);
  }
  v31[2] = v33 + v30;
LABEL_18:
  v18 = *(_DWORD *)(a1 + 608);
  if ( !v18 )
  {
    ++*(_QWORD *)(a1 + 584);
    goto LABEL_55;
  }
  n = 1;
  v19 = *(_QWORD *)(a1 + 592);
  v20 = 0;
  v21 = ((0xBF58476D1CE4E5B9LL
        * ((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
      ^ (756364221 * v5);
  for ( j = v21 & (v18 - 1); ; j = (v18 - 1) & v25 )
  {
    v23 = (__int64 *)(v19 + 24LL * j);
    v24 = *v23;
    if ( v8 == *v23 && v5 == *((_DWORD *)v23 + 2) )
      return v14;
    if ( v24 == -4096 )
      break;
    if ( v24 == -8192 && *((_DWORD *)v23 + 2) == -2 && !v20 )
      v20 = (__int64 *)(v19 + 24LL * j);
LABEL_26:
    v25 = n + j;
    ++n;
  }
  if ( *((_DWORD *)v23 + 2) != -1 )
    goto LABEL_26;
  v41 = *(_DWORD *)(a1 + 600);
  if ( v20 )
    v23 = v20;
  ++*(_QWORD *)(a1 + 584);
  v42 = v41 + 1;
  if ( 4 * v42 < 3 * v18 )
  {
    if ( v18 - *(_DWORD *)(a1 + 604) - v42 > v18 >> 3 )
      goto LABEL_46;
    ne = v17;
    sub_9E2CC0(a1 + 584, v18);
    v51 = *(_DWORD *)(a1 + 608);
    if ( v51 )
    {
      v52 = v51 - 1;
      LODWORD(v17) = ne;
      v23 = 0;
      v54 = v52 & v21;
      for ( k = 1; ; ++k )
      {
        v53 = *(_QWORD *)(a1 + 592);
        v56 = (__int64 *)(v53 + 24LL * v54);
        v57 = *v56;
        if ( v8 == *v56 && v5 == *((_DWORD *)v56 + 2) )
        {
          v42 = *(_DWORD *)(a1 + 600) + 1;
          v23 = (__int64 *)(v53 + 24LL * v54);
          goto LABEL_46;
        }
        if ( v57 == -4096 )
        {
          if ( *((_DWORD *)v56 + 2) == -1 )
          {
            v42 = *(_DWORD *)(a1 + 600) + 1;
            if ( !v23 )
              v23 = (__int64 *)(v53 + 24LL * v54);
            goto LABEL_46;
          }
        }
        else if ( v57 == -8192 && *((_DWORD *)v56 + 2) == -2 && !v23 )
        {
          v23 = (__int64 *)(v53 + 24LL * v54);
        }
        v58 = k + v54;
        v54 = v52 & v58;
      }
    }
LABEL_118:
    ++*(_DWORD *)(a1 + 600);
    BUG();
  }
LABEL_55:
  nd = v17;
  sub_9E2CC0(a1 + 584, 2 * v18);
  v43 = *(_DWORD *)(a1 + 608);
  if ( !v43 )
    goto LABEL_118;
  v44 = v43 - 1;
  LODWORD(v17) = nd;
  v46 = 1;
  v47 = 0;
  v48 = v44
      & (((0xBF58476D1CE4E5B9LL
         * ((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
       ^ (756364221 * v5));
  while ( 2 )
  {
    v45 = *(_QWORD *)(a1 + 592);
    v23 = (__int64 *)(v45 + 24LL * v48);
    v49 = *v23;
    if ( v8 == *v23 && v5 == *((_DWORD *)v23 + 2) )
    {
      v42 = *(_DWORD *)(a1 + 600) + 1;
      goto LABEL_46;
    }
    if ( v49 != -4096 )
    {
      if ( v49 == -8192 && *((_DWORD *)v23 + 2) == -2 && !v47 )
        v47 = (__int64 *)(v45 + 24LL * v48);
      goto LABEL_63;
    }
    if ( *((_DWORD *)v23 + 2) != -1 )
    {
LABEL_63:
      v50 = v46 + v48;
      ++v46;
      v48 = v44 & v50;
      continue;
    }
    break;
  }
  v42 = *(_DWORD *)(a1 + 600) + 1;
  if ( v47 )
    v23 = v47;
LABEL_46:
  *(_DWORD *)(a1 + 600) = v42;
  if ( *v23 != -4096 || *((_DWORD *)v23 + 2) != -1 )
    --*(_DWORD *)(a1 + 604);
  *v23 = v8;
  *((_DWORD *)v23 + 2) = v5;
  *((_DWORD *)v23 + 4) = v17;
  return v14;
}
