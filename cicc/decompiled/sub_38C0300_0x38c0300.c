// Function: sub_38C0300
// Address: 0x38c0300
//
__int64 __fastcall sub_38C0300(__int64 a1)
{
  __int64 *v2; // r13
  __int64 v3; // rsi
  __int64 *v4; // r14
  __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 v7; // r12
  bool v8; // cf
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r12
  unsigned __int64 i; // r15
  unsigned __int64 v13; // rdi
  _QWORD *v14; // r14
  _QWORD *v15; // r15
  unsigned __int64 v16; // r13
  unsigned __int64 j; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // r12
  unsigned __int64 *v20; // r13
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // r13
  __int64 v24; // rsi
  __int64 *v25; // r14
  __int64 v26; // rdx
  unsigned int v27; // ecx
  __int64 v28; // r12
  __int64 v29; // rcx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r12
  unsigned __int64 k; // r15
  unsigned __int64 v33; // rdi
  _QWORD *v34; // r14
  _QWORD *v35; // r15
  unsigned __int64 v36; // r13
  unsigned __int64 m; // r12
  unsigned __int64 v38; // rdi
  unsigned __int64 *v39; // r12
  unsigned __int64 *v40; // r13
  unsigned __int64 v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned int v46; // ecx
  __int64 v47; // r12
  _QWORD *v48; // rcx
  unsigned __int64 v49; // r12
  _QWORD *v50; // r15
  _QWORD *v51; // r14
  __int64 v52; // r15
  unsigned __int64 v53; // r12
  _QWORD *v54; // rcx
  _QWORD *v55; // r15
  _QWORD *v56; // r14
  unsigned __int64 *v57; // r12
  unsigned __int64 *v58; // r13
  unsigned __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  unsigned __int64 *v65; // r12
  unsigned __int64 *v66; // r13
  unsigned __int64 v67; // rdi
  __int64 v68; // rax
  int v69; // eax
  __int64 v70; // rdx
  _DWORD *v71; // rax
  _DWORD *n; // rdx
  _BYTE *v73; // rax
  int v74; // eax
  __int64 v75; // rdx
  _QWORD *v76; // rax
  _QWORD *jj; // rdx
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // r12
  __int64 v81; // r12
  __int64 v82; // rax
  __int64 v83; // r14
  unsigned __int64 *v84; // r13
  unsigned __int64 v85; // rdi
  _QWORD *v86; // rdi
  _QWORD *v87; // rdi
  int v88; // edx
  __int64 v89; // r12
  __int64 v90; // rax
  __int64 v91; // r14
  unsigned __int64 *v92; // r13
  unsigned __int64 v93; // rdi
  unsigned int v95; // ecx
  _QWORD *v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rax
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rax
  int v101; // r13d
  unsigned __int64 v102; // r12
  _QWORD *v103; // rax
  __int64 v104; // rdx
  _QWORD *kk; // rdx
  unsigned int v106; // ecx
  _DWORD *v107; // rdi
  unsigned int v108; // eax
  int v109; // eax
  unsigned __int64 v110; // rax
  unsigned __int64 v111; // rax
  int v112; // r13d
  unsigned __int64 v113; // r12
  _DWORD *v114; // rax
  __int64 v115; // rdx
  _DWORD *ii; // rdx
  _QWORD *v117; // r12
  __int64 v118; // rdx
  unsigned __int64 *v119; // r13
  unsigned __int64 *v120; // r12
  unsigned __int64 v121; // rdi
  _QWORD *v122; // r12
  __int64 v123; // rdx
  unsigned __int64 *v124; // r13
  unsigned __int64 *v125; // r12
  unsigned __int64 v126; // rdi
  _QWORD *v127; // r12
  __int64 v128; // rdx
  unsigned __int64 *v129; // r13
  unsigned __int64 *v130; // r12
  unsigned __int64 v131; // rdi
  _QWORD *v132; // r12
  __int64 v133; // rdx
  unsigned __int64 *v134; // r13
  unsigned __int64 *v135; // r12
  unsigned __int64 v136; // rdi
  _DWORD *v137; // rax
  _QWORD *v138; // rax
  __int64 *v139; // [rsp+0h] [rbp-40h]
  _QWORD *v140; // [rsp+0h] [rbp-40h]
  __int64 *v141; // [rsp+8h] [rbp-38h]
  _QWORD *v142; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 168);
  v3 = *(unsigned int *)(a1 + 176);
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    v5 = *(_QWORD *)(a1 + 168);
    while ( 1 )
    {
      v6 = (unsigned int)(((__int64)v2 - v5) >> 3) >> 7;
      v7 = 4096LL << v6;
      v8 = v6 < 0x1E;
      v9 = *v2;
      if ( !v8 )
        v7 = 0x40000000000LL;
      v10 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v11 = v9 + v7;
      if ( v9 == *(_QWORD *)(v5 + 8 * v3 - 8) )
        v11 = *(_QWORD *)(a1 + 152);
      for ( i = v10 + 192; i <= v11; i += 192LL )
      {
        v13 = i - 192;
        sub_38D7BA0(v13);
      }
      if ( v4 == ++v2 )
        break;
      v5 = *(_QWORD *)(a1 + 168);
      v3 = *(unsigned int *)(a1 + 176);
    }
  }
  v14 = *(_QWORD **)(a1 + 216);
  v15 = &v14[2 * *(unsigned int *)(a1 + 224)];
  if ( v14 != v15 )
  {
    do
    {
      v16 = *v14 + v14[1];
      for ( j = ((*v14 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 192; v16 >= j; j += 192LL )
      {
        v18 = j - 192;
        sub_38D7BA0(v18);
      }
      v14 += 2;
    }
    while ( v15 != v14 );
    v19 = *(unsigned __int64 **)(a1 + 216);
    v20 = &v19[2 * *(unsigned int *)(a1 + 224)];
    while ( v20 != v19 )
    {
      v21 = *v19;
      v19 += 2;
      _libc_free(v21);
    }
  }
  *(_DWORD *)(a1 + 224) = 0;
  v22 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v22 )
  {
    v132 = *(_QWORD **)(a1 + 168);
    *(_QWORD *)(a1 + 232) = 0;
    v133 = *v132;
    v134 = &v132[v22];
    v135 = v132 + 1;
    *(_QWORD *)(a1 + 152) = v133;
    *(_QWORD *)(a1 + 160) = v133 + 4096;
    while ( v134 != v135 )
    {
      v136 = *v135++;
      _libc_free(v136);
    }
    *(_DWORD *)(a1 + 176) = 1;
  }
  v23 = *(__int64 **)(a1 + 272);
  v24 = *(unsigned int *)(a1 + 280);
  v25 = &v23[v24];
  if ( v23 != v25 )
  {
    v26 = *(_QWORD *)(a1 + 272);
    while ( 1 )
    {
      v27 = (unsigned int)(((__int64)v23 - v26) >> 3) >> 7;
      v28 = 4096LL << v27;
      v8 = v27 < 0x1E;
      v29 = *v23;
      if ( !v8 )
        v28 = 0x40000000000LL;
      v30 = (v29 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v31 = v29 + v28;
      if ( v29 == *(_QWORD *)(v26 + 8 * v24 - 8) )
        v31 = *(_QWORD *)(a1 + 256);
      for ( k = v30 + 200; k <= v31; k += 200LL )
      {
        v33 = k - 200;
        sub_38D8820(v33);
      }
      if ( v25 == ++v23 )
        break;
      v26 = *(_QWORD *)(a1 + 272);
      v24 = *(unsigned int *)(a1 + 280);
    }
  }
  v34 = *(_QWORD **)(a1 + 320);
  v35 = &v34[2 * *(unsigned int *)(a1 + 328)];
  if ( v34 != v35 )
  {
    do
    {
      v36 = *v34 + v34[1];
      for ( m = ((*v34 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 200; v36 >= m; m += 200LL )
      {
        v38 = m - 200;
        sub_38D8820(v38);
      }
      v34 += 2;
    }
    while ( v35 != v34 );
    v39 = *(unsigned __int64 **)(a1 + 320);
    v40 = &v39[2 * *(unsigned int *)(a1 + 328)];
    while ( v40 != v39 )
    {
      v41 = *v39;
      v39 += 2;
      _libc_free(v41);
    }
  }
  *(_DWORD *)(a1 + 328) = 0;
  v42 = *(unsigned int *)(a1 + 280);
  if ( (_DWORD)v42 )
  {
    v127 = *(_QWORD **)(a1 + 272);
    *(_QWORD *)(a1 + 336) = 0;
    v128 = *v127;
    v129 = &v127[v42];
    v130 = v127 + 1;
    *(_QWORD *)(a1 + 256) = v128;
    *(_QWORD *)(a1 + 264) = v128 + 4096;
    while ( v129 != v130 )
    {
      v131 = *v130++;
      _libc_free(v131);
    }
    *(_DWORD *)(a1 + 280) = 1;
  }
  v43 = *(__int64 **)(a1 + 376);
  v44 = *(unsigned int *)(a1 + 384);
  v141 = v43;
  v139 = &v43[v44];
  if ( v43 != v139 )
  {
    while ( 1 )
    {
      v45 = *v141;
      v46 = (unsigned int)(v141 - v43) >> 7;
      v47 = 4096LL << v46;
      if ( v46 >= 0x1E )
        v47 = 0x40000000000LL;
      v48 = (_QWORD *)((v45 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v49 = v45 + v47;
      if ( v45 == v43[v44 - 1] )
        v49 = *(_QWORD *)(a1 + 360);
      v50 = v48 + 24;
      if ( v49 >= (unsigned __int64)(v48 + 24) )
      {
        while ( 1 )
        {
          *v48 = &unk_4A3E5C0;
          v51 = v48;
          sub_38D77F0(v48);
          v48 = v50;
          if ( v49 < (unsigned __int64)(v51 + 48) )
            break;
          v50 += 24;
        }
      }
      if ( v139 == ++v141 )
        break;
      v43 = *(__int64 **)(a1 + 376);
      v44 = *(unsigned int *)(a1 + 384);
    }
  }
  v52 = 2LL * *(unsigned int *)(a1 + 432);
  v142 = *(_QWORD **)(a1 + 424);
  v140 = &v142[v52];
  if ( v142 != &v142[v52] )
  {
    do
    {
      v53 = *v142 + v142[1];
      v54 = (_QWORD *)((*v142 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
      v55 = v54 + 24;
      if ( v53 >= (unsigned __int64)(v54 + 24) )
      {
        while ( 1 )
        {
          *v54 = &unk_4A3E5C0;
          v56 = v54;
          sub_38D77F0(v54);
          v54 = v55;
          if ( v53 < (unsigned __int64)(v56 + 48) )
            break;
          v55 += 24;
        }
      }
      v142 += 2;
    }
    while ( v140 != v142 );
    v57 = *(unsigned __int64 **)(a1 + 424);
    v58 = &v57[2 * *(unsigned int *)(a1 + 432)];
    while ( v58 != v57 )
    {
      v59 = *v57;
      v57 += 2;
      _libc_free(v59);
    }
  }
  *(_DWORD *)(a1 + 432) = 0;
  v60 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v60 )
  {
    v122 = *(_QWORD **)(a1 + 376);
    *(_QWORD *)(a1 + 440) = 0;
    v123 = *v122;
    v124 = &v122[v60];
    v125 = v122 + 1;
    *(_QWORD *)(a1 + 360) = v123;
    *(_QWORD *)(a1 + 368) = v123 + 4096;
    while ( v124 != v125 )
    {
      v126 = *v125++;
      _libc_free(v126);
    }
    *(_DWORD *)(a1 + 384) = 1;
  }
  sub_38C0010(a1 + 1376);
  if ( *(_DWORD *)(a1 + 644) )
  {
    v61 = 0;
    v62 = *(unsigned int *)(a1 + 640);
    if ( (_DWORD)v62 )
    {
      do
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 632) + v61) = 0;
        v61 += 8;
      }
      while ( 8 * v62 != v61 );
    }
    *(_QWORD *)(a1 + 644) = 0;
  }
  if ( *(_DWORD *)(a1 + 580) )
  {
    v63 = 0;
    v64 = *(unsigned int *)(a1 + 576);
    if ( (_DWORD)v64 )
    {
      do
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 568) + v63) = 0;
        v63 += 8;
      }
      while ( v63 != 8 * v64 );
    }
    *(_QWORD *)(a1 + 580) = 0;
  }
  v65 = *(unsigned __int64 **)(a1 + 112);
  v66 = &v65[2 * *(unsigned int *)(a1 + 120)];
  while ( v65 != v66 )
  {
    v67 = *v65;
    v65 += 2;
    _libc_free(v67);
  }
  v68 = *(unsigned int *)(a1 + 72);
  *(_DWORD *)(a1 + 120) = 0;
  if ( (_DWORD)v68 )
  {
    *(_QWORD *)(a1 + 128) = 0;
    v117 = *(_QWORD **)(a1 + 64);
    v118 = *v117;
    v119 = &v117[v68];
    v120 = v117 + 1;
    *(_QWORD *)(a1 + 48) = v118;
    *(_QWORD *)(a1 + 56) = v118 + 4096;
    while ( v119 != v120 )
    {
      v121 = *v120++;
      _libc_free(v121);
    }
    *(_DWORD *)(a1 + 72) = 1;
  }
  v69 = *(_DWORD *)(a1 + 712);
  ++*(_QWORD *)(a1 + 696);
  if ( !v69 )
  {
    if ( !*(_DWORD *)(a1 + 716) )
      goto LABEL_72;
    v70 = *(unsigned int *)(a1 + 720);
    if ( (unsigned int)v70 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 704));
      *(_QWORD *)(a1 + 704) = 0;
      *(_QWORD *)(a1 + 712) = 0;
      *(_DWORD *)(a1 + 720) = 0;
      goto LABEL_72;
    }
    goto LABEL_69;
  }
  v106 = 4 * v69;
  v70 = *(unsigned int *)(a1 + 720);
  if ( (unsigned int)(4 * v69) < 0x40 )
    v106 = 64;
  if ( v106 >= (unsigned int)v70 )
  {
LABEL_69:
    v71 = *(_DWORD **)(a1 + 704);
    for ( n = &v71[4 * v70]; n != v71; v71 += 4 )
      *v71 = -1;
    *(_QWORD *)(a1 + 712) = 0;
    goto LABEL_72;
  }
  v107 = *(_DWORD **)(a1 + 704);
  v108 = v69 - 1;
  if ( !v108 )
  {
    v113 = 2048;
    v112 = 128;
LABEL_120:
    j___libc_free_0((unsigned __int64)v107);
    *(_DWORD *)(a1 + 720) = v112;
    v114 = (_DWORD *)sub_22077B0(v113);
    v115 = *(unsigned int *)(a1 + 720);
    *(_QWORD *)(a1 + 712) = 0;
    *(_QWORD *)(a1 + 704) = v114;
    for ( ii = &v114[4 * v115]; ii != v114; v114 += 4 )
    {
      if ( v114 )
        *v114 = -1;
    }
    goto LABEL_72;
  }
  _BitScanReverse(&v108, v108);
  v109 = 1 << (33 - (v108 ^ 0x1F));
  if ( v109 < 64 )
    v109 = 64;
  if ( (_DWORD)v70 != v109 )
  {
    v110 = (4 * v109 / 3u + 1) | ((unsigned __int64)(4 * v109 / 3u + 1) >> 1);
    v111 = ((v110 | (v110 >> 2)) >> 4) | v110 | (v110 >> 2) | ((((v110 | (v110 >> 2)) >> 4) | v110 | (v110 >> 2)) >> 8);
    v112 = (v111 | (v111 >> 16)) + 1;
    v113 = 16 * ((v111 | (v111 >> 16)) + 1);
    goto LABEL_120;
  }
  *(_QWORD *)(a1 + 712) = 0;
  v137 = &v107[4 * (unsigned int)v70];
  do
  {
    if ( v107 )
      *v107 = -1;
    v107 += 4;
  }
  while ( v137 != v107 );
LABEL_72:
  v73 = *(_BYTE **)(a1 + 944);
  *(_DWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *v73 = 0;
  sub_38BBDE0(*(_QWORD **)(a1 + 992));
  ++*(_QWORD *)(a1 + 1048);
  *(_QWORD *)(a1 + 1000) = a1 + 984;
  *(_QWORD *)(a1 + 1008) = a1 + 984;
  v74 = *(_DWORD *)(a1 + 1064);
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  if ( !v74 )
  {
    if ( !*(_DWORD *)(a1 + 1068) )
      goto LABEL_78;
    v75 = *(unsigned int *)(a1 + 1072);
    if ( (unsigned int)v75 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 1056));
      *(_QWORD *)(a1 + 1056) = 0;
      *(_QWORD *)(a1 + 1064) = 0;
      *(_DWORD *)(a1 + 1072) = 0;
      goto LABEL_78;
    }
    goto LABEL_75;
  }
  v95 = 4 * v74;
  v75 = *(unsigned int *)(a1 + 1072);
  if ( (unsigned int)(4 * v74) < 0x40 )
    v95 = 64;
  if ( v95 >= (unsigned int)v75 )
  {
LABEL_75:
    v76 = *(_QWORD **)(a1 + 1056);
    for ( jj = &v76[v75]; jj != v76; ++v76 )
      *v76 = -8;
    *(_QWORD *)(a1 + 1064) = 0;
    goto LABEL_78;
  }
  v96 = *(_QWORD **)(a1 + 1056);
  v97 = v74 - 1;
  if ( !v97 )
  {
    v102 = 1024;
    v101 = 128;
LABEL_107:
    j___libc_free_0((unsigned __int64)v96);
    *(_DWORD *)(a1 + 1072) = v101;
    v103 = (_QWORD *)sub_22077B0(v102);
    v104 = *(unsigned int *)(a1 + 1072);
    *(_QWORD *)(a1 + 1064) = 0;
    *(_QWORD *)(a1 + 1056) = v103;
    for ( kk = &v103[v104]; kk != v103; ++v103 )
    {
      if ( v103 )
        *v103 = -8;
    }
    goto LABEL_78;
  }
  _BitScanReverse(&v97, v97);
  v98 = (unsigned int)(1 << (33 - (v97 ^ 0x1F)));
  if ( (int)v98 < 64 )
    v98 = 64;
  if ( (_DWORD)v98 != (_DWORD)v75 )
  {
    v99 = (4 * (int)v98 / 3u + 1) | ((unsigned __int64)(4 * (int)v98 / 3u + 1) >> 1);
    v100 = ((v99 | (v99 >> 2)) >> 4) | v99 | (v99 >> 2) | ((((v99 | (v99 >> 2)) >> 4) | v99 | (v99 >> 2)) >> 8);
    v101 = (v100 | (v100 >> 16)) + 1;
    v102 = 8 * ((v100 | (v100 >> 16)) + 1);
    goto LABEL_107;
  }
  *(_QWORD *)(a1 + 1064) = 0;
  v138 = &v96[v98];
  do
  {
    if ( v96 )
      *v96 = -8;
    ++v96;
  }
  while ( v138 != v96 );
LABEL_78:
  v78 = *(_QWORD *)(a1 + 1080);
  if ( v78 != *(_QWORD *)(a1 + 1088) )
    *(_QWORD *)(a1 + 1088) = v78;
  v79 = *(_QWORD *)(a1 + 1104);
  if ( v79 != *(_QWORD *)(a1 + 1112) )
    *(_QWORD *)(a1 + 1112) = v79;
  *(_QWORD *)(a1 + 1128) = 0;
  v80 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 1136) = 0;
  *(_DWORD *)(a1 + 1164) = 0;
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0x10000;
  *(_QWORD *)(a1 + 40) = 0;
  if ( v80 )
  {
    sub_390FCC0(v80);
    j_j___libc_free_0(v80);
  }
  if ( *(_DWORD *)(a1 + 1180) )
  {
    v81 = 0;
    v82 = *(unsigned int *)(a1 + 1176);
    v83 = 8 * v82;
    if ( (_DWORD)v82 )
    {
      do
      {
        v84 = (unsigned __int64 *)(v81 + *(_QWORD *)(a1 + 1168));
        v85 = *v84;
        if ( *v84 != -8 && v85 )
          _libc_free(v85);
        v81 += 8;
        *v84 = 0;
      }
      while ( v81 != v83 );
    }
    *(_QWORD *)(a1 + 1180) = 0;
  }
  sub_38BBFE0(*(_QWORD **)(a1 + 1216));
  *(_QWORD *)(a1 + 1216) = 0;
  v86 = *(_QWORD **)(a1 + 1264);
  *(_QWORD *)(a1 + 1224) = a1 + 1208;
  *(_QWORD *)(a1 + 1232) = a1 + 1208;
  *(_QWORD *)(a1 + 1240) = 0;
  sub_38BC2E0(v86);
  *(_QWORD *)(a1 + 1264) = 0;
  v87 = *(_QWORD **)(a1 + 1312);
  *(_QWORD *)(a1 + 1272) = a1 + 1256;
  *(_QWORD *)(a1 + 1280) = a1 + 1256;
  *(_QWORD *)(a1 + 1288) = 0;
  sub_38BC5E0(v87);
  v88 = *(_DWORD *)(a1 + 676);
  *(_QWORD *)(a1 + 1312) = 0;
  *(_QWORD *)(a1 + 1320) = a1 + 1304;
  *(_QWORD *)(a1 + 1328) = a1 + 1304;
  *(_QWORD *)(a1 + 1336) = 0;
  if ( v88 )
  {
    v89 = 0;
    v90 = *(unsigned int *)(a1 + 672);
    v91 = 8 * v90;
    if ( (_DWORD)v90 )
    {
      do
      {
        v92 = (unsigned __int64 *)(v89 + *(_QWORD *)(a1 + 664));
        v93 = *v92;
        if ( *v92 != -8 && v93 )
          _libc_free(v93);
        v89 += 8;
        *v92 = 0;
      }
      while ( v89 != v91 );
    }
    *(_QWORD *)(a1 + 676) = 0;
  }
  *(_BYTE *)(a1 + 1162) = 1;
  *(_WORD *)(a1 + 1040) = 0;
  *(_DWORD *)(a1 + 1044) = 0;
  *(_BYTE *)(a1 + 1481) = 0;
  return 0;
}
