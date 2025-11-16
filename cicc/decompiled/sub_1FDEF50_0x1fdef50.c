// Function: sub_1FDEF50
// Address: 0x1fdef50
//
__int64 __fastcall sub_1FDEF50(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  int v10; // eax
  __int64 v11; // rdx
  _DWORD *v12; // rax
  _DWORD *n; // rdx
  int v14; // eax
  __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *kk; // rdx
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // rdi
  void *v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *mm; // rdx
  int v29; // eax
  __int64 v30; // rdx
  _DWORD *v31; // rax
  _DWORD *i1; // rdx
  int v33; // eax
  __int64 v34; // rdx
  size_t v35; // rdx
  int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // rdx
  _QWORD *v39; // r12
  _QWORD *i4; // r13
  int v41; // eax
  __int64 result; // rax
  __int64 v43; // rdx
  __int64 i5; // rdx
  unsigned int v45; // ecx
  _QWORD *v46; // rdi
  unsigned int v47; // eax
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  int v51; // r13d
  __int64 v52; // r12
  __int64 v53; // rdx
  __int64 i6; // rdx
  unsigned int v55; // ecx
  _DWORD *v56; // rdi
  unsigned int v57; // eax
  __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  int v61; // r13d
  __int64 v62; // r12
  _DWORD *v63; // rax
  __int64 v64; // rdx
  _DWORD *i3; // rdx
  unsigned int v66; // ecx
  _DWORD *v67; // rdi
  unsigned int v68; // eax
  int v69; // eax
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rax
  int v72; // r13d
  __int64 v73; // r12
  _DWORD *v74; // rax
  __int64 v75; // rdx
  _DWORD *i2; // rdx
  unsigned int v77; // ecx
  _QWORD *v78; // rdi
  unsigned int v79; // eax
  int v80; // eax
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rax
  int v83; // r13d
  __int64 v84; // r12
  _QWORD *v85; // rax
  __int64 v86; // rdx
  _QWORD *nn; // rdx
  unsigned int v88; // ecx
  _QWORD *v89; // rdi
  unsigned int v90; // eax
  int v91; // eax
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // rax
  int v94; // r13d
  __int64 v95; // r12
  _QWORD *v96; // rax
  __int64 v97; // rdx
  _QWORD *jj; // rdx
  unsigned int v99; // ecx
  _QWORD *v100; // rdi
  unsigned int v101; // eax
  int v102; // eax
  unsigned __int64 v103; // rax
  unsigned __int64 v104; // rax
  int v105; // r13d
  __int64 v106; // r12
  _QWORD *v107; // rax
  __int64 v108; // rdx
  _QWORD *m; // rdx
  unsigned int v110; // ecx
  _QWORD *v111; // rdi
  unsigned int v112; // eax
  int v113; // eax
  unsigned __int64 v114; // rax
  unsigned __int64 v115; // rax
  int v116; // r13d
  __int64 v117; // r12
  _QWORD *v118; // rax
  __int64 v119; // rdx
  _QWORD *j; // rdx
  unsigned int v121; // ecx
  _DWORD *v122; // rdi
  unsigned int v123; // eax
  int v124; // eax
  unsigned __int64 v125; // rax
  unsigned __int64 v126; // rax
  int v127; // r13d
  __int64 v128; // r12
  _DWORD *v129; // rax
  __int64 v130; // rdx
  _DWORD *ii; // rdx
  _DWORD *v132; // rax
  _DWORD *v133; // rax
  _QWORD *v134; // rax
  _QWORD *v135; // rax
  _QWORD *v136; // rax
  _QWORD *v137; // rax
  _DWORD *v138; // rax

  v2 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 68) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 56));
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_DWORD *)(a1 + 72) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v110 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 72);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v110 = 64;
  if ( v110 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 56);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_7;
  }
  v111 = *(_QWORD **)(a1 + 56);
  v112 = v2 - 1;
  if ( !v112 )
  {
    v117 = 2048;
    v116 = 128;
LABEL_162:
    j___libc_free_0(v111);
    *(_DWORD *)(a1 + 72) = v116;
    v118 = (_QWORD *)sub_22077B0(v117);
    v119 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v118;
    for ( j = &v118[2 * v119]; j != v118; v118 += 2 )
    {
      if ( v118 )
        *v118 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v112, v112);
  v113 = 1 << (33 - (v112 ^ 0x1F));
  if ( v113 < 64 )
    v113 = 64;
  if ( (_DWORD)v3 != v113 )
  {
    v114 = (((4 * v113 / 3u + 1) | ((unsigned __int64)(4 * v113 / 3u + 1) >> 1)) >> 2)
         | (4 * v113 / 3u + 1)
         | ((unsigned __int64)(4 * v113 / 3u + 1) >> 1)
         | (((((4 * v113 / 3u + 1) | ((unsigned __int64)(4 * v113 / 3u + 1) >> 1)) >> 2)
           | (4 * v113 / 3u + 1)
           | ((unsigned __int64)(4 * v113 / 3u + 1) >> 1)) >> 4);
    v115 = (v114 >> 8) | v114;
    v116 = (v115 | (v115 >> 16)) + 1;
    v117 = 16 * ((v115 | (v115 >> 16)) + 1);
    goto LABEL_162;
  }
  *(_QWORD *)(a1 + 64) = 0;
  v137 = &v111[2 * (unsigned int)v3];
  do
  {
    if ( v111 )
      *v111 = -8;
    v111 += 2;
  }
  while ( v137 != v111 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 228) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 232);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 216));
      *(_QWORD *)(a1 + 216) = 0;
      *(_QWORD *)(a1 + 224) = 0;
      *(_DWORD *)(a1 + 232) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v99 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 232);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v99 = 64;
  if ( (unsigned int)v7 <= v99 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 216);
    for ( k = &v8[2 * v7]; k != v8; v8 += 2 )
      *v8 = -8;
    *(_QWORD *)(a1 + 224) = 0;
    goto LABEL_13;
  }
  v100 = *(_QWORD **)(a1 + 216);
  v101 = v6 - 1;
  if ( !v101 )
  {
    v106 = 2048;
    v105 = 128;
LABEL_149:
    j___libc_free_0(v100);
    *(_DWORD *)(a1 + 232) = v105;
    v107 = (_QWORD *)sub_22077B0(v106);
    v108 = *(unsigned int *)(a1 + 232);
    *(_QWORD *)(a1 + 224) = 0;
    *(_QWORD *)(a1 + 216) = v107;
    for ( m = &v107[2 * v108]; m != v107; v107 += 2 )
    {
      if ( v107 )
        *v107 = -8;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v101, v101);
  v102 = 1 << (33 - (v101 ^ 0x1F));
  if ( v102 < 64 )
    v102 = 64;
  if ( (_DWORD)v7 != v102 )
  {
    v103 = (((4 * v102 / 3u + 1) | ((unsigned __int64)(4 * v102 / 3u + 1) >> 1)) >> 2)
         | (4 * v102 / 3u + 1)
         | ((unsigned __int64)(4 * v102 / 3u + 1) >> 1)
         | (((((4 * v102 / 3u + 1) | ((unsigned __int64)(4 * v102 / 3u + 1) >> 1)) >> 2)
           | (4 * v102 / 3u + 1)
           | ((unsigned __int64)(4 * v102 / 3u + 1) >> 1)) >> 4);
    v104 = (v103 >> 8) | v103;
    v105 = (v104 | (v104 >> 16)) + 1;
    v106 = 16 * ((v104 | (v104 >> 16)) + 1);
    goto LABEL_149;
  }
  *(_QWORD *)(a1 + 224) = 0;
  v136 = &v100[2 * (unsigned int)v7];
  do
  {
    if ( v100 )
      *v100 = -8;
    v100 += 2;
  }
  while ( v136 != v100 );
LABEL_13:
  v10 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 260) )
      goto LABEL_19;
    v11 = *(unsigned int *)(a1 + 264);
    if ( (unsigned int)v11 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 248));
      *(_QWORD *)(a1 + 248) = 0;
      *(_QWORD *)(a1 + 256) = 0;
      *(_DWORD *)(a1 + 264) = 0;
      goto LABEL_19;
    }
    goto LABEL_16;
  }
  v121 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 264);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v121 = 64;
  if ( (unsigned int)v11 <= v121 )
  {
LABEL_16:
    v12 = *(_DWORD **)(a1 + 248);
    for ( n = &v12[4 * v11]; n != v12; v12 += 4 )
      *v12 = -1;
    *(_QWORD *)(a1 + 256) = 0;
    goto LABEL_19;
  }
  v122 = *(_DWORD **)(a1 + 248);
  v123 = v10 - 1;
  if ( !v123 )
  {
    v128 = 2048;
    v127 = 128;
LABEL_175:
    j___libc_free_0(v122);
    *(_DWORD *)(a1 + 264) = v127;
    v129 = (_DWORD *)sub_22077B0(v128);
    v130 = *(unsigned int *)(a1 + 264);
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 248) = v129;
    for ( ii = &v129[4 * v130]; ii != v129; v129 += 4 )
    {
      if ( v129 )
        *v129 = -1;
    }
    goto LABEL_19;
  }
  _BitScanReverse(&v123, v123);
  v124 = 1 << (33 - (v123 ^ 0x1F));
  if ( v124 < 64 )
    v124 = 64;
  if ( (_DWORD)v11 != v124 )
  {
    v125 = (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
         | (4 * v124 / 3u + 1)
         | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)
         | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
           | (4 * v124 / 3u + 1)
           | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4);
    v126 = (v125 >> 8) | v125;
    v127 = (v126 | (v126 >> 16)) + 1;
    v128 = 16 * ((v126 | (v126 >> 16)) + 1);
    goto LABEL_175;
  }
  *(_QWORD *)(a1 + 256) = 0;
  v138 = &v122[4 * (unsigned int)v11];
  do
  {
    if ( v122 )
      *v122 = -1;
    v122 += 4;
  }
  while ( v138 != v122 );
LABEL_19:
  v14 = *(_DWORD *)(a1 + 352);
  ++*(_QWORD *)(a1 + 336);
  if ( v14 )
  {
    v88 = 4 * v14;
    v15 = *(unsigned int *)(a1 + 360);
    if ( (unsigned int)(4 * v14) < 0x40 )
      v88 = 64;
    if ( (unsigned int)v15 <= v88 )
      goto LABEL_22;
    v89 = *(_QWORD **)(a1 + 344);
    v90 = v14 - 1;
    if ( v90 )
    {
      _BitScanReverse(&v90, v90);
      v91 = 1 << (33 - (v90 ^ 0x1F));
      if ( v91 < 64 )
        v91 = 64;
      if ( (_DWORD)v15 == v91 )
      {
        *(_QWORD *)(a1 + 352) = 0;
        v135 = &v89[2 * (unsigned int)v15];
        do
        {
          if ( v89 )
            *v89 = -8;
          v89 += 2;
        }
        while ( v135 != v89 );
        goto LABEL_25;
      }
      v92 = (((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)
          | (4 * v91 / 3u + 1)
          | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)
          | (((((4 * v91 / 3u + 1) | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 2)
            | (4 * v91 / 3u + 1)
            | ((unsigned __int64)(4 * v91 / 3u + 1) >> 1)) >> 4);
      v93 = (v92 >> 8) | v92;
      v94 = (v93 | (v93 >> 16)) + 1;
      v95 = 16 * ((v93 | (v93 >> 16)) + 1);
    }
    else
    {
      v95 = 2048;
      v94 = 128;
    }
    j___libc_free_0(v89);
    *(_DWORD *)(a1 + 360) = v94;
    v96 = (_QWORD *)sub_22077B0(v95);
    v97 = *(unsigned int *)(a1 + 360);
    *(_QWORD *)(a1 + 352) = 0;
    *(_QWORD *)(a1 + 344) = v96;
    for ( jj = &v96[2 * v97]; jj != v96; v96 += 2 )
    {
      if ( v96 )
        *v96 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 356) )
  {
    v15 = *(unsigned int *)(a1 + 360);
    if ( (unsigned int)v15 <= 0x40 )
    {
LABEL_22:
      v16 = *(_QWORD **)(a1 + 344);
      for ( kk = &v16[2 * v15]; kk != v16; v16 += 2 )
        *v16 = -8;
      *(_QWORD *)(a1 + 352) = 0;
      goto LABEL_25;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 344));
    *(_QWORD *)(a1 + 344) = 0;
    *(_QWORD *)(a1 + 352) = 0;
    *(_DWORD *)(a1 + 360) = 0;
  }
LABEL_25:
  v18 = *(_QWORD *)(a1 + 944);
  v19 = v18 + 40LL * *(unsigned int *)(a1 + 952);
  while ( v18 != v19 )
  {
    while ( 1 )
    {
      v19 -= 40;
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
      {
        v20 = *(_QWORD *)(v19 + 24);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
      if ( *(_DWORD *)(v19 + 16) <= 0x40u )
        break;
      v21 = *(_QWORD *)(v19 + 8);
      if ( !v21 )
        break;
      j_j___libc_free_0_0(v21);
      if ( v18 == v19 )
        goto LABEL_33;
    }
  }
LABEL_33:
  ++*(_QWORD *)(a1 + 832);
  v22 = *(void **)(a1 + 848);
  *(_DWORD *)(a1 + 952) = 0;
  if ( v22 == *(void **)(a1 + 840) )
    goto LABEL_38;
  v23 = 4 * (*(_DWORD *)(a1 + 860) - *(_DWORD *)(a1 + 864));
  v24 = *(unsigned int *)(a1 + 856);
  if ( v23 < 0x20 )
    v23 = 32;
  if ( (unsigned int)v24 <= v23 )
  {
    memset(v22, -1, 8 * v24);
LABEL_38:
    *(_QWORD *)(a1 + 860) = 0;
    goto LABEL_39;
  }
  sub_16CC920(a1 + 832);
LABEL_39:
  v25 = *(_DWORD *)(a1 + 384);
  ++*(_QWORD *)(a1 + 368);
  *(_DWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 496) = 0;
  if ( !v25 )
  {
    if ( !*(_DWORD *)(a1 + 388) )
      goto LABEL_45;
    v26 = *(unsigned int *)(a1 + 392);
    if ( (unsigned int)v26 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 376));
      *(_QWORD *)(a1 + 376) = 0;
      *(_QWORD *)(a1 + 384) = 0;
      *(_DWORD *)(a1 + 392) = 0;
      goto LABEL_45;
    }
    goto LABEL_42;
  }
  v77 = 4 * v25;
  v26 = *(unsigned int *)(a1 + 392);
  if ( (unsigned int)(4 * v25) < 0x40 )
    v77 = 64;
  if ( (unsigned int)v26 <= v77 )
  {
LABEL_42:
    v27 = *(_QWORD **)(a1 + 376);
    for ( mm = &v27[2 * v26]; mm != v27; v27 += 2 )
      *v27 = -8;
    *(_QWORD *)(a1 + 384) = 0;
    goto LABEL_45;
  }
  v78 = *(_QWORD **)(a1 + 376);
  v79 = v25 - 1;
  if ( !v79 )
  {
    v84 = 2048;
    v83 = 128;
LABEL_123:
    j___libc_free_0(v78);
    *(_DWORD *)(a1 + 392) = v83;
    v85 = (_QWORD *)sub_22077B0(v84);
    v86 = *(unsigned int *)(a1 + 392);
    *(_QWORD *)(a1 + 384) = 0;
    *(_QWORD *)(a1 + 376) = v85;
    for ( nn = &v85[2 * v86]; nn != v85; v85 += 2 )
    {
      if ( v85 )
        *v85 = -8;
    }
    goto LABEL_45;
  }
  _BitScanReverse(&v79, v79);
  v80 = 1 << (33 - (v79 ^ 0x1F));
  if ( v80 < 64 )
    v80 = 64;
  if ( (_DWORD)v26 != v80 )
  {
    v81 = (((4 * v80 / 3u + 1) | ((unsigned __int64)(4 * v80 / 3u + 1) >> 1)) >> 2)
        | (4 * v80 / 3u + 1)
        | ((unsigned __int64)(4 * v80 / 3u + 1) >> 1)
        | (((((4 * v80 / 3u + 1) | ((unsigned __int64)(4 * v80 / 3u + 1) >> 1)) >> 2)
          | (4 * v80 / 3u + 1)
          | ((unsigned __int64)(4 * v80 / 3u + 1) >> 1)) >> 4);
    v82 = (v81 >> 8) | v81;
    v83 = (v82 | (v82 >> 16)) + 1;
    v84 = 16 * ((v82 | (v82 >> 16)) + 1);
    goto LABEL_123;
  }
  *(_QWORD *)(a1 + 384) = 0;
  v134 = &v78[2 * (unsigned int)v26];
  do
  {
    if ( v78 )
      *v78 = -8;
    v78 += 2;
  }
  while ( v134 != v78 );
LABEL_45:
  v29 = *(_DWORD *)(a1 + 520);
  ++*(_QWORD *)(a1 + 504);
  if ( !v29 )
  {
    if ( !*(_DWORD *)(a1 + 524) )
      goto LABEL_51;
    v30 = *(unsigned int *)(a1 + 528);
    if ( (unsigned int)v30 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 512));
      *(_QWORD *)(a1 + 512) = 0;
      *(_QWORD *)(a1 + 520) = 0;
      *(_DWORD *)(a1 + 528) = 0;
      goto LABEL_51;
    }
    goto LABEL_48;
  }
  v66 = 4 * v29;
  v30 = *(unsigned int *)(a1 + 528);
  if ( (unsigned int)(4 * v29) < 0x40 )
    v66 = 64;
  if ( v66 >= (unsigned int)v30 )
  {
LABEL_48:
    v31 = *(_DWORD **)(a1 + 512);
    for ( i1 = &v31[2 * v30]; i1 != v31; v31 += 2 )
      *v31 = -1;
    *(_QWORD *)(a1 + 520) = 0;
    goto LABEL_51;
  }
  v67 = *(_DWORD **)(a1 + 512);
  v68 = v29 - 1;
  if ( !v68 )
  {
    v73 = 1024;
    v72 = 128;
LABEL_110:
    j___libc_free_0(v67);
    *(_DWORD *)(a1 + 528) = v72;
    v74 = (_DWORD *)sub_22077B0(v73);
    v75 = *(unsigned int *)(a1 + 528);
    *(_QWORD *)(a1 + 520) = 0;
    *(_QWORD *)(a1 + 512) = v74;
    for ( i2 = &v74[2 * v75]; i2 != v74; v74 += 2 )
    {
      if ( v74 )
        *v74 = -1;
    }
    goto LABEL_51;
  }
  _BitScanReverse(&v68, v68);
  v69 = 1 << (33 - (v68 ^ 0x1F));
  if ( v69 < 64 )
    v69 = 64;
  if ( (_DWORD)v30 != v69 )
  {
    v70 = (((4 * v69 / 3u + 1) | ((unsigned __int64)(4 * v69 / 3u + 1) >> 1)) >> 2)
        | (4 * v69 / 3u + 1)
        | ((unsigned __int64)(4 * v69 / 3u + 1) >> 1)
        | (((((4 * v69 / 3u + 1) | ((unsigned __int64)(4 * v69 / 3u + 1) >> 1)) >> 2)
          | (4 * v69 / 3u + 1)
          | ((unsigned __int64)(4 * v69 / 3u + 1) >> 1)) >> 4);
    v71 = (v70 >> 8) | v70;
    v72 = (v71 | (v71 >> 16)) + 1;
    v73 = 8 * ((v71 | (v71 >> 16)) + 1);
    goto LABEL_110;
  }
  *(_QWORD *)(a1 + 520) = 0;
  v133 = &v67[2 * v30];
  do
  {
    if ( v67 )
      *v67 = -1;
    v67 += 2;
  }
  while ( v133 != v67 );
LABEL_51:
  v33 = *(_DWORD *)(a1 + 552);
  ++*(_QWORD *)(a1 + 536);
  if ( !v33 )
  {
    if ( !*(_DWORD *)(a1 + 556) )
      goto LABEL_57;
    v34 = *(unsigned int *)(a1 + 560);
    if ( (unsigned int)v34 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 544));
      *(_QWORD *)(a1 + 544) = 0;
      *(_QWORD *)(a1 + 552) = 0;
      *(_DWORD *)(a1 + 560) = 0;
      goto LABEL_57;
    }
    goto LABEL_54;
  }
  v55 = 4 * v33;
  v34 = *(unsigned int *)(a1 + 560);
  if ( (unsigned int)(4 * v33) < 0x40 )
    v55 = 64;
  if ( (unsigned int)v34 <= v55 )
  {
LABEL_54:
    v35 = 4 * v34;
    if ( v35 )
      memset(*(void **)(a1 + 544), 255, v35);
    *(_QWORD *)(a1 + 552) = 0;
    goto LABEL_57;
  }
  v56 = *(_DWORD **)(a1 + 544);
  v57 = v33 - 1;
  if ( !v57 )
  {
    v62 = 512;
    v61 = 128;
LABEL_97:
    j___libc_free_0(v56);
    *(_DWORD *)(a1 + 560) = v61;
    v63 = (_DWORD *)sub_22077B0(v62);
    v64 = *(unsigned int *)(a1 + 560);
    *(_QWORD *)(a1 + 552) = 0;
    *(_QWORD *)(a1 + 544) = v63;
    for ( i3 = &v63[v64]; i3 != v63; ++v63 )
    {
      if ( v63 )
        *v63 = -1;
    }
    goto LABEL_57;
  }
  _BitScanReverse(&v57, v57);
  v58 = (unsigned int)(1 << (33 - (v57 ^ 0x1F)));
  if ( (int)v58 < 64 )
    v58 = 64;
  if ( (_DWORD)v58 != (_DWORD)v34 )
  {
    v59 = (((4 * (int)v58 / 3u + 1) | ((unsigned __int64)(4 * (int)v58 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v58 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v58 / 3u + 1) >> 1)
        | (((((4 * (int)v58 / 3u + 1) | ((unsigned __int64)(4 * (int)v58 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v58 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v58 / 3u + 1) >> 1)) >> 4);
    v60 = (v59 >> 8) | v59;
    v61 = (v60 | (v60 >> 16)) + 1;
    v62 = 4 * ((v60 | (v60 >> 16)) + 1);
    goto LABEL_97;
  }
  *(_QWORD *)(a1 + 552) = 0;
  v132 = &v56[v58];
  do
  {
    if ( v56 )
      *v56 = -1;
    ++v56;
  }
  while ( v132 != v56 );
LABEL_57:
  v36 = *(_DWORD *)(a1 + 320);
  ++*(_QWORD *)(a1 + 304);
  *(_DWORD *)(a1 + 576) = 0;
  if ( v36 || *(_DWORD *)(a1 + 324) )
  {
    v37 = 4 * v36;
    v38 = *(unsigned int *)(a1 + 328);
    if ( v37 < 0x40 )
      v37 = 64;
    if ( (unsigned int)v38 > v37 )
    {
      sub_1FDED60(a1 + 304);
    }
    else
    {
      v39 = *(_QWORD **)(a1 + 312);
      for ( i4 = &v39[9 * v38]; i4 != v39; v39 += 9 )
      {
        if ( *v39 != -8 )
        {
          if ( *v39 != -16 )
          {
            j___libc_free_0(v39[6]);
            j___libc_free_0(v39[2]);
          }
          *v39 = -8;
        }
      }
      *(_QWORD *)(a1 + 320) = 0;
    }
  }
  v41 = *(_DWORD *)(a1 + 816);
  ++*(_QWORD *)(a1 + 800);
  if ( !v41 )
  {
    result = *(unsigned int *)(a1 + 820);
    if ( !(_DWORD)result )
      return result;
    v43 = *(unsigned int *)(a1 + 824);
    if ( (unsigned int)v43 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 808));
      *(_QWORD *)(a1 + 808) = 0;
      *(_QWORD *)(a1 + 816) = 0;
      *(_DWORD *)(a1 + 824) = 0;
      return result;
    }
    goto LABEL_72;
  }
  v45 = 4 * v41;
  v43 = *(unsigned int *)(a1 + 824);
  if ( (unsigned int)(4 * v41) < 0x40 )
    v45 = 64;
  if ( v45 >= (unsigned int)v43 )
  {
LABEL_72:
    result = *(_QWORD *)(a1 + 808);
    for ( i5 = result + 16 * v43; i5 != result; result += 16 )
      *(_QWORD *)result = -8;
    *(_QWORD *)(a1 + 816) = 0;
    return result;
  }
  v46 = *(_QWORD **)(a1 + 808);
  v47 = v41 - 1;
  if ( !v47 )
  {
    v52 = 2048;
    v51 = 128;
LABEL_84:
    j___libc_free_0(v46);
    *(_DWORD *)(a1 + 824) = v51;
    result = sub_22077B0(v52);
    v53 = *(unsigned int *)(a1 + 824);
    *(_QWORD *)(a1 + 816) = 0;
    *(_QWORD *)(a1 + 808) = result;
    for ( i6 = result + 16 * v53; i6 != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
    return result;
  }
  _BitScanReverse(&v47, v47);
  v48 = (unsigned int)(1 << (33 - (v47 ^ 0x1F)));
  if ( (int)v48 < 64 )
    v48 = 64;
  if ( (_DWORD)v48 != (_DWORD)v43 )
  {
    v49 = (((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v48 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)
        | (((((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v48 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 4);
    v50 = (v49 >> 8) | v49;
    v51 = (v50 | (v50 >> 16)) + 1;
    v52 = 16 * ((v50 | (v50 >> 16)) + 1);
    goto LABEL_84;
  }
  *(_QWORD *)(a1 + 816) = 0;
  result = (__int64)&v46[2 * v48];
  do
  {
    if ( v46 )
      *v46 = -8;
    v46 += 2;
  }
  while ( (_QWORD *)result != v46 );
  return result;
}
