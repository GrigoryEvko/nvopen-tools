// Function: sub_19E9650
// Address: 0x19e9650
//
void __fastcall sub_19E9650(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rbx
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 *v25; // rcx
  __int64 v26; // rdx
  double v27; // xmm4_8
  double v28; // xmm5_8
  int v29; // eax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *i; // rdx
  __int64 v33; // r8
  _BYTE *v34; // r9
  unsigned int v35; // eax
  _QWORD *v36; // rdi
  __int64 v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rcx
  unsigned __int64 v40; // rdx
  __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rdx
  _QWORD *v44; // rax
  _QWORD *j; // rdx
  __int64 v46; // r14
  unsigned __int64 *v47; // r13
  unsigned __int64 *v48; // r14
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rax
  int v52; // edx
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  _QWORD *v55; // rax
  int v56; // eax
  __int64 v57; // rdx
  _QWORD *v58; // rax
  _QWORD *k; // rdx
  int v60; // eax
  __int64 v61; // rdx
  _QWORD *v62; // rax
  _QWORD *m; // rdx
  int v64; // r15d
  _QWORD *v65; // r13
  unsigned int v66; // eax
  __int64 v67; // rdx
  _QWORD *v68; // r14
  unsigned __int64 v69; // rdi
  int v70; // eax
  __int64 v71; // rdx
  _QWORD *v72; // rax
  _QWORD *n; // rdx
  int v74; // eax
  __int64 v75; // rdx
  _QWORD *v76; // rax
  _QWORD *ii; // rdx
  int v78; // eax
  __int64 v79; // rdx
  _QWORD *v80; // rax
  _QWORD *jj; // rdx
  int v82; // eax
  __int64 v83; // rdx
  _QWORD *v84; // rax
  _QWORD *kk; // rdx
  int v86; // eax
  __int64 v87; // rdx
  _QWORD *v88; // rax
  _QWORD *mm; // rdx
  int v90; // eax
  __int64 v91; // rdx
  _QWORD *v92; // rax
  _QWORD *nn; // rdx
  int v94; // eax
  __int64 v95; // rdx
  _QWORD *v96; // rax
  _QWORD *i1; // rdx
  int v98; // r15d
  _QWORD *v99; // r13
  unsigned int v100; // eax
  __int64 v101; // rdx
  _QWORD *v102; // r14
  unsigned __int64 v103; // rdi
  int v104; // r15d
  _QWORD *v105; // r13
  unsigned int v106; // eax
  __int64 v107; // rdx
  _QWORD *v108; // r14
  unsigned __int64 v109; // rdi
  int v110; // edx
  _QWORD **v111; // r13
  __int64 v112; // rcx
  _QWORD **v113; // r15
  unsigned int v114; // eax
  _QWORD **v115; // r14
  __int64 v116; // rax
  _QWORD *v117; // r13
  _QWORD *v118; // rdi
  int v119; // esi
  unsigned int v120; // ecx
  unsigned int v121; // eax
  int v122; // r14d
  unsigned int v123; // eax
  unsigned int v124; // ecx
  unsigned int v125; // eax
  int v126; // r14d
  unsigned int v127; // eax
  unsigned int v128; // ecx
  unsigned int v129; // eax
  int v130; // r14d
  unsigned int v131; // eax
  unsigned int v132; // ecx
  unsigned int v133; // eax
  int v134; // r14d
  unsigned int v135; // eax
  unsigned int v136; // ecx
  unsigned int v137; // eax
  int v138; // r14d
  unsigned int v139; // eax
  unsigned int v140; // ecx
  __int64 v141; // rsi
  unsigned int v142; // eax
  int v143; // r14d
  unsigned int v144; // eax
  unsigned int v145; // ecx
  unsigned int v146; // eax
  int v147; // eax
  unsigned __int64 v148; // r14
  unsigned int v149; // eax
  unsigned int v150; // ecx
  unsigned int v151; // eax
  int v152; // r14d
  unsigned int v153; // eax
  unsigned int v154; // ecx
  unsigned int v155; // eax
  int v156; // r14d
  unsigned int v157; // eax
  unsigned int v158; // ecx
  unsigned int v159; // eax
  int v160; // r14d
  unsigned int v161; // eax
  unsigned int v162; // ecx
  unsigned int v163; // eax
  int v164; // r14d
  unsigned int v165; // eax
  unsigned __int64 v166; // rdi
  unsigned __int64 v167; // rdi
  unsigned __int64 v168; // rdi
  unsigned int v169; // ecx
  unsigned int v170; // eax
  int v171; // eax
  unsigned __int64 v172; // r14
  unsigned int v173; // eax
  _QWORD *v174; // r13
  __int64 v175; // rdx
  unsigned __int64 *v176; // r14
  unsigned __int64 *v177; // r13
  unsigned __int64 v178; // rdi
  int v179; // edx
  int v180; // r14d
  unsigned int v181; // eax
  unsigned int v182; // eax
  int v183; // edx
  int v184; // r14d
  unsigned int v185; // eax
  unsigned int v186; // eax
  int v187; // edx
  int v188; // r14d
  unsigned int v189; // eax
  unsigned int v190; // eax
  _QWORD **i2; // r13
  __int64 v192; // rax
  _QWORD *i3; // r14
  _QWORD *v194; // rdi
  int v195; // r13d
  unsigned int v196; // eax
  unsigned int v197; // eax
  int v198; // [rsp+Ch] [rbp-C4h]
  _BYTE *v199; // [rsp+50h] [rbp-80h] BYREF
  __int64 v200; // [rsp+58h] [rbp-78h]
  _BYTE v201[112]; // [rsp+60h] [rbp-70h] BYREF

  v15 = *(_QWORD *)(a1 + 1440);
  v16 = (*(_QWORD *)(a1 + 1448) - v15) >> 3;
  if ( (_DWORD)v16 )
  {
    v17 = 0;
    v18 = 8LL * (unsigned int)(v16 - 1);
    while ( 1 )
    {
      v19 = (_QWORD *)(v17 + v15);
      v20 = (_QWORD *)*v19;
      if ( *v19 )
      {
        v21 = v20[18];
        if ( v21 != v20[17] )
          _libc_free(v21);
        v22 = v20[9];
        if ( v22 != v20[8] )
          _libc_free(v22);
        j_j___libc_free_0(v20, 192);
        v19 = (_QWORD *)(v17 + *(_QWORD *)(a1 + 1440));
      }
      *v19 = 0;
      if ( v17 == v18 )
        break;
      v15 = *(_QWORD *)(a1 + 1440);
      v17 += 8;
    }
  }
  v23 = *(__int64 **)(a1 + 1840);
  v24 = *(_QWORD *)(a1 + 1832);
  v25 = &v23[*(unsigned int *)(a1 + 1856)];
  v26 = *(unsigned int *)(a1 + 1848);
  if ( (_DWORD)v26 )
  {
    for ( ; v25 != v23; ++v23 )
    {
      v26 = *v23;
      if ( *v23 != -8 && v26 != -16 )
        break;
    }
  }
  else
  {
    v23 += *(unsigned int *)(a1 + 1856);
  }
  v200 = 0x800000000LL;
  v199 = v201;
  sub_19E7570((__int64)&v199, v24, v26, (__int64)v25, a13, a14, a1 + 1832, v24, v23, v25, a1 + 1832, v24, v25);
  v29 = *(_DWORD *)(a1 + 1848);
  ++*(_QWORD *)(a1 + 1832);
  if ( v29 )
  {
    v136 = 4 * v29;
    v24 = 64;
    v30 = *(unsigned int *)(a1 + 1856);
    if ( (unsigned int)(4 * v29) < 0x40 )
      v136 = 64;
    if ( v136 >= (unsigned int)v30 )
      goto LABEL_16;
    v137 = v29 - 1;
    if ( v137 )
    {
      _BitScanReverse(&v137, v137);
      v138 = 1 << (33 - (v137 ^ 0x1F));
      if ( v138 < 64 )
        v138 = 64;
      if ( (_DWORD)v30 == v138 )
        goto LABEL_214;
    }
    else
    {
      v138 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1840));
    v139 = sub_19E3C70(v138);
    *(_DWORD *)(a1 + 1856) = v139;
    if ( !v139 )
      goto LABEL_377;
    *(_QWORD *)(a1 + 1840) = sub_22077B0(8LL * v139);
LABEL_214:
    sub_19E95D0(a1 + 1832);
    goto LABEL_19;
  }
  if ( *(_DWORD *)(a1 + 1852) )
  {
    v30 = *(unsigned int *)(a1 + 1856);
    if ( (unsigned int)v30 <= 0x40 )
    {
LABEL_16:
      v31 = *(_QWORD **)(a1 + 1840);
      for ( i = &v31[v30]; i != v31; ++v31 )
        *v31 = -8;
      goto LABEL_18;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1840));
    *(_DWORD *)(a1 + 1856) = 0;
LABEL_377:
    *(_QWORD *)(a1 + 1840) = 0;
LABEL_18:
    *(_QWORD *)(a1 + 1848) = 0;
  }
LABEL_19:
  v33 = (__int64)v199;
  v34 = &v199[8 * (unsigned int)v200];
  v35 = v200;
  if ( v199 == v34 )
    goto LABEL_31;
  do
  {
    v36 = *(_QWORD **)v33;
    v37 = 24LL * (*(_DWORD *)(*(_QWORD *)v33 + 20LL) & 0xFFFFFFF);
    if ( (*(_BYTE *)(*(_QWORD *)v33 + 23LL) & 0x40) != 0 )
    {
      v38 = (_QWORD *)*(v36 - 1);
      v36 = &v38[(unsigned __int64)v37 / 8];
    }
    else
    {
      v38 = &v36[v37 / 0xFFFFFFFFFFFFFFF8LL];
    }
    for ( ; v36 != v38; v38 += 3 )
    {
      if ( *v38 )
      {
        v39 = v38[1];
        v40 = v38[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v40 = v39;
        if ( v39 )
        {
          v24 = *(_QWORD *)(v39 + 16) & 3LL;
          *(_QWORD *)(v39 + 16) = v24 | v40;
        }
      }
      *v38 = 0;
    }
    v33 += 8;
  }
  while ( (_BYTE *)v33 != v34 );
  while ( 1 )
  {
    v35 = v200;
LABEL_31:
    if ( !v35 )
      break;
    v41 = *(_QWORD *)&v199[8 * v35 - 8];
    LODWORD(v200) = v35 - 1;
    sub_164BEC0(v41, v24, (__int64)v199, v35, a2, a3, a4, a5, v27, v28, a8, a9);
  }
  v42 = *(_DWORD *)(a1 + 1488);
  ++*(_QWORD *)(a1 + 1472);
  if ( !v42 )
  {
    if ( !*(_DWORD *)(a1 + 1492) )
      goto LABEL_38;
    v43 = *(unsigned int *)(a1 + 1496);
    if ( (unsigned int)v43 <= 0x40 )
      goto LABEL_35;
    j___libc_free_0(*(_QWORD *)(a1 + 1480));
    *(_DWORD *)(a1 + 1496) = 0;
LABEL_370:
    *(_QWORD *)(a1 + 1480) = 0;
LABEL_37:
    *(_QWORD *)(a1 + 1488) = 0;
    goto LABEL_38;
  }
  v145 = 4 * v42;
  v43 = *(unsigned int *)(a1 + 1496);
  if ( (unsigned int)(4 * v42) < 0x40 )
    v145 = 64;
  if ( v145 >= (unsigned int)v43 )
  {
LABEL_35:
    v44 = *(_QWORD **)(a1 + 1480);
    for ( j = &v44[2 * v43]; j != v44; v44 += 2 )
      *v44 = -8;
    goto LABEL_37;
  }
  v146 = v42 - 1;
  if ( v146 )
  {
    _BitScanReverse(&v146, v146);
    v147 = 1 << (33 - (v146 ^ 0x1F));
    if ( v147 < 64 )
      v147 = 64;
    if ( (_DWORD)v43 == v147 )
      goto LABEL_238;
    v148 = 4 * v147 / 3u + 1;
  }
  else
  {
    v148 = 86;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1480));
  v149 = sub_1454B60(v148);
  *(_DWORD *)(a1 + 1496) = v149;
  if ( !v149 )
    goto LABEL_370;
  *(_QWORD *)(a1 + 1480) = sub_22077B0(16LL * v149);
LABEL_238:
  sub_19E0CD0(a1 + 1472);
LABEL_38:
  v46 = *(unsigned int *)(a1 + 136);
  v47 = *(unsigned __int64 **)(a1 + 128);
  *(_DWORD *)(a1 + 176) = 0;
  v48 = &v47[2 * v46];
  while ( v47 != v48 )
  {
    v49 = *v47;
    v47 += 2;
    _libc_free(v49);
  }
  v50 = *(unsigned int *)(a1 + 88);
  *(_DWORD *)(a1 + 136) = 0;
  if ( (_DWORD)v50 )
  {
    v174 = *(_QWORD **)(a1 + 80);
    *(_QWORD *)(a1 + 144) = 0;
    v175 = *v174;
    v176 = &v174[v50];
    v177 = v174 + 1;
    *(_QWORD *)(a1 + 64) = v175;
    *(_QWORD *)(a1 + 72) = v175 + 4096;
    while ( v176 != v177 )
    {
      v178 = *v177++;
      _libc_free(v178);
    }
    *(_DWORD *)(a1 + 88) = 1;
  }
  v51 = *(_QWORD *)(a1 + 1440);
  if ( v51 != *(_QWORD *)(a1 + 1448) )
    *(_QWORD *)(a1 + 1448) = v51;
  v52 = *(_DWORD *)(a1 + 2072);
  ++*(_QWORD *)(a1 + 2056);
  if ( !v52 )
  {
    if ( !*(_DWORD *)(a1 + 2076) )
      goto LABEL_48;
    v53 = *(_QWORD **)(a1 + 2064);
    v54 = *(unsigned int *)(a1 + 2080);
    v55 = &v53[2 * v54];
    if ( (unsigned int)v54 > 0x40 )
    {
      j___libc_free_0(v53);
      *(_DWORD *)(a1 + 2080) = 0;
LABEL_47:
      *(_QWORD *)(a1 + 2064) = 0;
      *(_QWORD *)(a1 + 2072) = 0;
      goto LABEL_48;
    }
    goto LABEL_226;
  }
  v53 = *(_QWORD **)(a1 + 2064);
  v140 = 4 * v52;
  v141 = *(unsigned int *)(a1 + 2080);
  v55 = &v53[2 * v141];
  if ( (unsigned int)(4 * v52) < 0x40 )
    v140 = 64;
  if ( (unsigned int)v141 > v140 )
  {
    if ( v52 == 1 )
    {
      v143 = 64;
    }
    else
    {
      _BitScanReverse(&v142, v52 - 1);
      v143 = 1 << (33 - (v142 ^ 0x1F));
      if ( v143 < 64 )
        v143 = 64;
      if ( (_DWORD)v141 == v143 )
      {
LABEL_224:
        sub_19E0D50(a1 + 2056);
        goto LABEL_48;
      }
    }
    j___libc_free_0(v53);
    v144 = sub_19E3C70(v143);
    *(_DWORD *)(a1 + 2080) = v144;
    if ( !v144 )
      goto LABEL_47;
    *(_QWORD *)(a1 + 2064) = sub_22077B0(16LL * v144);
    goto LABEL_224;
  }
LABEL_226:
  while ( v53 != v55 )
  {
    *v53 = -8;
    v53 += 2;
  }
  *(_QWORD *)(a1 + 2072) = 0;
LABEL_48:
  v56 = *(_DWORD *)(a1 + 1520);
  ++*(_QWORD *)(a1 + 1504);
  if ( !v56 )
  {
    if ( !*(_DWORD *)(a1 + 1524) )
      goto LABEL_54;
    v57 = *(unsigned int *)(a1 + 1528);
    if ( (unsigned int)v57 <= 0x40 )
      goto LABEL_51;
    j___libc_free_0(*(_QWORD *)(a1 + 1512));
    *(_DWORD *)(a1 + 1528) = 0;
LABEL_368:
    *(_QWORD *)(a1 + 1512) = 0;
LABEL_53:
    *(_QWORD *)(a1 + 1520) = 0;
    goto LABEL_54;
  }
  v162 = 4 * v56;
  v57 = *(unsigned int *)(a1 + 1528);
  if ( (unsigned int)(4 * v56) < 0x40 )
    v162 = 64;
  if ( (unsigned int)v57 <= v162 )
  {
LABEL_51:
    v58 = *(_QWORD **)(a1 + 1512);
    for ( k = &v58[2 * v57]; k != v58; v58 += 2 )
      *v58 = -8;
    goto LABEL_53;
  }
  v163 = v56 - 1;
  if ( v163 )
  {
    _BitScanReverse(&v163, v163);
    v164 = 1 << (33 - (v163 ^ 0x1F));
    if ( v164 < 64 )
      v164 = 64;
    if ( (_DWORD)v57 == v164 )
      goto LABEL_278;
  }
  else
  {
    v164 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1512));
  v165 = sub_19E3C70(v164);
  *(_DWORD *)(a1 + 1528) = v165;
  if ( !v165 )
    goto LABEL_368;
  *(_QWORD *)(a1 + 1512) = sub_22077B0(16LL * v165);
LABEL_278:
  sub_19E8CB0(a1 + 1504);
LABEL_54:
  v60 = *(_DWORD *)(a1 + 1720);
  ++*(_QWORD *)(a1 + 1704);
  if ( v60 )
  {
    v158 = 4 * v60;
    v61 = *(unsigned int *)(a1 + 1728);
    if ( (unsigned int)(4 * v60) < 0x40 )
      v158 = 64;
    if ( v158 >= (unsigned int)v61 )
      goto LABEL_57;
    v159 = v60 - 1;
    if ( v159 )
    {
      _BitScanReverse(&v159, v159);
      v160 = 1 << (33 - (v159 ^ 0x1F));
      if ( v160 < 64 )
        v160 = 64;
      if ( (_DWORD)v61 == v160 )
        goto LABEL_268;
    }
    else
    {
      v160 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1712));
    v161 = sub_19E3C70(v160);
    *(_DWORD *)(a1 + 1728) = v161;
    if ( !v161 )
      goto LABEL_375;
    *(_QWORD *)(a1 + 1712) = sub_22077B0(16LL * v161);
LABEL_268:
    sub_19E8D70(a1 + 1704);
    goto LABEL_60;
  }
  if ( *(_DWORD *)(a1 + 1724) )
  {
    v61 = *(unsigned int *)(a1 + 1728);
    if ( (unsigned int)v61 <= 0x40 )
    {
LABEL_57:
      v62 = *(_QWORD **)(a1 + 1712);
      for ( m = &v62[2 * v61]; m != v62; v62 += 2 )
        *v62 = -8;
      goto LABEL_59;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1712));
    *(_DWORD *)(a1 + 1728) = 0;
LABEL_375:
    *(_QWORD *)(a1 + 1712) = 0;
LABEL_59:
    *(_QWORD *)(a1 + 1720) = 0;
  }
LABEL_60:
  v64 = *(_DWORD *)(a1 + 1752);
  ++*(_QWORD *)(a1 + 1736);
  if ( v64 || *(_DWORD *)(a1 + 1756) )
  {
    v65 = *(_QWORD **)(a1 + 1744);
    v66 = 4 * v64;
    v67 = *(unsigned int *)(a1 + 1760);
    v68 = &v65[8 * v67];
    if ( (unsigned int)(4 * v64) < 0x40 )
      v66 = 64;
    if ( v66 >= (unsigned int)v67 )
    {
      for ( ; v65 != v68; v65 += 8 )
      {
        if ( *v65 != -8 )
        {
          if ( *v65 != -16 )
          {
            v69 = v65[3];
            if ( v69 != v65[2] )
              _libc_free(v69);
          }
          *v65 = -8;
        }
      }
      goto LABEL_72;
    }
    do
    {
      if ( *v65 != -8 && *v65 != -16 )
      {
        v166 = v65[3];
        if ( v166 != v65[2] )
          _libc_free(v166);
      }
      v65 += 8;
    }
    while ( v65 != v68 );
    v179 = *(_DWORD *)(a1 + 1760);
    if ( v64 )
    {
      v180 = 64;
      if ( v64 != 1 )
      {
        _BitScanReverse(&v181, v64 - 1);
        v180 = 1 << (33 - (v181 ^ 0x1F));
        if ( v180 < 64 )
          v180 = 64;
      }
      if ( v180 == v179 )
        goto LABEL_323;
      j___libc_free_0(*(_QWORD *)(a1 + 1744));
      v182 = sub_19E3C70(v180);
      *(_DWORD *)(a1 + 1760) = v182;
      if ( v182 )
      {
        *(_QWORD *)(a1 + 1744) = sub_22077B0((unsigned __int64)v182 << 6);
LABEL_323:
        sub_19E8DB0(a1 + 1736);
        goto LABEL_73;
      }
    }
    else
    {
      if ( !v179 )
        goto LABEL_323;
      j___libc_free_0(*(_QWORD *)(a1 + 1744));
      *(_DWORD *)(a1 + 1760) = 0;
    }
    *(_QWORD *)(a1 + 1744) = 0;
LABEL_72:
    *(_QWORD *)(a1 + 1752) = 0;
  }
LABEL_73:
  sub_19E69C0(a1 + 1768);
  v70 = *(_DWORD *)(a1 + 1688);
  ++*(_QWORD *)(a1 + 1672);
  if ( !v70 )
  {
    if ( !*(_DWORD *)(a1 + 1692) )
      goto LABEL_79;
    v71 = *(unsigned int *)(a1 + 1696);
    if ( (unsigned int)v71 <= 0x40 )
      goto LABEL_76;
    j___libc_free_0(*(_QWORD *)(a1 + 1680));
    *(_DWORD *)(a1 + 1696) = 0;
LABEL_352:
    *(_QWORD *)(a1 + 1680) = 0;
LABEL_78:
    *(_QWORD *)(a1 + 1688) = 0;
    goto LABEL_79;
  }
  v154 = 4 * v70;
  v71 = *(unsigned int *)(a1 + 1696);
  if ( (unsigned int)(4 * v70) < 0x40 )
    v154 = 64;
  if ( (unsigned int)v71 <= v154 )
  {
LABEL_76:
    v72 = *(_QWORD **)(a1 + 1680);
    for ( n = &v72[2 * v71]; n != v72; v72 += 2 )
      *v72 = -8;
    goto LABEL_78;
  }
  v155 = v70 - 1;
  if ( v155 )
  {
    _BitScanReverse(&v155, v155);
    v156 = 1 << (33 - (v155 ^ 0x1F));
    if ( v156 < 64 )
      v156 = 64;
    if ( (_DWORD)v71 == v156 )
      goto LABEL_258;
  }
  else
  {
    v156 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1680));
  v157 = sub_19E3C70(v156);
  *(_DWORD *)(a1 + 1696) = v157;
  if ( !v157 )
    goto LABEL_352;
  *(_QWORD *)(a1 + 1680) = sub_22077B0(16LL * v157);
LABEL_258:
  sub_19E8D30(a1 + 1672);
LABEL_79:
  v74 = *(_DWORD *)(a1 + 1816);
  ++*(_QWORD *)(a1 + 1800);
  if ( !v74 )
  {
    if ( !*(_DWORD *)(a1 + 1820) )
      goto LABEL_85;
    v75 = *(unsigned int *)(a1 + 1824);
    if ( (unsigned int)v75 <= 0x40 )
      goto LABEL_82;
    j___libc_free_0(*(_QWORD *)(a1 + 1808));
    *(_DWORD *)(a1 + 1824) = 0;
LABEL_350:
    *(_QWORD *)(a1 + 1808) = 0;
LABEL_84:
    *(_QWORD *)(a1 + 1816) = 0;
    goto LABEL_85;
  }
  v150 = 4 * v74;
  v75 = *(unsigned int *)(a1 + 1824);
  if ( (unsigned int)(4 * v74) < 0x40 )
    v150 = 64;
  if ( (unsigned int)v75 <= v150 )
  {
LABEL_82:
    v76 = *(_QWORD **)(a1 + 1808);
    for ( ii = &v76[2 * v75]; ii != v76; v76 += 2 )
      *v76 = -8;
    goto LABEL_84;
  }
  v151 = v74 - 1;
  if ( v151 )
  {
    _BitScanReverse(&v151, v151);
    v152 = 1 << (33 - (v151 ^ 0x1F));
    if ( v152 < 64 )
      v152 = 64;
    if ( (_DWORD)v75 == v152 )
      goto LABEL_248;
  }
  else
  {
    v152 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1808));
  v153 = sub_19E3C70(v152);
  *(_DWORD *)(a1 + 1824) = v153;
  if ( !v153 )
    goto LABEL_350;
  *(_QWORD *)(a1 + 1808) = sub_22077B0(16LL * v153);
LABEL_248:
  sub_19E8DF0(a1 + 1800);
LABEL_85:
  sub_18CE100(a1 + 1536);
  v78 = *(_DWORD *)(a1 + 1656);
  ++*(_QWORD *)(a1 + 1640);
  if ( !v78 )
  {
    if ( !*(_DWORD *)(a1 + 1660) )
      goto LABEL_91;
    v79 = *(unsigned int *)(a1 + 1664);
    if ( (unsigned int)v79 <= 0x40 )
      goto LABEL_88;
    j___libc_free_0(*(_QWORD *)(a1 + 1648));
    *(_DWORD *)(a1 + 1664) = 0;
LABEL_348:
    *(_QWORD *)(a1 + 1648) = 0;
LABEL_90:
    *(_QWORD *)(a1 + 1656) = 0;
    goto LABEL_91;
  }
  v132 = 4 * v78;
  v79 = *(unsigned int *)(a1 + 1664);
  if ( (unsigned int)(4 * v78) < 0x40 )
    v132 = 64;
  if ( (unsigned int)v79 <= v132 )
  {
LABEL_88:
    v80 = *(_QWORD **)(a1 + 1648);
    for ( jj = &v80[2 * v79]; jj != v80; v80 += 2 )
      *v80 = -8;
    goto LABEL_90;
  }
  v133 = v78 - 1;
  if ( v133 )
  {
    _BitScanReverse(&v133, v133);
    v134 = 1 << (33 - (v133 ^ 0x1F));
    if ( v134 < 64 )
      v134 = 64;
    if ( (_DWORD)v79 == v134 )
      goto LABEL_204;
  }
  else
  {
    v134 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1648));
  v135 = sub_19E3C70(v134);
  *(_DWORD *)(a1 + 1664) = v135;
  if ( !v135 )
    goto LABEL_348;
  *(_QWORD *)(a1 + 1648) = sub_22077B0(16LL * v135);
LABEL_204:
  sub_19E8CF0(a1 + 1640);
LABEL_91:
  sub_18CE100(a1 + 2232);
  v82 = *(_DWORD *)(a1 + 2216);
  ++*(_QWORD *)(a1 + 2200);
  if ( !v82 )
  {
    if ( !*(_DWORD *)(a1 + 2220) )
      goto LABEL_97;
    v83 = *(unsigned int *)(a1 + 2224);
    if ( (unsigned int)v83 <= 0x40 )
      goto LABEL_94;
    j___libc_free_0(*(_QWORD *)(a1 + 2208));
    *(_DWORD *)(a1 + 2224) = 0;
LABEL_379:
    *(_QWORD *)(a1 + 2208) = 0;
LABEL_96:
    *(_QWORD *)(a1 + 2216) = 0;
    goto LABEL_97;
  }
  v128 = 4 * v82;
  v83 = *(unsigned int *)(a1 + 2224);
  if ( (unsigned int)(4 * v82) < 0x40 )
    v128 = 64;
  if ( (unsigned int)v83 <= v128 )
  {
LABEL_94:
    v84 = *(_QWORD **)(a1 + 2208);
    for ( kk = &v84[2 * v83]; kk != v84; *(v84 - 1) = -8 )
    {
      *v84 = -8;
      v84 += 2;
    }
    goto LABEL_96;
  }
  v129 = v82 - 1;
  if ( v129 )
  {
    _BitScanReverse(&v129, v129);
    v130 = 1 << (33 - (v129 ^ 0x1F));
    if ( v130 < 64 )
      v130 = 64;
    if ( (_DWORD)v83 == v130 )
      goto LABEL_194;
  }
  else
  {
    v130 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2208));
  v131 = sub_19E3C70(v130);
  *(_DWORD *)(a1 + 2224) = v131;
  if ( !v131 )
    goto LABEL_379;
  *(_QWORD *)(a1 + 2208) = sub_22077B0(16LL * v131);
LABEL_194:
  sub_19E9610(a1 + 2200);
LABEL_97:
  v86 = *(_DWORD *)(a1 + 2408);
  ++*(_QWORD *)(a1 + 2392);
  if ( !v86 )
  {
    if ( !*(_DWORD *)(a1 + 2412) )
      goto LABEL_103;
    v87 = *(unsigned int *)(a1 + 2416);
    if ( (unsigned int)v87 <= 0x40 )
      goto LABEL_100;
    j___libc_free_0(*(_QWORD *)(a1 + 2400));
    *(_DWORD *)(a1 + 2416) = 0;
LABEL_356:
    *(_QWORD *)(a1 + 2400) = 0;
LABEL_102:
    *(_QWORD *)(a1 + 2408) = 0;
    goto LABEL_103;
  }
  v124 = 4 * v86;
  v87 = *(unsigned int *)(a1 + 2416);
  if ( (unsigned int)(4 * v86) < 0x40 )
    v124 = 64;
  if ( (unsigned int)v87 <= v124 )
  {
LABEL_100:
    v88 = *(_QWORD **)(a1 + 2400);
    for ( mm = &v88[2 * v87]; mm != v88; v88 += 2 )
      *v88 = -8;
    goto LABEL_102;
  }
  v125 = v86 - 1;
  if ( v125 )
  {
    _BitScanReverse(&v125, v125);
    v126 = 1 << (33 - (v125 ^ 0x1F));
    if ( v126 < 64 )
      v126 = 64;
    if ( (_DWORD)v87 == v126 )
      goto LABEL_184;
  }
  else
  {
    v126 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2400));
  v127 = sub_19E3C70(v126);
  *(_DWORD *)(a1 + 2416) = v127;
  if ( !v127 )
    goto LABEL_356;
  *(_QWORD *)(a1 + 2400) = sub_22077B0(16LL * v127);
LABEL_184:
  sub_19E8C70(a1 + 2392);
LABEL_103:
  sub_18CE100(a1 + 2696);
  v90 = *(_DWORD *)(a1 + 2376);
  ++*(_QWORD *)(a1 + 2360);
  *(_DWORD *)(a1 + 2432) = 0;
  if ( !v90 )
  {
    if ( !*(_DWORD *)(a1 + 2380) )
      goto LABEL_109;
    v91 = *(unsigned int *)(a1 + 2384);
    if ( (unsigned int)v91 <= 0x40 )
      goto LABEL_106;
    j___libc_free_0(*(_QWORD *)(a1 + 2368));
    *(_DWORD *)(a1 + 2384) = 0;
LABEL_354:
    *(_QWORD *)(a1 + 2368) = 0;
LABEL_108:
    *(_QWORD *)(a1 + 2376) = 0;
    goto LABEL_109;
  }
  v120 = 4 * v90;
  v91 = *(unsigned int *)(a1 + 2384);
  if ( (unsigned int)(4 * v90) < 0x40 )
    v120 = 64;
  if ( (unsigned int)v91 <= v120 )
  {
LABEL_106:
    v92 = *(_QWORD **)(a1 + 2368);
    for ( nn = &v92[2 * v91]; nn != v92; v92 += 2 )
      *v92 = -8;
    goto LABEL_108;
  }
  v121 = v90 - 1;
  if ( v121 )
  {
    _BitScanReverse(&v121, v121);
    v122 = 1 << (33 - (v121 ^ 0x1F));
    if ( v122 < 64 )
      v122 = 64;
    if ( v122 == (_DWORD)v91 )
      goto LABEL_174;
  }
  else
  {
    v122 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2368));
  v123 = sub_19E3C70(v122);
  *(_DWORD *)(a1 + 2384) = v123;
  if ( !v123 )
    goto LABEL_354;
  *(_QWORD *)(a1 + 2368) = sub_22077B0(16LL * v123);
LABEL_174:
  sub_19E8EF0(a1 + 2360);
LABEL_109:
  v94 = *(_DWORD *)(a1 + 1976);
  ++*(_QWORD *)(a1 + 1960);
  *(_DWORD *)(a1 + 2352) = 0;
  if ( v94 )
  {
    v169 = 4 * v94;
    v95 = *(unsigned int *)(a1 + 1984);
    if ( (unsigned int)(4 * v94) < 0x40 )
      v169 = 64;
    if ( v169 >= (unsigned int)v95 )
      goto LABEL_112;
    v170 = v94 - 1;
    if ( v170 )
    {
      _BitScanReverse(&v170, v170);
      v171 = 1 << (33 - (v170 ^ 0x1F));
      if ( v171 < 64 )
        v171 = 64;
      if ( (_DWORD)v95 == v171 )
        goto LABEL_307;
      v172 = 4 * v171 / 3u + 1;
    }
    else
    {
      v172 = 86;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1968));
    v173 = sub_1454B60(v172);
    *(_DWORD *)(a1 + 1984) = v173;
    if ( !v173 )
      goto LABEL_384;
    *(_QWORD *)(a1 + 1968) = sub_22077B0(16LL * v173);
LABEL_307:
    sub_19E0D10(a1 + 1960);
    goto LABEL_115;
  }
  if ( *(_DWORD *)(a1 + 1980) )
  {
    v95 = *(unsigned int *)(a1 + 1984);
    if ( (unsigned int)v95 <= 0x40 )
    {
LABEL_112:
      v96 = *(_QWORD **)(a1 + 1968);
      for ( i1 = &v96[2 * v95]; i1 != v96; v96 += 2 )
        *v96 = -8;
      goto LABEL_114;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 1968));
    *(_DWORD *)(a1 + 1984) = 0;
LABEL_384:
    *(_QWORD *)(a1 + 1968) = 0;
LABEL_114:
    *(_QWORD *)(a1 + 1976) = 0;
  }
LABEL_115:
  v98 = *(_DWORD *)(a1 + 1912);
  ++*(_QWORD *)(a1 + 1896);
  if ( v98 || *(_DWORD *)(a1 + 1916) )
  {
    v99 = *(_QWORD **)(a1 + 1904);
    v100 = 4 * v98;
    v101 = *(unsigned int *)(a1 + 1920);
    v102 = &v99[8 * v101];
    if ( (unsigned int)(4 * v98) < 0x40 )
      v100 = 64;
    if ( (unsigned int)v101 <= v100 )
    {
      while ( v99 != v102 )
      {
        if ( *v99 != -8 )
        {
          if ( *v99 != -16 )
          {
            v103 = v99[3];
            if ( v103 != v99[2] )
              _libc_free(v103);
          }
          *v99 = -8;
        }
        v99 += 8;
      }
    }
    else
    {
      do
      {
        if ( *v99 != -8 && *v99 != -16 )
        {
          v168 = v99[3];
          if ( v168 != v99[2] )
            _libc_free(v168);
        }
        v99 += 8;
      }
      while ( v99 != v102 );
      v183 = *(_DWORD *)(a1 + 1920);
      if ( v98 )
      {
        v184 = 64;
        if ( v98 != 1 )
        {
          _BitScanReverse(&v185, v98 - 1);
          v184 = 1 << (33 - (v185 ^ 0x1F));
          if ( v184 < 64 )
            v184 = 64;
        }
        if ( v184 == v183 )
          goto LABEL_331;
        j___libc_free_0(*(_QWORD *)(a1 + 1904));
        v186 = sub_19E3C70(v184);
        *(_DWORD *)(a1 + 1920) = v186;
        if ( v186 )
        {
          *(_QWORD *)(a1 + 1904) = sub_22077B0((unsigned __int64)v186 << 6);
LABEL_331:
          sub_19E8E70(a1 + 1896);
          goto LABEL_129;
        }
      }
      else
      {
        if ( !v183 )
          goto LABEL_331;
        j___libc_free_0(*(_QWORD *)(a1 + 1904));
        *(_DWORD *)(a1 + 1920) = 0;
      }
      *(_QWORD *)(a1 + 1904) = 0;
    }
    *(_QWORD *)(a1 + 1912) = 0;
  }
LABEL_129:
  v104 = *(_DWORD *)(a1 + 1944);
  ++*(_QWORD *)(a1 + 1928);
  if ( v104 || *(_DWORD *)(a1 + 1948) )
  {
    v105 = *(_QWORD **)(a1 + 1936);
    v106 = 4 * v104;
    v107 = *(unsigned int *)(a1 + 1952);
    v108 = &v105[8 * v107];
    if ( (unsigned int)(4 * v104) < 0x40 )
      v106 = 64;
    if ( v106 >= (unsigned int)v107 )
    {
      while ( v105 != v108 )
      {
        if ( *v105 != -8 )
        {
          if ( *v105 != -16 )
          {
            v109 = v105[3];
            if ( v109 != v105[2] )
              _libc_free(v109);
          }
          *v105 = -8;
        }
        v105 += 8;
      }
    }
    else
    {
      do
      {
        if ( *v105 != -16 && *v105 != -8 )
        {
          v167 = v105[3];
          if ( v167 != v105[2] )
            _libc_free(v167);
        }
        v105 += 8;
      }
      while ( v105 != v108 );
      v187 = *(_DWORD *)(a1 + 1952);
      if ( v104 )
      {
        v188 = 64;
        if ( v104 != 1 )
        {
          _BitScanReverse(&v189, v104 - 1);
          v188 = 1 << (33 - (v189 ^ 0x1F));
          if ( v188 < 64 )
            v188 = 64;
        }
        if ( v187 == v188 )
          goto LABEL_339;
        j___libc_free_0(*(_QWORD *)(a1 + 1936));
        v190 = sub_19E3C70(v188);
        *(_DWORD *)(a1 + 1952) = v190;
        if ( v190 )
        {
          *(_QWORD *)(a1 + 1936) = sub_22077B0((unsigned __int64)v190 << 6);
LABEL_339:
          sub_19E8EB0(a1 + 1928);
          goto LABEL_143;
        }
      }
      else
      {
        if ( !v187 )
          goto LABEL_339;
        j___libc_free_0(*(_QWORD *)(a1 + 1936));
        *(_DWORD *)(a1 + 1952) = 0;
      }
      *(_QWORD *)(a1 + 1936) = 0;
    }
    *(_QWORD *)(a1 + 1944) = 0;
  }
LABEL_143:
  v110 = *(_DWORD *)(a1 + 1880);
  ++*(_QWORD *)(a1 + 1864);
  if ( !v110 && !*(_DWORD *)(a1 + 1884) )
    goto LABEL_161;
  v111 = *(_QWORD ***)(a1 + 1872);
  v112 = *(unsigned int *)(a1 + 1888);
  v113 = &v111[5 * v112];
  v114 = 4 * v110;
  if ( (unsigned int)(4 * v110) < 0x40 )
    v114 = 64;
  if ( (unsigned int)v112 <= v114 )
  {
    v115 = v111 + 2;
    if ( v111 != v113 )
    {
      while ( 1 )
      {
        v116 = (__int64)*(v115 - 2);
        if ( v116 != -8 )
        {
          if ( v116 != -16 )
          {
            v117 = *v115;
            while ( v115 != v117 )
            {
              v118 = v117;
              v117 = (_QWORD *)*v117;
              j_j___libc_free_0(v118, 40);
            }
          }
          *(v115 - 2) = (_QWORD *)-8LL;
        }
        if ( v113 == v115 + 3 )
          break;
        v115 += 5;
      }
    }
    goto LABEL_160;
  }
  for ( i2 = v111 + 2; ; i2 += 5 )
  {
    v192 = (__int64)*(i2 - 2);
    if ( v192 != -16 && v192 != -8 )
    {
      for ( i3 = *i2; i2 != i3; v110 = v198 )
      {
        v194 = i3;
        i3 = (_QWORD *)*i3;
        v198 = v110;
        j_j___libc_free_0(v194, 40);
      }
    }
    if ( v113 == i2 + 3 )
      break;
  }
  v119 = *(_DWORD *)(a1 + 1888);
  if ( v110 )
  {
    v195 = 64;
    if ( v110 != 1 )
    {
      _BitScanReverse(&v196, v110 - 1);
      v195 = 1 << (33 - (v196 ^ 0x1F));
      if ( v195 < 64 )
        v195 = 64;
    }
    if ( v119 != v195 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 1872));
      v197 = sub_19E3C70(v195);
      *(_DWORD *)(a1 + 1888) = v197;
      if ( !v197 )
        goto LABEL_159;
      *(_QWORD *)(a1 + 1872) = sub_22077B0(40LL * v197);
    }
  }
  else if ( v119 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 1872));
    *(_DWORD *)(a1 + 1888) = 0;
LABEL_159:
    *(_QWORD *)(a1 + 1872) = 0;
LABEL_160:
    *(_QWORD *)(a1 + 1880) = 0;
    goto LABEL_161;
  }
  sub_19E8E30(a1 + 1864);
LABEL_161:
  if ( v199 != v201 )
    _libc_free((unsigned __int64)v199);
}
