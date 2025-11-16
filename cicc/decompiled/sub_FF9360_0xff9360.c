// Function: sub_FF9360
// Address: 0xff9360
//
__int64 __fastcall sub_FF9360(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // eax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r9
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  unsigned int v46; // r14d
  __int64 v47; // rsi
  __int64 v48; // rdx
  _BYTE *v49; // rcx
  _BYTE *v50; // rax
  _QWORD *v51; // r15
  unsigned __int64 v52; // rdi
  int v53; // eax
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rdi
  unsigned int v60; // eax
  unsigned int v61; // eax
  unsigned int v62; // ecx
  __int64 v63; // rdx
  _QWORD *v64; // rax
  __int64 v65; // rdx
  _QWORD *i; // rdx
  unsigned int v67; // eax
  unsigned int v68; // eax
  unsigned int v69; // ecx
  __int64 v70; // rdx
  _QWORD *v71; // rax
  __int64 v72; // rdx
  _QWORD *m; // rdx
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // r14
  __int64 v77; // rsi
  __int64 v78; // rdi
  size_t v79; // r14
  const void *v80; // r13
  const char *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rbx
  __int64 v84; // r12
  __int64 v85; // r13
  __int64 v86; // rdi
  __int64 result; // rax
  __int64 v88; // rbx
  __int64 v89; // r12
  __int64 v90; // r13
  __int64 v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rdi
  int v94; // eax
  __int64 v95; // rax
  __int64 v96; // rdi
  int v97; // eax
  unsigned int v98; // eax
  unsigned int v99; // ebx
  char v100; // al
  __int64 v101; // rax
  bool v102; // zf
  _QWORD *v103; // rax
  __int64 v104; // rdx
  _QWORD *k; // rdx
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 j; // rdx
  unsigned int v109; // eax
  unsigned int v110; // ebx
  char v111; // al
  __int64 v112; // rax
  _QWORD *v113; // rax
  __int64 v114; // rdx
  _QWORD *v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rdx
  unsigned __int64 v121; // [rsp+48h] [rbp-A68h]
  __int64 v122; // [rsp+50h] [rbp-A60h]
  __int64 v124; // [rsp+58h] [rbp-A58h]
  _QWORD v125[54]; // [rsp+60h] [rbp-A50h] BYREF
  __int64 v126; // [rsp+210h] [rbp-8A0h] BYREF
  _QWORD *v127; // [rsp+218h] [rbp-898h]
  int v128; // [rsp+220h] [rbp-890h]
  int v129; // [rsp+224h] [rbp-88Ch]
  int v130; // [rsp+228h] [rbp-888h]
  char v131; // [rsp+22Ch] [rbp-884h]
  _QWORD v132[8]; // [rsp+230h] [rbp-880h] BYREF
  unsigned __int64 *v133; // [rsp+270h] [rbp-840h] BYREF
  __int64 v134; // [rsp+278h] [rbp-838h]
  unsigned __int64 v135; // [rsp+280h] [rbp-830h] BYREF
  int v136; // [rsp+288h] [rbp-828h]
  unsigned __int64 v137; // [rsp+290h] [rbp-820h]
  int v138; // [rsp+298h] [rbp-818h]
  __int64 v139; // [rsp+2A0h] [rbp-810h]
  char v140[8]; // [rsp+3C0h] [rbp-6F0h] BYREF
  __int64 v141; // [rsp+3C8h] [rbp-6E8h]
  char v142; // [rsp+3DCh] [rbp-6D4h]
  _BYTE v143[64]; // [rsp+3E0h] [rbp-6D0h] BYREF
  _BYTE *v144; // [rsp+420h] [rbp-690h] BYREF
  __int64 v145; // [rsp+428h] [rbp-688h]
  _BYTE v146[320]; // [rsp+430h] [rbp-680h] BYREF
  char v147[8]; // [rsp+570h] [rbp-540h] BYREF
  __int64 v148; // [rsp+578h] [rbp-538h]
  char v149; // [rsp+58Ch] [rbp-524h]
  _BYTE v150[64]; // [rsp+590h] [rbp-520h] BYREF
  _BYTE *v151; // [rsp+5D0h] [rbp-4E0h] BYREF
  __int64 v152; // [rsp+5D8h] [rbp-4D8h]
  _BYTE v153[320]; // [rsp+5E0h] [rbp-4D0h] BYREF
  char v154[8]; // [rsp+720h] [rbp-390h] BYREF
  __int64 v155; // [rsp+728h] [rbp-388h]
  char v156; // [rsp+73Ch] [rbp-374h]
  char v157[64]; // [rsp+740h] [rbp-370h] BYREF
  _BYTE *v158; // [rsp+780h] [rbp-330h] BYREF
  __int64 v159; // [rsp+788h] [rbp-328h]
  _BYTE v160[320]; // [rsp+790h] [rbp-320h] BYREF
  char v161[8]; // [rsp+8D0h] [rbp-1E0h] BYREF
  __int64 v162; // [rsp+8D8h] [rbp-1D8h]
  char v163; // [rsp+8ECh] [rbp-1C4h]
  _BYTE v164[64]; // [rsp+8F0h] [rbp-1C0h] BYREF
  _BYTE *v165; // [rsp+930h] [rbp-180h] BYREF
  __int64 v166; // [rsp+938h] [rbp-178h]
  _BYTE v167[368]; // [rsp+940h] [rbp-170h] BYREF

  v8 = (__int64)a1;
  a1[8] = a2;
  a1[9] = a3;
  v9 = sub_22077B0(56);
  v10 = v9;
  if ( v9 )
    sub_FF8F20(v9, a2);
  v11 = a1[10];
  a1[10] = v10;
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 32);
    if ( *(_QWORD *)(v11 + 40) != v12 )
    {
      v13 = *(_QWORD *)(v11 + 32);
      v14 = *(_QWORD *)(v11 + 40);
      do
      {
        v15 = *(unsigned int *)(v13 + 24);
        v16 = *(_QWORD *)(v13 + 8);
        v13 += 32;
        sub_C7D6A0(v16, 16 * v15, 8);
      }
      while ( v14 != v13 );
      v8 = (__int64)a1;
      v12 = *(_QWORD *)(v11 + 32);
    }
    if ( v12 )
      j_j___libc_free_0(v12, *(_QWORD *)(v11 + 48) - v12);
    sub_C7D6A0(*(_QWORD *)(v11 + 8), 16LL * *(unsigned int *)(v11 + 24), 8);
    j_j___libc_free_0(v11, 56);
  }
  v122 = 0;
  if ( a5 )
  {
    v124 = 0;
    if ( a6 )
      goto LABEL_13;
    goto LABEL_153;
  }
  v92 = sub_22077B0(128);
  a5 = v92;
  if ( v92 )
  {
    *(_BYTE *)(v92 + 112) = 0;
    v93 = v92;
    *(_QWORD *)v92 = v92 + 16;
    *(_QWORD *)(v92 + 8) = 0x100000000LL;
    *(_QWORD *)(v92 + 24) = v92 + 40;
    *(_QWORD *)(v92 + 32) = 0x600000000LL;
    *(_QWORD *)(v92 + 96) = 0;
    *(_QWORD *)(v92 + 104) = a2;
    v94 = *(_DWORD *)(a2 + 92);
    *(_DWORD *)(a5 + 116) = 0;
    *(_DWORD *)(a5 + 120) = v94;
    sub_B1F440(v93);
  }
  v122 = a5;
  v124 = 0;
  if ( !a6 )
  {
LABEL_153:
    v95 = sub_22077B0(152);
    a6 = v95;
    if ( v95 )
    {
      v96 = v95;
      *(_QWORD *)(v95 + 120) = 0;
      *(_QWORD *)v95 = v95 + 16;
      *(_QWORD *)(v95 + 8) = 0x400000000LL;
      *(_QWORD *)(v95 + 48) = v95 + 64;
      *(_QWORD *)(v95 + 56) = 0x600000000LL;
      *(_BYTE *)(v95 + 136) = 0;
      *(_QWORD *)(v95 + 128) = a2;
      v97 = *(_DWORD *)(a2 + 92);
      *(_DWORD *)(a6 + 140) = 0;
      *(_DWORD *)(a6 + 144) = v97;
      sub_B29120(v96);
    }
    v124 = a6;
  }
LABEL_13:
  sub_FF3DD0(v8, a2, a5, a6);
  v17 = *(_QWORD *)(a2 + 80);
  v127 = v132;
  v128 = 8;
  if ( v17 )
    v17 -= 24;
  v130 = 0;
  memset(v125, 0, sizeof(v125));
  HIDWORD(v125[13]) = 8;
  v125[12] = &v125[14];
  v133 = &v135;
  v134 = 0x800000000LL;
  v18 = *(_QWORD *)(v17 + 48);
  v125[1] = &v125[4];
  v19 = v18 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v125[3]) = 1;
  LODWORD(v125[2]) = 8;
  v131 = 1;
  v129 = 1;
  v132[0] = v17;
  v126 = 1;
  if ( v19 == v17 + 48 )
    goto LABEL_149;
  if ( !v19 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
  {
LABEL_149:
    v20 = 0;
    v22 = 0;
    v21 = 0;
  }
  else
  {
    v121 = v19 - 24;
    v20 = sub_B46E30(v19 - 24);
    v21 = v121;
    v22 = v121;
  }
  v136 = v20;
  v135 = v22;
  v137 = v21;
  v139 = v17;
  v138 = 0;
  LODWORD(v134) = 1;
  sub_FDEBC0((__int64)&v126);
  sub_C8CF70((__int64)v147, v150, 8, (__int64)&v125[4], (__int64)v125);
  v151 = v153;
  v152 = 0x800000000LL;
  if ( LODWORD(v125[13]) )
    sub_FF1510((__int64)&v151, (__int64)&v125[12], v23, v24, v25, v26);
  sub_C8CF70((__int64)v140, v143, 8, (__int64)v132, (__int64)&v126);
  v144 = v146;
  v145 = 0x800000000LL;
  if ( (_DWORD)v134 )
    sub_FF1510((__int64)&v144, (__int64)&v133, v27, v28, v29, (unsigned int)v134);
  sub_C8CF70((__int64)v154, v157, 8, (__int64)v143, (__int64)v140);
  v158 = v160;
  v159 = 0x800000000LL;
  if ( (_DWORD)v145 )
    sub_FF1510((__int64)&v158, (__int64)&v144, v30, v31, (unsigned int)v145, v32);
  v33 = v164;
  sub_C8CF70((__int64)v161, v164, 8, (__int64)v150, (__int64)v147);
  v165 = v167;
  v166 = 0x800000000LL;
  if ( (_DWORD)v152 )
  {
    v33 = &v151;
    sub_FF1510((__int64)&v165, (__int64)&v151, v34, v35, v36, v37);
  }
  if ( v144 != v146 )
    _libc_free(v144, v33);
  if ( !v142 )
    _libc_free(v141, v33);
  if ( v151 != v153 )
    _libc_free(v151, v33);
  if ( !v149 )
    _libc_free(v148, v33);
  if ( v133 != &v135 )
    _libc_free(v133, v33);
  if ( !v131 )
    _libc_free(v127, v33);
  if ( (_QWORD *)v125[12] != &v125[14] )
    _libc_free(v125[12], v33);
  if ( !BYTE4(v125[3]) )
    _libc_free(v125[1], v33);
  sub_C8CD80((__int64)v140, (__int64)v143, (__int64)v154, v35, v36, v37);
  v144 = v146;
  v145 = 0x800000000LL;
  if ( (_DWORD)v159 )
    sub_FF16F0((__int64)&v144, (__int64 *)&v158, v38, v39, v40, v41);
  sub_C8CD80((__int64)v147, (__int64)v150, (__int64)v161, v39, v40, v41);
  v45 = (unsigned int)v166;
  v151 = v153;
  v152 = 0x800000000LL;
  if ( (_DWORD)v166 )
  {
    sub_FF16F0((__int64)&v151, (__int64 *)&v165, v42, (unsigned int)v166, v43, v44);
    v45 = (unsigned int)v152;
  }
LABEL_47:
  v46 = v145;
  while ( 1 )
  {
    v47 = (__int64)v144;
    v48 = 40LL * v46;
    if ( v46 != v45 )
      goto LABEL_52;
    if ( v144 == &v144[v48] )
      break;
    v49 = v151;
    v50 = v144;
    while ( *((_QWORD *)v50 + 4) == *((_QWORD *)v49 + 4)
         && *((_DWORD *)v50 + 6) == *((_DWORD *)v49 + 6)
         && *((_DWORD *)v50 + 2) == *((_DWORD *)v49 + 2) )
    {
      v50 += 40;
      v49 += 40;
      if ( &v144[v48] == v50 )
        goto LABEL_69;
    }
LABEL_52:
    v51 = *(_QWORD **)&v144[v48 - 8];
    v52 = v51[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v52 == v51 + 6 )
    {
      v54 = 0;
    }
    else
    {
      if ( !v52 )
        BUG();
      v53 = *(unsigned __int8 *)(v52 - 24);
      v54 = v52 - 24;
      if ( (unsigned int)(v53 - 30) >= 0xB )
        v54 = 0;
    }
    if ( (unsigned int)sub_B46E30(v54) > 1 )
    {
      if ( !(unsigned __int8)sub_FF66A0(v8, (__int64)v51)
        && !(unsigned __int8)sub_FF6E20(v8, (unsigned __int64)v51)
        && !(unsigned __int8)sub_FF6D40(v8, (__int64)v51)
        && !(unsigned __int8)sub_FF7D20(v8, (__int64)v51, a4) )
      {
        sub_FF80D0(v8, v51, v55, v56, v57, v58);
      }
      v46 = v145;
    }
    LODWORD(v145) = --v46;
    if ( v46 )
    {
      sub_FDEBC0((__int64)v140);
      v45 = (unsigned int)v152;
      goto LABEL_47;
    }
    v45 = (unsigned int)v152;
  }
LABEL_69:
  if ( v151 != v153 )
    _libc_free(v151, v144);
  if ( !v149 )
    _libc_free(v148, v47);
  if ( v144 != v146 )
    _libc_free(v144, v47);
  if ( !v142 )
    _libc_free(v141, v47);
  if ( v165 != v167 )
    _libc_free(v165, v47);
  if ( !v163 )
    _libc_free(v162, v47);
  v59 = (__int64)v158;
  if ( v158 != v160 )
    _libc_free(v158, v47);
  if ( !v156 )
  {
    v59 = v155;
    _libc_free(v155, v47);
  }
  v60 = *(_DWORD *)(v8 + 176);
  ++*(_QWORD *)(v8 + 168);
  v61 = v60 >> 1;
  if ( v61 )
  {
    if ( (*(_BYTE *)(v8 + 176) & 1) == 0 )
    {
      v62 = 4 * v61;
      goto LABEL_88;
    }
LABEL_146:
    v64 = (_QWORD *)(v8 + 184);
    v65 = 12;
LABEL_91:
    for ( i = &v64[v65]; i != v64; *((_DWORD *)v64 - 4) = 0x7FFFFFFF )
    {
      *v64 = -4096;
      v64 += 3;
    }
    *(_QWORD *)(v8 + 176) &= 1uLL;
    goto LABEL_94;
  }
  if ( !*(_DWORD *)(v8 + 180) )
    goto LABEL_94;
  v62 = 0;
  if ( (*(_BYTE *)(v8 + 176) & 1) != 0 )
    goto LABEL_146;
LABEL_88:
  v63 = *(unsigned int *)(v8 + 192);
  if ( v62 >= (unsigned int)v63 || (unsigned int)v63 <= 0x40 )
  {
    v64 = *(_QWORD **)(v8 + 184);
    v65 = 3 * v63;
    goto LABEL_91;
  }
  if ( v61 )
  {
    v109 = v61 - 1;
    if ( !v109 )
    {
      v47 = 24 * v63;
      v59 = *(_QWORD *)(v8 + 184);
      sub_C7D6A0(v59, 24 * v63, 8);
      *(_BYTE *)(v8 + 176) |= 1u;
      goto LABEL_171;
    }
    _BitScanReverse(&v109, v109);
    v110 = 1 << (33 - (v109 ^ 0x1F));
    if ( v110 - 5 > 0x3A )
    {
      if ( (_DWORD)v63 == v110 )
      {
        v102 = (*(_QWORD *)(v8 + 176) & 1LL) == 0;
        *(_QWORD *)(v8 + 176) &= 1uLL;
        if ( v102 )
        {
          v116 = *(_QWORD *)(v8 + 184);
          v117 = 24 * v63;
        }
        else
        {
          v116 = v8 + 184;
          v117 = 96;
        }
        v118 = v116 + v117;
        do
        {
          if ( v116 )
          {
            *(_QWORD *)v116 = -4096;
            *(_DWORD *)(v116 + 8) = 0x7FFFFFFF;
          }
          v116 += 24;
        }
        while ( v118 != v116 );
        goto LABEL_94;
      }
      v47 = 24 * v63;
      v59 = *(_QWORD *)(v8 + 184);
      sub_C7D6A0(v59, 24 * v63, 8);
      v111 = *(_BYTE *)(v8 + 176) | 1;
      *(_BYTE *)(v8 + 176) = v111;
      if ( v110 <= 4 )
        goto LABEL_171;
      v59 = 24LL * v110;
    }
    else
    {
      v110 = 64;
      sub_C7D6A0(*(_QWORD *)(v8 + 184), 24 * v63, 8);
      v59 = 1536;
      v111 = *(_BYTE *)(v8 + 176);
    }
    v47 = 8;
    *(_BYTE *)(v8 + 176) = v111 & 0xFE;
    v112 = sub_C7D670(v59, 8);
    *(_DWORD *)(v8 + 192) = v110;
    *(_QWORD *)(v8 + 184) = v112;
    goto LABEL_171;
  }
  v59 = *(_QWORD *)(v8 + 184);
  v47 = 24 * v63;
  sub_C7D6A0(v59, 24 * v63, 8);
  *(_BYTE *)(v8 + 176) |= 1u;
LABEL_171:
  v102 = (*(_QWORD *)(v8 + 176) & 1LL) == 0;
  *(_QWORD *)(v8 + 176) &= 1uLL;
  if ( v102 )
  {
    v106 = *(_QWORD *)(v8 + 184);
    v107 = 24LL * *(unsigned int *)(v8 + 192);
  }
  else
  {
    v106 = v8 + 184;
    v107 = 96;
  }
  for ( j = v106 + v107; j != v106; v106 += 24 )
  {
    if ( v106 )
    {
      *(_QWORD *)v106 = -4096;
      *(_DWORD *)(v106 + 8) = 0x7FFFFFFF;
    }
  }
LABEL_94:
  v67 = *(_DWORD *)(v8 + 96);
  ++*(_QWORD *)(v8 + 88);
  v68 = v67 >> 1;
  if ( v68 )
  {
    if ( (*(_BYTE *)(v8 + 96) & 1) == 0 )
    {
      v69 = 4 * v68;
      goto LABEL_97;
    }
  }
  else
  {
    if ( !*(_DWORD *)(v8 + 100) )
      goto LABEL_103;
    v69 = 0;
    if ( (*(_BYTE *)(v8 + 96) & 1) == 0 )
    {
LABEL_97:
      v70 = *(unsigned int *)(v8 + 112);
      if ( v69 >= (unsigned int)v70 || (unsigned int)v70 <= 0x40 )
      {
        v71 = *(_QWORD **)(v8 + 104);
        v72 = 2 * v70;
        goto LABEL_100;
      }
      if ( v68 )
      {
        v98 = v68 - 1;
        if ( v98 )
        {
          _BitScanReverse(&v98, v98);
          v99 = 1 << (33 - (v98 ^ 0x1F));
          if ( v99 - 5 <= 0x3A )
          {
            v99 = 64;
            sub_C7D6A0(*(_QWORD *)(v8 + 104), 16 * v70, 8);
            v100 = *(_BYTE *)(v8 + 96);
            v59 = 1024;
            goto LABEL_161;
          }
          if ( (_DWORD)v70 == v99 )
          {
            v102 = (*(_QWORD *)(v8 + 96) & 1LL) == 0;
            *(_QWORD *)(v8 + 96) &= 1uLL;
            if ( v102 )
            {
              v113 = *(_QWORD **)(v8 + 104);
              v114 = 2 * v70;
            }
            else
            {
              v113 = (_QWORD *)(v8 + 104);
              v114 = 8;
            }
            v115 = &v113[v114];
            do
            {
              if ( v113 )
                *v113 = -4096;
              v113 += 2;
            }
            while ( v115 != v113 );
            goto LABEL_103;
          }
          v59 = *(_QWORD *)(v8 + 104);
          v47 = 16 * v70;
          sub_C7D6A0(v59, 16 * v70, 8);
          v100 = *(_BYTE *)(v8 + 96) | 1;
          *(_BYTE *)(v8 + 96) = v100;
          if ( v99 > 4 )
          {
            v59 = 16LL * v99;
LABEL_161:
            v47 = 8;
            *(_BYTE *)(v8 + 96) = v100 & 0xFE;
            v101 = sub_C7D670(v59, 8);
            *(_DWORD *)(v8 + 112) = v99;
            *(_QWORD *)(v8 + 104) = v101;
          }
        }
        else
        {
          v59 = *(_QWORD *)(v8 + 104);
          v47 = 16 * v70;
          sub_C7D6A0(v59, 16 * v70, 8);
          *(_BYTE *)(v8 + 96) |= 1u;
        }
      }
      else
      {
        v59 = *(_QWORD *)(v8 + 104);
        v47 = 16LL * (unsigned int)v70;
        sub_C7D6A0(v59, v47, 8);
        *(_BYTE *)(v8 + 96) |= 1u;
      }
      v102 = (*(_QWORD *)(v8 + 96) & 1LL) == 0;
      *(_QWORD *)(v8 + 96) &= 1uLL;
      if ( v102 )
      {
        v103 = *(_QWORD **)(v8 + 104);
        v104 = 2LL * *(unsigned int *)(v8 + 112);
      }
      else
      {
        v103 = (_QWORD *)(v8 + 104);
        v104 = 8;
      }
      for ( k = &v103[v104]; k != v103; v103 += 2 )
      {
        if ( v103 )
          *v103 = -4096;
      }
      goto LABEL_103;
    }
  }
  v71 = (_QWORD *)(v8 + 104);
  v72 = 8;
LABEL_100:
  for ( m = &v71[v72]; m != v71; v71 += 2 )
    *v71 = -4096;
  *(_QWORD *)(v8 + 96) &= 1uLL;
LABEL_103:
  v74 = *(_QWORD *)(v8 + 80);
  *(_QWORD *)(v8 + 80) = 0;
  if ( v74 )
  {
    v75 = *(_QWORD *)(v74 + 40);
    v76 = *(_QWORD *)(v74 + 32);
    if ( v75 != v76 )
    {
      do
      {
        v77 = *(unsigned int *)(v76 + 24);
        v78 = *(_QWORD *)(v76 + 8);
        v76 += 32;
        sub_C7D6A0(v78, 16 * v77, 8);
      }
      while ( v75 != v76 );
      v76 = *(_QWORD *)(v74 + 32);
    }
    if ( v76 )
      j_j___libc_free_0(v76, *(_QWORD *)(v74 + 48) - v76);
    sub_C7D6A0(*(_QWORD *)(v74 + 8), 16LL * *(unsigned int *)(v74 + 24), 8);
    v47 = 56;
    v59 = v74;
    j_j___libc_free_0(v74, 56);
  }
  if ( (_BYTE)qword_4F8E9A8 )
  {
    v79 = qword_4F8E8B0;
    if ( !qword_4F8E8B0
      || (v80 = qword_4F8E8A8, v81 = sub_BD5D20(a2), v79 == v82)
      && (v47 = (__int64)v80, v59 = (__int64)v81, !memcmp(v81, v80, v79)) )
    {
      v47 = sub_C5F790(v59, v47);
      sub_FF0A60(v8, v47);
    }
  }
  if ( v124 )
  {
    v83 = *(_QWORD *)(v124 + 48);
    v84 = v83 + 8LL * *(unsigned int *)(v124 + 56);
    if ( v83 != v84 )
    {
      do
      {
        v85 = *(_QWORD *)(v84 - 8);
        v84 -= 8;
        if ( v85 )
        {
          v86 = *(_QWORD *)(v85 + 24);
          if ( v86 != v85 + 40 )
            _libc_free(v86, v47);
          v47 = 80;
          j_j___libc_free_0(v85, 80);
        }
      }
      while ( v83 != v84 );
      v84 = *(_QWORD *)(v124 + 48);
    }
    if ( v84 != v124 + 64 )
      _libc_free(v84, v47);
    if ( *(_QWORD *)v124 != v124 + 16 )
      _libc_free(*(_QWORD *)v124, v47);
    v47 = 152;
    j_j___libc_free_0(v124, 152);
  }
  result = v122;
  if ( v122 )
  {
    v88 = *(_QWORD *)(v122 + 24);
    v89 = v88 + 8LL * *(unsigned int *)(v122 + 32);
    if ( v88 != v89 )
    {
      do
      {
        v90 = *(_QWORD *)(v89 - 8);
        v89 -= 8;
        if ( v90 )
        {
          v91 = *(_QWORD *)(v90 + 24);
          if ( v91 != v90 + 40 )
            _libc_free(v91, v47);
          v47 = 80;
          j_j___libc_free_0(v90, 80);
        }
      }
      while ( v88 != v89 );
      v89 = *(_QWORD *)(v122 + 24);
    }
    if ( v89 != v122 + 40 )
      _libc_free(v89, v47);
    if ( *(_QWORD *)v122 != v122 + 16 )
      _libc_free(*(_QWORD *)v122, v47);
    return j_j___libc_free_0(v122, 128);
  }
  return result;
}
