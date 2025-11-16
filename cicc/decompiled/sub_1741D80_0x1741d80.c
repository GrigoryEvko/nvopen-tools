// Function: sub_1741D80
// Address: 0x1741d80
//
unsigned __int64 __fastcall sub_1741D80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 ***v6; // r15
  __int64 ***v7; // r12
  int v8; // r14d
  __int64 **v9; // rax
  unsigned __int64 v10; // rax
  void *v11; // r10
  signed __int64 v12; // r8
  unsigned __int64 v13; // rax
  __int64 v14; // r12
  signed __int64 v15; // r15
  __int64 v16; // rax
  char *v17; // rcx
  signed __int64 v18; // rdx
  unsigned __int64 v19; // rax
  char *v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // r12
  char *v23; // rax
  unsigned __int64 v24; // rdx
  int v25; // r13d
  __int64 *v26; // r12
  unsigned __int64 v27; // rbx
  char *v28; // rdx
  char *v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rax
  char *v32; // rsi
  __int64 ***v33; // rdx
  __int64 v34; // rdi
  char *v35; // rsi
  char *v36; // rsi
  char *v37; // r13
  __int64 v38; // r12
  char *v39; // rsi
  __int64 v40; // rax
  _QWORD *v41; // rcx
  _QWORD *v42; // r15
  int v43; // r12d
  _QWORD *v44; // r13
  __int64 *v45; // rax
  __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  _QWORD *v49; // r12
  void *v51; // r10
  signed __int64 v52; // rdx
  signed __int64 v53; // rax
  __int64 v54; // rdi
  bool v55; // cf
  unsigned __int64 v56; // rax
  _BYTE *v57; // r8
  char *v58; // r9
  char *v59; // r11
  __int64 *v60; // r13
  __int64 **v61; // r12
  __int64 **v62; // rax
  char *v63; // r15
  char *v64; // r12
  __int64 v65; // r13
  __int64 v66; // rax
  unsigned __int64 v67; // r15
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r12
  __int64 v71; // r12
  __int64 v72; // r14
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 *v80; // rbx
  __int64 *v81; // r15
  char *v82; // r8
  signed __int64 v83; // r14
  __int64 v84; // rsi
  int v85; // r9d
  __int64 v86; // rax
  _QWORD *v87; // rax
  __int16 v88; // dx
  unsigned int v89; // eax
  __int64 *v90; // r14
  unsigned __int8 *v91; // rsi
  __int64 v92; // rbx
  char *v93; // r13
  __int64 v94; // rdi
  char *v95; // rax
  char *v96; // rsi
  __int64 v97; // r15
  char *v98; // rcx
  __int64 v99; // r12
  __int64 v100; // rax
  char *v101; // rdx
  __int64 v102; // r12
  char *v103; // rax
  _BYTE *v104; // r15
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // r12
  __int64 v108; // rax
  __int64 v109; // r14
  __int64 v110; // r12
  __int64 v111; // rax
  __int64 *v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rsi
  __int64 v115; // rax
  __int64 v116; // rcx
  __int64 v117; // rsi
  unsigned __int8 *v118; // rsi
  __int64 v119; // r9
  int v120; // esi
  __int64 v121; // rax
  __int64 *v122; // r10
  __int64 v123; // r11
  __int64 v124; // rdx
  __int64 v125; // rax
  int v126; // r8d
  signed __int64 v127; // r9
  char *v128; // rax
  signed __int64 v129; // rsi
  _QWORD *v130; // rax
  __int64 v131; // r8
  __int64 v132; // rax
  __int64 v133; // [rsp+0h] [rbp-190h]
  signed __int64 v134; // [rsp+0h] [rbp-190h]
  __int64 v135; // [rsp+8h] [rbp-188h]
  _BYTE *v136; // [rsp+8h] [rbp-188h]
  char *v137; // [rsp+8h] [rbp-188h]
  void *v138; // [rsp+8h] [rbp-188h]
  void *v139; // [rsp+10h] [rbp-180h]
  char *v140; // [rsp+10h] [rbp-180h]
  __int64 *v141; // [rsp+10h] [rbp-180h]
  __int64 v142; // [rsp+10h] [rbp-180h]
  __int64 v143; // [rsp+10h] [rbp-180h]
  __int64 v144; // [rsp+10h] [rbp-180h]
  unsigned int v145; // [rsp+20h] [rbp-170h]
  char *v146; // [rsp+20h] [rbp-170h]
  _BYTE *v147; // [rsp+20h] [rbp-170h]
  __int64 v148; // [rsp+28h] [rbp-168h]
  char *v149; // [rsp+28h] [rbp-168h]
  __int64 v150; // [rsp+28h] [rbp-168h]
  void *v151; // [rsp+28h] [rbp-168h]
  char *v152; // [rsp+28h] [rbp-168h]
  signed __int64 v153; // [rsp+28h] [rbp-168h]
  __int64 v155; // [rsp+30h] [rbp-160h]
  __int64 *v156; // [rsp+30h] [rbp-160h]
  __int64 v157; // [rsp+38h] [rbp-158h]
  __int64 v158; // [rsp+40h] [rbp-150h]
  __int64 v159; // [rsp+40h] [rbp-150h]
  __int64 v160; // [rsp+48h] [rbp-148h]
  unsigned __int64 v161; // [rsp+48h] [rbp-148h]
  __int64 v162; // [rsp+50h] [rbp-140h]
  __int64 v164; // [rsp+58h] [rbp-138h]
  __int64 v165; // [rsp+60h] [rbp-130h]
  __int64 v166; // [rsp+68h] [rbp-128h] BYREF
  __int64 v167; // [rsp+70h] [rbp-120h] BYREF
  __int64 v168; // [rsp+78h] [rbp-118h] BYREF
  __int64 **v169; // [rsp+80h] [rbp-110h] BYREF
  __int64 v170; // [rsp+88h] [rbp-108h] BYREF
  void *src; // [rsp+90h] [rbp-100h] BYREF
  char *v172; // [rsp+98h] [rbp-F8h]
  char *v173; // [rsp+A0h] [rbp-F0h]
  char *v174; // [rsp+B0h] [rbp-E0h] BYREF
  char *v175; // [rsp+B8h] [rbp-D8h]
  char *v176; // [rsp+C0h] [rbp-D0h]
  void *v177; // [rsp+D0h] [rbp-C0h] BYREF
  char *v178; // [rsp+D8h] [rbp-B8h]
  char *v179; // [rsp+E0h] [rbp-B0h]
  unsigned __int8 *v180; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned __int64 v181; // [rsp+F8h] [rbp-98h]
  __int64 v182; // [rsp+100h] [rbp-90h]
  char *v183; // [rsp+110h] [rbp-80h] BYREF
  __int64 v184; // [rsp+118h] [rbp-78h]
  _BYTE v185[112]; // [rsp+120h] [rbp-70h] BYREF

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v166 = a2;
  if ( (a2 & 4) != 0 )
    v160 = **(_QWORD **)(v3 - 24);
  else
    v160 = **(_QWORD **)(v3 - 72);
  v162 = *(_QWORD *)(v160 + 24);
  v167 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (unsigned __int8)sub_1560490(&v167, 19, 0) )
    return 0;
  v165 = sub_1649C60(*(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))));
  v4 = *(_QWORD *)(v165 + 24);
  v168 = *(_QWORD *)(v165 + 112);
  if ( !v168 )
    goto LABEL_50;
  v169 = 0;
  v5 = *(_QWORD *)(v4 + 16);
  v170 = 0;
  v6 = (__int64 ***)(v5 + 8);
  v7 = (__int64 ***)(v5 + 8LL * *(unsigned int *)(v4 + 12));
  if ( v7 == (__int64 ***)(v5 + 8) )
    goto LABEL_50;
  v8 = 0;
  do
  {
    v183 = (char *)sub_1560230(&v168, v8);
    if ( (unsigned __int8)sub_155EE10((__int64)&v183, 19) )
    {
      v9 = *v6;
      v169 = *v6;
      v170 = (__int64)v183;
      goto LABEL_10;
    }
    ++v6;
    ++v8;
  }
  while ( v7 != v6 );
  v9 = v169;
LABEL_10:
  if ( !v9 )
  {
LABEL_50:
    if ( *(_QWORD *)v165 != v160 )
      v165 = sub_15A4510((__int64 ***)v165, (__int64 **)v160, 0);
    v45 = (__int64 *)((v166 & 0xFFFFFFFFFFFFFFF8LL) - 72);
    if ( (v166 & 4) != 0 )
      v45 = (__int64 *)((v166 & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( *v45 )
    {
      v46 = v45[1];
      v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v47 = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
    }
    *v45 = v165;
    if ( v165 )
    {
      v48 = *(_QWORD *)(v165 + 8);
      v45[1] = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
      v45[2] = (v165 + 8) | v45[2] & 3;
      *(_QWORD *)(v165 + 8) = v45;
    }
    return v166 & 0xFFFFFFFFFFFFFFF8LL;
  }
  src = 0;
  v172 = 0;
  v173 = 0;
  v161 = v166 & 0xFFFFFFFFFFFFFFF8LL;
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v10 = sub_1389B50(&v166);
  v11 = src;
  v12 = v173 - (_BYTE *)src;
  v13 = -1431655765
      * (unsigned int)((__int64)(v10
                               - ((v166 & 0xFFFFFFFFFFFFFFF8LL)
                                - 24LL * (*(_DWORD *)((v166 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3)
      + 1;
  if ( v13 <= (v173 - (_BYTE *)src) >> 3 )
    goto LABEL_17;
  v14 = 8 * v13;
  v15 = v172 - (_BYTE *)src;
  if ( v13 )
  {
    v16 = sub_22077B0(8 * v13);
    v11 = src;
    v17 = (char *)v16;
    v18 = v172 - (_BYTE *)src;
    v12 = v173 - (_BYTE *)src;
  }
  else
  {
    v18 = v172 - (_BYTE *)src;
    v17 = 0;
  }
  if ( v18 > 0 )
  {
    v142 = v12;
    v151 = v11;
    v128 = (char *)memmove(v17, v11, v18);
    v11 = v151;
    v12 = v142;
    v17 = v128;
    goto LABEL_159;
  }
  if ( v11 )
  {
LABEL_159:
    v152 = v17;
    j_j___libc_free_0(v11, v12);
    v17 = v152;
  }
  src = v17;
  v172 = &v17[v15];
  v173 = &v17[v14];
LABEL_17:
  v19 = sub_1389B50(&v166);
  v20 = v174;
  v21 = v19;
  v22 = v166 & 0xFFFFFFFFFFFFFFF8LL;
  v23 = v174;
  v24 = -1431655765
      * (unsigned int)((__int64)(v21
                               - ((v166 & 0xFFFFFFFFFFFFFFF8LL)
                                - 24LL * (*(_DWORD *)((v166 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
  if ( v24 > (v176 - v174) >> 3 )
  {
    v96 = v175;
    v97 = 8 * v24;
    v98 = 0;
    v99 = v175 - v174;
    if ( v24 )
    {
      v100 = sub_22077B0(8 * v24);
      v20 = v174;
      v96 = v175;
      v98 = (char *)v100;
      v23 = v174;
    }
    if ( v96 != v20 )
    {
      v101 = v98;
      do
      {
        if ( v101 )
          *(_QWORD *)v101 = *(_QWORD *)v23;
        v23 += 8;
        v101 += 8;
      }
      while ( v23 != v96 );
    }
    if ( v20 )
    {
      v149 = v98;
      j_j___libc_free_0(v20, v176 - v20);
      v98 = v149;
    }
    v174 = v98;
    v175 = &v98[v99];
    v176 = &v98[v97];
    v22 = v166 & 0xFFFFFFFFFFFFFFF8LL;
  }
  v25 = 0;
  v26 = (__int64 *)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF));
  v27 = sub_1389B50(&v166);
  while ( 1 )
  {
    if ( v25 != v8 )
      goto LABEL_19;
    v33 = *(__int64 ****)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    v180 = (unsigned __int8 *)v33;
    if ( v169 != *v33 )
    {
      v185[1] = 1;
      v185[0] = 3;
      v34 = *(_QWORD *)(a1 + 8);
      v183 = "nest";
      v180 = sub_1708970(v34, 47, (__int64)v33, v169, (__int64 *)&v183);
    }
    v35 = v172;
    if ( v172 == v173 )
    {
      sub_1287830((__int64)&src, v172, &v180);
    }
    else
    {
      if ( v172 )
      {
        *(_QWORD *)v172 = v180;
        v35 = v172;
      }
      v172 = v35 + 8;
    }
    v36 = v175;
    if ( v175 != v176 )
      break;
    sub_173FD70(&v174, v175, &v170);
LABEL_19:
    if ( v26 == (__int64 *)v27 )
      goto LABEL_40;
LABEL_20:
    v28 = v172;
    v29 = v173;
    v30 = *v26;
    if ( v172 == v173 )
    {
      v51 = src;
      v52 = v172 - (_BYTE *)src;
      v53 = (v172 - (_BYTE *)src) >> 3;
      if ( v53 == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v54 = 1;
      if ( v53 )
        v54 = v52 >> 3;
      v55 = __CFADD__(v54, v53);
      v56 = v54 + v53;
      if ( v55 )
      {
        v131 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v56 )
        {
          v57 = 0;
          v58 = 0;
          goto LABEL_70;
        }
        if ( v56 > 0xFFFFFFFFFFFFFFFLL )
          v56 = 0xFFFFFFFFFFFFFFFLL;
        v131 = 8 * v56;
      }
      v134 = v172 - (_BYTE *)src;
      v138 = src;
      v144 = v131;
      v132 = sub_22077B0(v131);
      v29 = v173;
      v51 = v138;
      v58 = (char *)v132;
      v52 = v134;
      v57 = (_BYTE *)(v132 + v144);
LABEL_70:
      if ( &v58[v52] )
        *(_QWORD *)&v58[v52] = v30;
      v59 = &v58[v52 + 8];
      if ( v52 > 0 )
      {
        v133 = (__int64)&v58[v52 + 8];
        v136 = v57;
        v146 = v29;
        v139 = v51;
        v95 = (char *)memmove(v58, v51, v52);
        v51 = v139;
        v29 = v146;
        v57 = v136;
        v59 = (char *)v133;
        v58 = v95;
      }
      else if ( !v51 )
      {
LABEL_74:
        src = v58;
        v172 = v59;
        v173 = v57;
        goto LABEL_24;
      }
      v137 = v59;
      v147 = v57;
      v140 = v58;
      j_j___libc_free_0(v51, v29 - (_BYTE *)v51);
      v59 = v137;
      v57 = v147;
      v58 = v140;
      goto LABEL_74;
    }
    if ( v172 )
    {
      *(_QWORD *)v172 = v30;
      v28 = v172;
    }
    v172 = v28 + 8;
LABEL_24:
    v31 = sub_1560230(&v167, v25);
    v32 = v175;
    v183 = (char *)v31;
    if ( v175 == v176 )
    {
      sub_17401C0(&v174, v175, &v183);
    }
    else
    {
      if ( v175 )
      {
        *(_QWORD *)v175 = v31;
        v32 = v175;
      }
      v175 = v32 + 8;
    }
    ++v25;
    v26 += 3;
  }
  if ( v175 )
  {
    *(_QWORD *)v175 = v170;
    v36 = v175;
  }
  v175 = v36 + 8;
  if ( v26 != (__int64 *)v27 )
    goto LABEL_20;
LABEL_40:
  v37 = 0;
  v178 = 0;
  v179 = 0;
  v177 = 0;
  v38 = *(unsigned int *)(v162 + 12);
  if ( !*(_DWORD *)(v162 + 12) )
    goto LABEL_41;
  v102 = 8 * v38;
  v103 = (char *)sub_22077B0(v102);
  v104 = v177;
  v37 = v103;
  if ( v178 - (_BYTE *)v177 > 0 )
  {
    memmove(v103, v177, v178 - (_BYTE *)v177);
    v129 = v179 - v104;
    goto LABEL_161;
  }
  if ( v177 )
  {
    v129 = v179 - (_BYTE *)v177;
LABEL_161:
    j_j___libc_free_0(v104, v129);
  }
  v178 = v37;
  v179 = &v37[v102];
  v177 = v37;
  v38 = *(unsigned int *)(v162 + 12);
LABEL_41:
  v39 = v37;
  v40 = *(_QWORD *)(v162 + 16);
  v41 = (_QWORD *)(v40 + 8 * v38);
  v42 = (_QWORD *)(v40 + 8);
  v43 = 0;
  v44 = v41;
  while ( 2 )
  {
    if ( v43 != v8 )
    {
LABEL_47:
      if ( v42 == v44 )
        break;
      goto LABEL_48;
    }
    if ( v179 == v39 )
    {
      sub_1277EB0((__int64)&v177, v39, &v169);
      v39 = v178;
      goto LABEL_47;
    }
    if ( v39 )
    {
      *(_QWORD *)v39 = v169;
      v39 = v178;
    }
    v39 += 8;
    v178 = v39;
    if ( v42 != v44 )
    {
LABEL_48:
      if ( v179 == v39 )
      {
        sub_1277EB0((__int64)&v177, v39, v42);
        v39 = v178;
      }
      else
      {
        if ( v39 )
        {
          *(_QWORD *)v39 = *v42;
          v39 = v178;
        }
        v39 += 8;
        v178 = v39;
      }
      ++v43;
      ++v42;
      continue;
    }
    break;
  }
  v60 = (__int64 *)sub_1644EA0(
                     **(__int64 ***)(v162 + 16),
                     v177,
                     (v39 - (_BYTE *)v177) >> 3,
                     *(_DWORD *)(v162 + 8) >> 8 != 0);
  v61 = *(__int64 ***)v165;
  if ( v61 != (__int64 **)sub_1646BA0(v60, 0) )
  {
    v62 = (__int64 **)sub_1646BA0(v60, 0);
    v165 = sub_15A4510((__int64 ***)v165, v62, 0);
  }
  v63 = v174;
  v64 = v175;
  v65 = sub_1560240(&v167);
  v66 = sub_1560250(&v167);
  v164 = sub_155FDB0(*(__int64 **)v162, v66, v65, v63, (v64 - v63) >> 3);
  v183 = v185;
  v184 = 0x100000000LL;
  v67 = v166 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v166 & 4) != 0 )
  {
    if ( *(char *)(v67 + 23) < 0 )
    {
      v105 = sub_1648A40(v166 & 0xFFFFFFFFFFFFFFF8LL);
      v107 = v105 + v106;
      if ( *(char *)(v67 + 23) >= 0 )
        v108 = v107 >> 4;
      else
        LODWORD(v108) = (v107 - sub_1648A40(v67)) >> 4;
      v109 = 0;
      v110 = 16LL * (unsigned int)v108;
      if ( (_DWORD)v108 )
      {
        do
        {
          v111 = 0;
          if ( *(char *)(v67 + 23) < 0 )
            v111 = sub_1648A40(v67);
          v112 = (__int64 *)(v109 + v111);
          v109 += 16;
          v113 = *((unsigned int *)v112 + 2);
          v114 = *v112;
          v115 = *((unsigned int *)v112 + 3);
          v116 = 3LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
          v182 = v114;
          v113 *= 24;
          v180 = (unsigned __int8 *)(v67 + v113 - 8 * v116);
          v181 = 0xAAAAAAAAAAAAAAABLL * ((24 * v115 - v113) >> 3);
          sub_1740580((__int64)&v183, (__int64)&v180);
        }
        while ( v110 != v109 );
      }
    }
  }
  else if ( *(char *)(v67 + 23) < 0 )
  {
    v68 = sub_1648A40(v166 & 0xFFFFFFFFFFFFFFF8LL);
    v70 = v68 + v69;
    if ( *(char *)(v67 + 23) < 0 )
      v70 -= sub_1648A40(v67);
    v71 = v70 >> 4;
    if ( (_DWORD)v71 )
    {
      v72 = 0;
      v73 = 16LL * (unsigned int)v71;
      do
      {
        v74 = 0;
        if ( *(char *)(v67 + 23) < 0 )
          v74 = sub_1648A40(v67);
        v75 = (__int64 *)(v72 + v74);
        v72 += 16;
        v76 = *((unsigned int *)v75 + 2);
        v77 = *v75;
        v78 = *((unsigned int *)v75 + 3);
        v79 = 3LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
        v182 = v77;
        v76 *= 24;
        v180 = (unsigned __int8 *)(v67 + v76 - 8 * v79);
        v181 = 0xAAAAAAAAAAAAAAABLL * ((24 * v78 - v76) >> 3);
        sub_1740580((__int64)&v183, (__int64)&v180);
      }
      while ( v73 != v72 );
    }
  }
  v80 = (__int64 *)v183;
  v81 = (__int64 *)src;
  v82 = &v183[56 * (unsigned int)v184];
  v83 = (v172 - (_BYTE *)src) >> 3;
  if ( *(_BYTE *)(v161 + 16) != 29 )
  {
    LOWORD(v182) = 257;
    v159 = *(_QWORD *)(*(_QWORD *)v165 + 24LL);
    if ( v183 == v82 )
    {
      v143 = (unsigned int)v184;
      v153 = (v172 - (_BYTE *)src) >> 3;
      v130 = sub_1648AB0(72, (int)v83 + 1, 16 * (int)v184);
      v126 = v83 + 1;
      v127 = v153;
      v49 = v130;
      if ( !v130 )
        goto LABEL_155;
      v122 = v80;
      v123 = v143;
    }
    else
    {
      v119 = (__int64)v183;
      v120 = 0;
      do
      {
        v121 = *(_QWORD *)(v119 + 40) - *(_QWORD *)(v119 + 32);
        v119 += 56;
        v120 += v121 >> 3;
      }
      while ( v82 != (char *)v119 );
      v141 = (__int64 *)&v183[56 * (unsigned int)v184];
      v150 = (unsigned int)v184;
      v49 = sub_1648AB0(72, (int)v83 + 1 + v120, 16 * (int)v184);
      if ( !v49 )
        goto LABEL_155;
      v122 = v80;
      v123 = v150;
      LODWORD(v124) = 0;
      do
      {
        v125 = v80[5] - v80[4];
        v80 += 7;
        v124 = (unsigned int)(v125 >> 3) + (unsigned int)v124;
      }
      while ( v141 != v80 );
      v126 = v83 + 1 + v124;
      v127 = v83 + v124;
    }
    v156 = v122;
    v157 = v123;
    sub_15F1EA0((__int64)v49, **(_QWORD **)(v159 + 16), 54, (__int64)&v49[-3 * v127 - 3], v126, 0);
    v49[7] = 0;
    sub_15F5B40((__int64)v49, v159, v165, v81, v83, (__int64)&v180, v156, v157);
LABEL_155:
    v88 = *(_WORD *)(v161 + 18) & 3 | *((_WORD *)v49 + 9) & 0xFFFC;
    *((_WORD *)v49 + 9) = v88;
    v89 = *(unsigned __int16 *)(v161 + 18);
    goto LABEL_99;
  }
  LOWORD(v182) = 257;
  v155 = *(_QWORD *)(v161 - 48);
  v148 = *(_QWORD *)(v161 - 24);
  v158 = *(_QWORD *)(*(_QWORD *)v165 + 24LL);
  if ( v183 == v82 )
  {
    v85 = 0;
  }
  else
  {
    v84 = (__int64)v183;
    v85 = 0;
    do
    {
      v86 = *(_QWORD *)(v84 + 40) - *(_QWORD *)(v84 + 32);
      v84 += 56;
      v85 += v86 >> 3;
    }
    while ( v82 != (char *)v84 );
  }
  v135 = (unsigned int)v184;
  v145 = v85 + v83 + 3;
  v87 = sub_1648AB0(72, v145, 16 * (int)v184);
  v49 = v87;
  if ( v87 )
  {
    sub_15F1EA0((__int64)v87, **(_QWORD **)(v158 + 16), 5, (__int64)&v87[-3 * v145], v145, 0);
    v49[7] = 0;
    sub_15F6500((__int64)v49, v158, v165, v155, v148, (__int64)&v180, v81, v83, v80, v135);
  }
  v88 = *((_WORD *)v49 + 9);
  v89 = *(unsigned __int16 *)(v161 + 18);
LABEL_99:
  v90 = v49 + 6;
  *((_WORD *)v49 + 9) = v88 & 0x8000 | v88 & 3 | (4 * ((v89 >> 2) & 0xDFFF));
  v49[7] = v164;
  v91 = *(unsigned __int8 **)(v161 + 48);
  v180 = v91;
  if ( !v91 )
  {
    if ( v90 == (__int64 *)&v180 )
      goto LABEL_103;
    v117 = v49[6];
    if ( !v117 )
      goto LABEL_103;
LABEL_144:
    sub_161E7C0((__int64)(v49 + 6), v117);
    goto LABEL_145;
  }
  sub_1623A60((__int64)&v180, (__int64)v91, 2);
  if ( v90 == (__int64 *)&v180 )
  {
    if ( v180 )
      sub_161E7C0((__int64)&v180, (__int64)v180);
    goto LABEL_103;
  }
  v117 = v49[6];
  if ( v117 )
    goto LABEL_144;
LABEL_145:
  v118 = v180;
  v49[6] = v180;
  if ( v118 )
    sub_1623210((__int64)&v180, v118, (__int64)(v49 + 6));
LABEL_103:
  v92 = (__int64)v183;
  v93 = &v183[56 * (unsigned int)v184];
  if ( v183 != v93 )
  {
    do
    {
      v94 = *((_QWORD *)v93 - 3);
      v93 -= 56;
      if ( v94 )
        j_j___libc_free_0(v94, *((_QWORD *)v93 + 6) - v94);
      if ( *(char **)v93 != v93 + 16 )
        j_j___libc_free_0(*(_QWORD *)v93, *((_QWORD *)v93 + 2) + 1LL);
    }
    while ( (char *)v92 != v93 );
    v93 = v183;
  }
  if ( v93 != v185 )
    _libc_free((unsigned __int64)v93);
  if ( v177 )
    j_j___libc_free_0(v177, v179 - (_BYTE *)v177);
  if ( v174 )
    j_j___libc_free_0(v174, v176 - v174);
  if ( src )
    j_j___libc_free_0(src, v173 - (_BYTE *)src);
  return (unsigned __int64)v49;
}
