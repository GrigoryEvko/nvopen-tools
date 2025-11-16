// Function: sub_1808150
// Address: 0x1808150
//
void __fastcall sub_1808150(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _QWORD *v9; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 **v14; // rdi
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r15
  _QWORD *v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned __int8 *v27; // rsi
  __int64 *v28; // rbx
  __int64 *v29; // r12
  __int64 v30; // rsi
  __int64 *v31; // r12
  __int64 *v32; // rbx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rbx
  _QWORD *v36; // rax
  __int64 v37; // rax
  unsigned __int8 *v38; // rsi
  _QWORD *v39; // rbx
  unsigned int v40; // r15d
  _QWORD *v41; // rax
  _QWORD *v42; // r12
  unsigned __int64 *v43; // rbx
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rcx
  unsigned __int8 *v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // r15
  _QWORD *v51; // rax
  _QWORD *v52; // rbx
  unsigned __int64 *v53; // r12
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  unsigned __int8 v58; // al
  __int64 v59; // r9
  __int64 v60; // r10
  __int64 v61; // rcx
  __int64 v62; // rax
  _QWORD *v63; // r15
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r15
  _QWORD *v69; // rax
  _QWORD *v70; // r15
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rax
  unsigned __int8 *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rax
  double v81; // xmm4_8
  double v82; // xmm5_8
  __int64 v83; // rbx
  _QWORD *v84; // rax
  __int64 v85; // rdx
  unsigned __int8 *v86; // rsi
  unsigned int v87; // eax
  __int64 v88; // r15
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 **v93; // rsi
  __int64 v94; // r9
  __int64 v95; // r10
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rsi
  __int64 v99; // rax
  _QWORD *v100; // rax
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rdx
  unsigned __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rdx
  unsigned __int64 v108; // rax
  __int64 v109; // rax
  unsigned __int64 v110; // rsi
  __int64 v111; // rax
  __int64 v112; // rsi
  __int64 v113; // rdx
  unsigned __int8 *v114; // rsi
  __int64 v115; // rax
  __int64 v116; // r10
  __int64 *v117; // r15
  __int64 v118; // rcx
  __int64 v119; // rax
  __int64 v120; // rsi
  __int64 v121; // rdx
  unsigned __int8 *v122; // rsi
  __int64 v123; // rax
  __int64 v124; // r9
  __int64 *v125; // r15
  __int64 v126; // rcx
  __int64 v127; // rax
  unsigned int v128; // edx
  int v129; // eax
  __int64 v130; // rax
  __int64 v131; // r10
  __int64 v132; // r9
  __int64 v133; // rsi
  __int64 v134; // rax
  __int64 *j; // [rsp-138h] [rbp-138h]
  __int64 v136; // [rsp-130h] [rbp-130h]
  _QWORD *v137; // [rsp-128h] [rbp-128h]
  __int64 v138; // [rsp-128h] [rbp-128h]
  __int64 v139; // [rsp-120h] [rbp-120h]
  __int64 *v140; // [rsp-120h] [rbp-120h]
  __int64 *v141; // [rsp-120h] [rbp-120h]
  __int64 v142; // [rsp-120h] [rbp-120h]
  __int64 v143; // [rsp-120h] [rbp-120h]
  __int64 v144; // [rsp-120h] [rbp-120h]
  __int64 v145; // [rsp-120h] [rbp-120h]
  __int64 v146; // [rsp-120h] [rbp-120h]
  int v147; // [rsp-120h] [rbp-120h]
  __int64 *v148; // [rsp-120h] [rbp-120h]
  __int64 v149; // [rsp-120h] [rbp-120h]
  __int64 *v150; // [rsp-118h] [rbp-118h]
  __int64 v151; // [rsp-118h] [rbp-118h]
  unsigned __int64 *v152; // [rsp-118h] [rbp-118h]
  __int64 v153; // [rsp-110h] [rbp-110h]
  __int64 v154; // [rsp-110h] [rbp-110h]
  __int64 v155; // [rsp-110h] [rbp-110h]
  __int64 v156; // [rsp-110h] [rbp-110h]
  __int64 i; // [rsp-108h] [rbp-108h]
  __int64 v158; // [rsp-108h] [rbp-108h]
  __int64 v159; // [rsp-108h] [rbp-108h]
  unsigned int v160; // [rsp-108h] [rbp-108h]
  __int64 v161; // [rsp-100h] [rbp-100h]
  __int64 *v162; // [rsp-100h] [rbp-100h]
  char v163; // [rsp-F0h] [rbp-F0h]
  __int64 v164; // [rsp-F0h] [rbp-F0h]
  __int64 v165; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v166; // [rsp-F0h] [rbp-F0h]
  __int64 v167; // [rsp-F0h] [rbp-F0h]
  __int64 v168; // [rsp-F0h] [rbp-F0h]
  __int64 v169; // [rsp-F0h] [rbp-F0h]
  __int64 v170[2]; // [rsp-E8h] [rbp-E8h] BYREF
  __int16 v171; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 *v172; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v173; // [rsp-C0h] [rbp-C0h]
  __int16 v174; // [rsp-B8h] [rbp-B8h]
  unsigned __int8 *v175[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v176; // [rsp-98h] [rbp-98h]
  unsigned __int8 *v177; // [rsp-88h] [rbp-88h] BYREF
  __int64 v178; // [rsp-80h] [rbp-80h]
  __int64 *v179; // [rsp-78h] [rbp-78h]
  _QWORD *v180; // [rsp-70h] [rbp-70h]
  __int64 v181; // [rsp-68h] [rbp-68h]
  int v182; // [rsp-60h] [rbp-60h]
  __int64 v183; // [rsp-58h] [rbp-58h]
  __int64 v184; // [rsp-50h] [rbp-50h]

  if ( !byte_4FA7BA0 || !*(_DWORD *)(a1 + 3712) )
    return;
  v11 = *(_QWORD *)(a1 + 3160);
  v12 = 32LL * *(unsigned int *)(a1 + 3168);
  for ( i = v11 + v12; i != v11; v11 += 32 )
  {
    v18 = *(_QWORD *)v11;
    v19 = (_QWORD *)sub_16498A0(*(_QWORD *)v11);
    v177 = 0;
    v180 = v19;
    v181 = 0;
    v182 = 0;
    v183 = 0;
    v184 = 0;
    v178 = *(_QWORD *)(v18 + 40);
    v179 = (__int64 *)(v18 + 24);
    v20 = *(unsigned __int8 **)(v18 + 48);
    v175[0] = v20;
    if ( v20 )
    {
      sub_1623A60((__int64)v175, (__int64)v20, 2);
      if ( v177 )
        sub_161E7C0((__int64)&v177, (__int64)v177);
      v177 = v175[0];
      if ( v175[0] )
        sub_1623210((__int64)v175, v175[0], (__int64)&v177);
    }
    v15 = *(_QWORD *)(v11 + 8);
    v21 = *(_QWORD *)(a1 + 488);
    v163 = *(_BYTE *)(v11 + 24);
    v161 = *(_QWORD *)(v11 + 16);
    v174 = 257;
    v14 = *(__int64 ***)v15;
    if ( v21 != *(_QWORD *)v15 )
    {
      if ( *(_BYTE *)(v15 + 16) <= 0x10u )
      {
        v13 = sub_15A4A70((__int64 ***)v15, v21);
        v14 = *(__int64 ***)(a1 + 488);
        v15 = v13;
      }
      else
      {
        v176 = 257;
        v22 = sub_15FDFF0(v15, v21, (__int64)v175, 0);
        v15 = v22;
        if ( v178 )
        {
          v150 = v179;
          sub_157E9D0(v178 + 40, v22);
          v23 = *v150;
          v24 = *(_QWORD *)(v15 + 24) & 7LL;
          *(_QWORD *)(v15 + 32) = v150;
          v23 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v15 + 24) = v23 | v24;
          *(_QWORD *)(v23 + 8) = v15 + 24;
          *v150 = *v150 & 7 | (v15 + 24);
        }
        sub_164B780(v15, (__int64 *)&v172);
        if ( v177 )
        {
          v170[0] = (__int64)v177;
          sub_1623A60((__int64)v170, (__int64)v177, 2);
          v25 = *(_QWORD *)(v15 + 48);
          v26 = v15 + 48;
          if ( v25 )
          {
            sub_161E7C0(v15 + 48, v25);
            v26 = v15 + 48;
          }
          v27 = (unsigned __int8 *)v170[0];
          *(_QWORD *)(v15 + 48) = v170[0];
          if ( v27 )
            sub_1623210((__int64)v170, v27, v26);
        }
        v14 = *(__int64 ***)(a1 + 488);
      }
    }
    v16 = sub_15A0680((__int64)v14, v161, 0);
    v176 = 257;
    v17 = *(_QWORD *)(a1 + 3136);
    if ( v163 )
      v17 = *(_QWORD *)(a1 + 3128);
    v172 = (unsigned __int8 *)v15;
    v173 = v16;
    sub_1285290((__int64 *)&v177, *(_QWORD *)(v17 + 24), v17, (int)&v172, 2, (__int64)v175, 0);
    if ( v177 )
      sub_161E7C0((__int64)&v177, (__int64)v177);
  }
  v34 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( !v34 )
    BUG();
  v35 = *(_QWORD *)(v34 + 24);
  if ( !v35 )
  {
    v9 = (_QWORD *)sub_16498A0(0);
    v177 = 0;
    v179 = 0;
    v180 = v9;
    v181 = 0;
    v182 = 0;
    v183 = 0;
    v184 = 0;
    v178 = 0;
    BUG();
  }
  v36 = (_QWORD *)sub_16498A0(v35 - 24);
  v177 = 0;
  v180 = v36;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v37 = *(_QWORD *)(v35 + 16);
  v179 = (__int64 *)v35;
  v178 = v37;
  v38 = *(unsigned __int8 **)(v35 + 24);
  v175[0] = v38;
  if ( v38 )
  {
    sub_1623A60((__int64)v175, (__int64)v38, 2);
    if ( v177 )
      sub_161E7C0((__int64)&v177, (__int64)v177);
    v177 = v175[0];
    if ( v175[0] )
      sub_1623210((__int64)v175, v175[0], (__int64)&v177);
  }
  v39 = *(_QWORD **)(a1 + 488);
  v174 = 257;
  v40 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v178 + 56) + 40LL)) + 4);
  v176 = 257;
  v41 = sub_1648A60(64, 1u);
  v42 = v41;
  if ( v41 )
    sub_15F8BC0((__int64)v41, v39, v40, 0, (__int64)v175, 0);
  if ( v178 )
  {
    v43 = (unsigned __int64 *)v179;
    sub_157E9D0(v178 + 40, (__int64)v42);
    v44 = v42[3];
    v45 = *v43;
    v42[4] = v43;
    v45 &= 0xFFFFFFFFFFFFFFF8LL;
    v42[3] = v45 | v44 & 7;
    *(_QWORD *)(v45 + 8) = v42 + 3;
    *v43 = *v43 & 7 | (unsigned __int64)(v42 + 3);
  }
  sub_164B780((__int64)v42, (__int64 *)&v172);
  v48 = v177;
  if ( v177 )
  {
    v170[0] = (__int64)v177;
    sub_1623A60((__int64)v170, (__int64)v177, 2);
    v49 = v42[6];
    if ( v49 )
      sub_161E7C0((__int64)(v42 + 6), v49);
    v48 = (unsigned __int8 *)v170[0];
    v42[6] = v170[0];
    if ( v48 )
      sub_1623210((__int64)v170, v48, (__int64)(v42 + 6));
  }
  *(_QWORD *)(a1 + 3752) = v42;
  v50 = sub_15A06D0(*(__int64 ***)(a1 + 488), (__int64)v48, v46, v47);
  v176 = 257;
  v51 = sub_1648A60(64, 2u);
  v52 = v51;
  if ( v51 )
    sub_15F9650((__int64)v51, v50, (__int64)v42, 0, 0);
  if ( v178 )
  {
    v53 = (unsigned __int64 *)v179;
    sub_157E9D0(v178 + 40, (__int64)v52);
    v54 = v52[3];
    v55 = *v53;
    v52[4] = v53;
    v55 &= 0xFFFFFFFFFFFFFFF8LL;
    v52[3] = v55 | v54 & 7;
    *(_QWORD *)(v55 + 8) = v52 + 3;
    *v53 = *v53 & 7 | (unsigned __int64)(v52 + 3);
  }
  sub_164B780((__int64)v52, (__int64 *)v175);
  if ( v177 )
  {
    v172 = v177;
    sub_1623A60((__int64)&v172, (__int64)v177, 2);
    v56 = v52[6];
    if ( v56 )
      sub_161E7C0((__int64)(v52 + 6), v56);
    v57 = v172;
    v52[6] = v172;
    if ( v57 )
      sub_1623210((__int64)&v172, v57, (__int64)(v52 + 6));
  }
  sub_15F8A20(*(_QWORD *)(a1 + 3752), 0x20u);
  if ( v177 )
    sub_161E7C0((__int64)&v177, (__int64)v177);
  v162 = *(__int64 **)(a1 + 3704);
  for ( j = &v162[*(unsigned int *)(a1 + 3712)]; j != v162; ++v162 )
  {
    v83 = *v162;
    v84 = (_QWORD *)sub_16498A0(*v162);
    v177 = 0;
    v180 = v84;
    v181 = 0;
    v182 = 0;
    v183 = 0;
    v184 = 0;
    v178 = *(_QWORD *)(v83 + 40);
    v179 = (__int64 *)(v83 + 24);
    v86 = *(unsigned __int8 **)(v83 + 48);
    v175[0] = v86;
    if ( v86 )
    {
      sub_1623A60((__int64)v175, (__int64)v86, 2);
      if ( v177 )
        sub_161E7C0((__int64)&v177, (__int64)v177);
      v86 = v175[0];
      v177 = v175[0];
      if ( v175[0] )
        sub_1623210((__int64)v175, v175[0], (__int64)&v177);
    }
    v87 = 32;
    if ( (unsigned int)(1 << *(_WORD *)(v83 + 18)) > 0x3F )
      v87 = (unsigned int)(1 << *(_WORD *)(v83 + 18)) >> 1;
    v160 = v87;
    v151 = sub_15A06D0(*(__int64 ***)(a1 + 488), (__int64)v86, v85, (unsigned int)(1 << *(_WORD *)(v83 + 18)) >> 1);
    v154 = sub_15A0680(*(_QWORD *)(a1 + 488), 32, 0);
    v88 = sub_15A0680(*(_QWORD *)(a1 + 488), 31, 0);
    v89 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
    v90 = *(_QWORD *)(v83 + 56);
    v139 = v89;
    v166 = (unsigned int)sub_15A9FE0(v89, v90);
    v91 = sub_127FA20(v139, v90);
    v174 = 257;
    v92 = sub_15A0680(
            *(_QWORD *)(a1 + 488),
            (unsigned int)v166 * (unsigned int)((v166 + ((unsigned __int64)(v91 + 7) >> 3) - 1) / v166),
            0);
    v93 = *(__int64 ***)(a1 + 488);
    v171 = 257;
    v94 = *(_QWORD *)(v83 - 24);
    v95 = v92;
    if ( v93 != *(__int64 ***)v94 )
    {
      v167 = v92;
      if ( *(_BYTE *)(v94 + 16) > 0x10u )
      {
        v176 = 257;
        v130 = sub_15FE0A0((_QWORD *)v94, (__int64)v93, 0, (__int64)v175, 0);
        v131 = v167;
        v132 = v130;
        if ( v178 )
        {
          v138 = v167;
          v168 = v130;
          v148 = v179;
          sub_157E9D0(v178 + 40, v130);
          v132 = v168;
          v131 = v138;
          v133 = *v148;
          v134 = *(_QWORD *)(v168 + 24);
          *(_QWORD *)(v168 + 32) = v148;
          v133 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v168 + 24) = v133 | v134 & 7;
          *(_QWORD *)(v133 + 8) = v168 + 24;
          *v148 = *v148 & 7 | (v168 + 24);
        }
        v149 = v131;
        v169 = v132;
        sub_164B780(v132, v170);
        sub_12A86E0((__int64 *)&v177, v169);
        v95 = v149;
        v94 = v169;
      }
      else
      {
        v96 = sub_15A4750((__int64 ***)v94, v93, 0);
        v95 = v167;
        v94 = v96;
      }
    }
    if ( *(_BYTE *)(v94 + 16) > 0x10u || *(_BYTE *)(v95 + 16) > 0x10u )
    {
      v176 = 257;
      v97 = sub_15FB440(15, (__int64 *)v94, v95, (__int64)v175, 0);
      v164 = v97;
      if ( v178 )
      {
        v140 = v179;
        sub_157E9D0(v178 + 40, v97);
        v98 = *v140;
        v99 = *(_QWORD *)(v164 + 24);
        *(_QWORD *)(v164 + 32) = v140;
        v98 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v164 + 24) = v98 | v99 & 7;
        *(_QWORD *)(v98 + 8) = v164 + 24;
        *v140 = *v140 & 7 | (v164 + 24);
      }
      sub_164B780(v164, (__int64 *)&v172);
      sub_12A86E0((__int64 *)&v177, v164);
    }
    else
    {
      v164 = sub_15A2C20((__int64 *)v94, v95, 0, 0, *(double *)a2.m128_u64, a3, a4);
    }
    v174 = 257;
    v58 = *(_BYTE *)(v88 + 16);
    if ( v58 > 0x10u )
    {
LABEL_123:
      v176 = 257;
      v123 = sub_15FB440(26, (__int64 *)v164, v88, (__int64)v175, 0);
      v124 = v123;
      if ( v178 )
      {
        v125 = v179;
        v145 = v123;
        sub_157E9D0(v178 + 40, v123);
        v124 = v145;
        v126 = *v125;
        v127 = *(_QWORD *)(v145 + 24);
        *(_QWORD *)(v145 + 32) = v125;
        v126 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v145 + 24) = v126 | v127 & 7;
        *(_QWORD *)(v126 + 8) = v145 + 24;
        *v125 = *v125 & 7 | (v145 + 24);
      }
      v146 = v124;
      sub_164B780(v124, (__int64 *)&v172);
      sub_12A86E0((__int64 *)&v177, v146);
      v59 = v146;
      goto LABEL_66;
    }
    if ( v58 == 13 )
    {
      v128 = *(_DWORD *)(v88 + 32);
      if ( v128 <= 0x40 )
      {
        v59 = v164;
        if ( *(_QWORD *)(v88 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v128) )
          goto LABEL_66;
      }
      else
      {
        v147 = *(_DWORD *)(v88 + 32);
        v129 = sub_16A58F0(v88 + 24);
        v59 = v164;
        if ( v147 == v129 )
          goto LABEL_66;
      }
    }
    if ( *(_BYTE *)(v164 + 16) > 0x10u )
      goto LABEL_123;
    v59 = sub_15A2CF0((__int64 *)v164, v88, *(double *)a2.m128_u64, a3, a4);
LABEL_66:
    v174 = 257;
    if ( *(_BYTE *)(v154 + 16) > 0x10u || *(_BYTE *)(v59 + 16) > 0x10u )
    {
      v176 = 257;
      v115 = sub_15FB440(13, (__int64 *)v154, v59, (__int64)v175, 0);
      v116 = v115;
      if ( v178 )
      {
        v117 = v179;
        v142 = v115;
        sub_157E9D0(v178 + 40, v115);
        v116 = v142;
        v118 = *v117;
        v119 = *(_QWORD *)(v142 + 24);
        *(_QWORD *)(v142 + 32) = v117;
        v118 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v142 + 24) = v118 | v119 & 7;
        *(_QWORD *)(v118 + 8) = v142 + 24;
        *v117 = *v117 & 7 | (v142 + 24);
      }
      v143 = v116;
      sub_164B780(v116, (__int64 *)&v172);
      v60 = v143;
      if ( v177 )
      {
        v170[0] = (__int64)v177;
        sub_1623A60((__int64)v170, (__int64)v177, 2);
        v60 = v143;
        v120 = *(_QWORD *)(v143 + 48);
        v121 = v143 + 48;
        if ( v120 )
        {
          sub_161E7C0(v143 + 48, v120);
          v60 = v143;
          v121 = v143 + 48;
        }
        v122 = (unsigned __int8 *)v170[0];
        *(_QWORD *)(v60 + 48) = v170[0];
        if ( v122 )
        {
          v144 = v60;
          sub_1623210((__int64)v170, v122, v121);
          v60 = v144;
        }
      }
    }
    else
    {
      v60 = sub_15A2B60((__int64 *)v154, v59, 0, 0, *(double *)a2.m128_u64, a3, a4);
    }
    v61 = v154;
    v153 = v60;
    v176 = 257;
    v62 = sub_12AA0C0((__int64 *)&v177, 0x21u, (_BYTE *)v60, v61, (__int64)v175);
    v174 = 257;
    if ( *(_BYTE *)(v62 + 16) > 0x10u || *(_BYTE *)(v153 + 16) > 0x10u || *(_BYTE *)(v151 + 16) > 0x10u )
    {
      v141 = (__int64 *)v153;
      v155 = v62;
      v176 = 257;
      v100 = sub_1648A60(56, 3u);
      v63 = v100;
      if ( v100 )
      {
        v136 = v155;
        v137 = v100 - 9;
        v156 = (__int64)v100;
        sub_15F1EA0((__int64)v100, *v141, 55, (__int64)(v100 - 9), 3, 0);
        if ( *(v63 - 9) )
        {
          v101 = *(v63 - 8);
          v102 = *(v63 - 7) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v102 = v101;
          if ( v101 )
            *(_QWORD *)(v101 + 16) = *(_QWORD *)(v101 + 16) & 3LL | v102;
        }
        *(v63 - 9) = v136;
        v103 = *(_QWORD *)(v136 + 8);
        *(v63 - 8) = v103;
        if ( v103 )
          *(_QWORD *)(v103 + 16) = (unsigned __int64)(v63 - 8) | *(_QWORD *)(v103 + 16) & 3LL;
        *(v63 - 7) = (v136 + 8) | *(v63 - 7) & 3LL;
        *(_QWORD *)(v136 + 8) = v137;
        if ( *(v63 - 6) )
        {
          v104 = *(v63 - 5);
          v105 = *(v63 - 4) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v105 = v104;
          if ( v104 )
            *(_QWORD *)(v104 + 16) = *(_QWORD *)(v104 + 16) & 3LL | v105;
        }
        *(v63 - 6) = v141;
        v106 = v141[1];
        *(v63 - 5) = v106;
        if ( v106 )
          *(_QWORD *)(v106 + 16) = (unsigned __int64)(v63 - 5) | *(_QWORD *)(v106 + 16) & 3LL;
        *(v63 - 4) = (unsigned __int64)(v141 + 1) | *(v63 - 4) & 3LL;
        v141[1] = (__int64)(v63 - 6);
        if ( *(v63 - 3) )
        {
          v107 = *(v63 - 2);
          v108 = *(v63 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v108 = v107;
          if ( v107 )
            *(_QWORD *)(v107 + 16) = *(_QWORD *)(v107 + 16) & 3LL | v108;
        }
        *(v63 - 3) = v151;
        if ( v151 )
        {
          v109 = *(_QWORD *)(v151 + 8);
          *(v63 - 2) = v109;
          if ( v109 )
            *(_QWORD *)(v109 + 16) = (unsigned __int64)(v63 - 2) | *(_QWORD *)(v109 + 16) & 3LL;
          *(v63 - 1) = (v151 + 8) | *(v63 - 1) & 3LL;
          *(_QWORD *)(v151 + 8) = v63 - 3;
        }
        sub_164B780((__int64)v63, (__int64 *)v175);
      }
      else
      {
        v156 = 0;
      }
      if ( v178 )
      {
        v152 = (unsigned __int64 *)v179;
        sub_157E9D0(v178 + 40, (__int64)v63);
        v110 = *v152;
        v111 = v63[3] & 7LL;
        v63[4] = v152;
        v110 &= 0xFFFFFFFFFFFFFFF8LL;
        v63[3] = v110 | v111;
        *(_QWORD *)(v110 + 8) = v63 + 3;
        *v152 = *v152 & 7 | (unsigned __int64)(v63 + 3);
      }
      sub_164B780(v156, (__int64 *)&v172);
      if ( v177 )
      {
        v175[0] = v177;
        sub_1623A60((__int64)v175, (__int64)v177, 2);
        v112 = v63[6];
        v113 = (__int64)(v63 + 6);
        if ( v112 )
        {
          sub_161E7C0((__int64)(v63 + 6), v112);
          v113 = (__int64)(v63 + 6);
        }
        v114 = v175[0];
        v63[6] = v175[0];
        if ( v114 )
          sub_1623210((__int64)v175, v114, v113);
      }
    }
    else
    {
      v63 = (_QWORD *)sub_15A2DC0(v62, (__int64 *)v153, v151, 0);
    }
    v64 = *(_QWORD *)(a1 + 488);
    v176 = 257;
    v65 = sub_15A0680(v64, v160 + 32, 0);
    v66 = sub_12899C0((__int64 *)&v177, v65, (__int64)v63, (__int64)v175, 0, 0);
    v176 = 257;
    v67 = sub_12899C0((__int64 *)&v177, v164, v66, (__int64)v175, 0, 0);
    v176 = 257;
    v68 = v67;
    v69 = (_QWORD *)sub_1643330(v180);
    v70 = sub_17CEAE0((__int64 *)&v177, v69, v68, (__int64 *)v175);
    sub_15F8A20((__int64)v70, v160);
    v71 = *(_QWORD *)(a1 + 488);
    v176 = 257;
    v72 = sub_15A0680(v71, v160, 0);
    v73 = *(_QWORD *)(a1 + 488);
    v158 = v72;
    v174 = 257;
    v74 = sub_12AA3B0((__int64 *)&v177, 0x2Du, (__int64)v70, v73, (__int64)&v172);
    v75 = (unsigned __int8 *)sub_12899C0((__int64 *)&v177, v74, v158, (__int64)v175, 0, 0);
    v76 = *(_QWORD *)(a1 + 3144);
    v176 = 257;
    v172 = v75;
    v173 = v164;
    v159 = (__int64)v75;
    sub_1285290((__int64 *)&v177, *(_QWORD *)(v76 + 24), v76, (int)&v172, 2, (__int64)v175, 0);
    v77 = *(_QWORD *)(a1 + 3752);
    v78 = *(_QWORD *)(a1 + 488);
    v176 = 257;
    v165 = v77;
    v79 = sub_12AA3B0((__int64 *)&v177, 0x2Du, (__int64)v70, v78, (__int64)v175);
    sub_12A8F50((__int64 *)&v177, v79, v165, 0);
    v176 = 257;
    v80 = sub_12AA3B0((__int64 *)&v177, 0x2Eu, v159, *(_QWORD *)v83, (__int64)v175);
    sub_164D160(v83, v80, a2, a3, a4, a5, v81, v82, a8, a9);
    sub_15F20C0((_QWORD *)v83);
    if ( v177 )
      sub_161E7C0((__int64)&v177, (__int64)v177);
  }
  v28 = *(__int64 **)(a1 + 816);
  v29 = &v28[*(unsigned int *)(a1 + 824)];
  while ( v29 != v28 )
  {
    v30 = *v28++;
    sub_1805950((unsigned __int8 **)a1, v30, *(_QWORD *)(a1 + 3752));
  }
  v31 = *(__int64 **)(a1 + 3728);
  v32 = &v31[*(unsigned int *)(a1 + 3736)];
  while ( v32 != v31 )
  {
    v33 = *v31++;
    sub_1805950((unsigned __int8 **)a1, v33, *(_QWORD *)(v33 - 24LL * (*(_DWORD *)(v33 + 20) & 0xFFFFFFF)));
  }
}
