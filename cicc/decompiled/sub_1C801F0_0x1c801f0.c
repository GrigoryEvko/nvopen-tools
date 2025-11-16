// Function: sub_1C801F0
// Address: 0x1c801f0
//
void __fastcall sub_1C801F0(
        __int64 *a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        _QWORD *a15,
        __int64 a16)
{
  unsigned int v17; // eax
  unsigned int v18; // ecx
  unsigned __int64 v19; // r13
  unsigned int v20; // ebx
  bool v21; // r15
  unsigned int v22; // ebx
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  _QWORD *v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r13
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 **v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rax
  char *v35; // rsi
  __int64 v36; // r13
  __int64 v37; // rax
  int v38; // r14d
  __int64 v39; // rdx
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // r12
  __int64 v42; // rax
  _QWORD *v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // rdx
  _BYTE *v47; // r11
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r13
  _QWORD *v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 v55; // rbx
  _QWORD *v56; // rcx
  __int64 v57; // rax
  __int64 *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 *v61; // r11
  __int64 v62; // rax
  unsigned __int8 *v63; // rdi
  __int64 *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rdx
  unsigned __int8 *v69; // rsi
  __int64 *v70; // r14
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  __int64 v75; // rax
  __int64 v76; // r14
  __int64 **v77; // rdx
  __int64 v78; // r9
  _QWORD *v79; // r9
  __int64 v80; // rax
  _QWORD *v81; // rax
  double v82; // xmm4_8
  double v83; // xmm5_8
  _BYTE *v84; // r10
  __int64 v85; // rdx
  _QWORD *v86; // rax
  int v87; // r8d
  int v88; // r9d
  __int64 v89; // r13
  __int64 v90; // rax
  int v91; // r13d
  _QWORD *v92; // rax
  __int64 v93; // r13
  int v94; // r8d
  __int64 v95; // rax
  __int64 *v96; // rax
  __int64 *v97; // rax
  int v98; // r8d
  __int64 *v99; // r10
  __int64 *v100; // rax
  __int64 *v101; // rcx
  __int64 v102; // rdx
  __int64 *v103; // rax
  __int64 *v104; // rax
  __int64 *v105; // r15
  _QWORD *v106; // rax
  __int64 v107; // r14
  __int64 v108; // r13
  _QWORD *v109; // rdi
  __int64 v110; // rax
  double v111; // xmm4_8
  double v112; // xmm5_8
  __int64 v113; // r14
  __int64 v114; // rax
  double v115; // xmm4_8
  double v116; // xmm5_8
  __int64 v117; // rax
  __int64 v118; // r9
  __int64 *v119; // r14
  __int64 v120; // rsi
  __int64 v121; // rax
  __int64 v122; // rsi
  __int64 v123; // r14
  unsigned __int8 *v124; // rsi
  __int64 v125; // rax
  __int64 *v126; // rax
  __int64 v127; // r9
  __int64 v128; // rax
  __int64 v129; // rsi
  __int64 v130; // rsi
  __int64 v131; // rdx
  unsigned __int8 *v132; // rsi
  int v133; // [rsp+8h] [rbp-178h]
  int v134; // [rsp+Ch] [rbp-174h]
  unsigned __int64 v135; // [rsp+10h] [rbp-170h]
  unsigned int v136; // [rsp+18h] [rbp-168h]
  __int64 v137; // [rsp+18h] [rbp-168h]
  __int64 v138; // [rsp+20h] [rbp-160h]
  _BYTE *v139; // [rsp+20h] [rbp-160h]
  __int64 v140; // [rsp+20h] [rbp-160h]
  _BYTE *v141; // [rsp+20h] [rbp-160h]
  unsigned __int8 v142; // [rsp+20h] [rbp-160h]
  __int64 v144; // [rsp+28h] [rbp-158h]
  __int64 v145; // [rsp+30h] [rbp-150h]
  __int64 v146; // [rsp+30h] [rbp-150h]
  __int64 v147; // [rsp+30h] [rbp-150h]
  __int64 v148; // [rsp+38h] [rbp-148h]
  int v149; // [rsp+38h] [rbp-148h]
  __int64 *v150; // [rsp+38h] [rbp-148h]
  __int64 *v151; // [rsp+48h] [rbp-138h]
  __int64 v152; // [rsp+48h] [rbp-138h]
  __int64 *v153; // [rsp+48h] [rbp-138h]
  __int64 *v154; // [rsp+48h] [rbp-138h]
  __int64 v155; // [rsp+48h] [rbp-138h]
  _QWORD *v156; // [rsp+48h] [rbp-138h]
  _QWORD *v157; // [rsp+48h] [rbp-138h]
  __int64 v158; // [rsp+48h] [rbp-138h]
  __int64 v159; // [rsp+48h] [rbp-138h]
  __int64 v160; // [rsp+48h] [rbp-138h]
  __int64 v163; // [rsp+60h] [rbp-120h]
  char *v165; // [rsp+78h] [rbp-108h] BYREF
  __int64 v166[2]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v167; // [rsp+90h] [rbp-F0h]
  const char *v168; // [rsp+A0h] [rbp-E0h] BYREF
  char v169; // [rsp+B0h] [rbp-D0h]
  char v170; // [rsp+B1h] [rbp-CFh]
  __int64 v171[2]; // [rsp+C0h] [rbp-C0h] BYREF
  char v172; // [rsp+D0h] [rbp-B0h]
  char v173; // [rsp+D1h] [rbp-AFh]
  unsigned __int8 *v174; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v175; // [rsp+E8h] [rbp-98h]
  __int64 v176[2]; // [rsp+F0h] [rbp-90h] BYREF
  char *v177; // [rsp+100h] [rbp-80h] BYREF
  __int64 v178; // [rsp+108h] [rbp-78h]
  __int64 *v179; // [rsp+110h] [rbp-70h]
  _QWORD *v180; // [rsp+118h] [rbp-68h]
  __int64 v181; // [rsp+120h] [rbp-60h]
  int v182; // [rsp+128h] [rbp-58h]
  __int64 v183; // [rsp+130h] [rbp-50h]
  __int64 v184; // [rsp+138h] [rbp-48h]

  v17 = 1;
  if ( a5 <= 1 )
  {
    v110 = sub_1643330(a15);
    sub_1C7F460(a1, v110, a2, a3, a4, a6, a7, a8, a9, a10, v111, v112, a13, a14, (__int64)a15, a16);
    return;
  }
  do
  {
    v18 = v17;
    v17 *= 2;
  }
  while ( dword_4FBD640 > v17 );
  v19 = (v18 | a5) & (unsigned __int64)-(__int64)(v18 | a5);
  if ( *(_BYTE *)(a4 + 16) == 13 )
  {
    v20 = *(_DWORD *)(a4 + 32);
    v21 = v20 <= 0x40 ? *(_QWORD *)(a4 + 24) == 0 : v20 == (unsigned int)sub_16A57B0(a4 + 24);
    v22 = v19;
    if ( v21 )
    {
      if ( (unsigned int)v19 > 1 )
        goto LABEL_8;
LABEL_29:
      v145 = a3;
      v36 = a4;
      goto LABEL_30;
    }
  }
  v21 = 0;
  v22 = (v19 | 0x10) & -(v19 | 0x10);
  if ( v22 <= 1 )
    goto LABEL_29;
LABEL_8:
  v23 = *(_QWORD *)a3;
  if ( *(_BYTE *)(a3 + 16) == 13 )
  {
    v24 = *(_QWORD *)(a3 + 24);
    if ( *(_DWORD *)(a3 + 32) > 0x40u )
      v24 = *(_QWORD *)v24;
    v145 = sub_15A0680(v23, (unsigned int)(v24 / v22), 0);
  }
  else
  {
    v177 = "udiv";
    LOWORD(v179) = 259;
    v42 = sub_15A0680(v23, v22, 0);
    v145 = sub_15FB440(17, (__int64 *)a3, v42, (__int64)&v177, (__int64)a1);
  }
  v25 = (_QWORD *)sub_16498A0((__int64)a1);
  v26 = (unsigned __int8 *)a1[6];
  v177 = 0;
  v180 = v25;
  v27 = a1[5];
  v181 = 0;
  v178 = v27;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v179 = a1 + 3;
  v174 = v26;
  if ( v26 )
  {
    sub_1623A60((__int64)&v174, (__int64)v26, 2);
    if ( v177 )
      sub_161E7C0((__int64)&v177, (__int64)v177);
    v177 = (char *)v174;
    if ( v174 )
      sub_1623210((__int64)&v174, v174, (__int64)&v177);
  }
  if ( v22 > 4 )
  {
    v75 = sub_1643350(v180);
    v170 = 1;
    v76 = sub_159C470(v75, 0x101010101LL, 0);
    v169 = 3;
    v168 = "i8x4";
    v173 = 1;
    v171[0] = (__int64)"zext";
    v172 = 3;
    v77 = *(__int64 ***)v76;
    if ( *(_QWORD *)v76 == *(_QWORD *)a4 )
    {
      v78 = a4;
    }
    else if ( *(_BYTE *)(a4 + 16) > 0x10u )
    {
      LOWORD(v176[0]) = 257;
      v127 = sub_15FDBD0(37, a4, (__int64)v77, (__int64)&v174, 0);
      if ( v178 )
      {
        v158 = v127;
        v150 = v179;
        sub_157E9D0(v178 + 40, v127);
        v127 = v158;
        v128 = *(_QWORD *)(v158 + 24);
        v129 = *v150;
        *(_QWORD *)(v158 + 32) = v150;
        v129 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v158 + 24) = v129 | v128 & 7;
        *(_QWORD *)(v129 + 8) = v158 + 24;
        *v150 = *v150 & 7 | (v158 + 24);
      }
      v159 = v127;
      sub_164B780(v127, v171);
      v78 = v159;
      if ( v177 )
      {
        v166[0] = (__int64)v177;
        sub_1623A60((__int64)v166, (__int64)v177, 2);
        v78 = v159;
        v130 = *(_QWORD *)(v159 + 48);
        v131 = v159 + 48;
        if ( v130 )
        {
          sub_161E7C0(v159 + 48, v130);
          v78 = v159;
          v131 = v159 + 48;
        }
        v132 = (unsigned __int8 *)v166[0];
        *(_QWORD *)(v78 + 48) = v166[0];
        if ( v132 )
        {
          v160 = v78;
          sub_1623210((__int64)v166, v132, v131);
          v78 = v160;
        }
      }
    }
    else
    {
      v78 = sub_15A46C0(37, (__int64 ***)a4, v77, 0);
    }
    if ( *(_BYTE *)(v78 + 16) > 0x10u || *(_BYTE *)(v76 + 16) > 0x10u )
    {
      LOWORD(v176[0]) = 257;
      v118 = sub_15FB440(15, (__int64 *)v78, v76, (__int64)&v174, 0);
      if ( v178 )
      {
        v119 = v179;
        v155 = v118;
        sub_157E9D0(v178 + 40, v118);
        v118 = v155;
        v120 = *v119;
        v121 = *(_QWORD *)(v155 + 24);
        *(_QWORD *)(v155 + 32) = v119;
        v120 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v155 + 24) = v120 | v121 & 7;
        *(_QWORD *)(v120 + 8) = v155 + 24;
        *v119 = *v119 & 7 | (v155 + 24);
      }
      v156 = (_QWORD *)v118;
      sub_164B780(v118, (__int64 *)&v168);
      v79 = v156;
      if ( v177 )
      {
        v166[0] = (__int64)v177;
        sub_1623A60((__int64)v166, (__int64)v177, 2);
        v79 = v156;
        v122 = v156[6];
        v123 = (__int64)(v156 + 6);
        if ( v122 )
        {
          sub_161E7C0((__int64)(v156 + 6), v122);
          v79 = v156;
        }
        v124 = (unsigned __int8 *)v166[0];
        v79[6] = v166[0];
        if ( v124 )
        {
          v157 = v79;
          sub_1623210((__int64)v166, v124, v123);
          v79 = v157;
        }
      }
    }
    else
    {
      v79 = (_QWORD *)sub_15A2C20((__int64 *)v78, v76, 0, 0, *(double *)a7.m128_u64, a8, a9);
    }
    LOWORD(v176[0]) = 257;
    v80 = sub_156DA60((__int64 *)&v177, v22 >> 2, v79, (__int64 *)&v174);
    v35 = v177;
    v36 = v80;
    goto LABEL_26;
  }
  v28 = 0;
  v29 = 0;
  do
  {
    ++v29;
    v28 |= (v28 << 8) | 1;
  }
  while ( v22 != v29 );
  v30 = sub_1644900(v180, 8 * v22);
  LODWORD(v168) = v22;
  v31 = sub_159C470(v30, v28, 0);
  v173 = 1;
  v166[0] = (__int64)"i8x";
  v172 = 3;
  v166[1] = (__int64)v168;
  v167 = 2307;
  v171[0] = (__int64)"zext";
  v32 = *(__int64 ***)v31;
  if ( *(_QWORD *)v31 == *(_QWORD *)a4 )
  {
    v33 = a4;
  }
  else if ( *(_BYTE *)(a4 + 16) > 0x10u )
  {
    LOWORD(v176[0]) = 257;
    v33 = sub_15FDBD0(37, a4, (__int64)v32, (__int64)&v174, 0);
    if ( v178 )
    {
      v153 = v179;
      sub_157E9D0(v178 + 40, v33);
      v65 = *v153;
      v66 = *(_QWORD *)(v33 + 24) & 7LL;
      *(_QWORD *)(v33 + 32) = v153;
      v65 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v33 + 24) = v65 | v66;
      *(_QWORD *)(v65 + 8) = v33 + 24;
      *v153 = *v153 & 7 | (v33 + 24);
    }
    sub_164B780(v33, v171);
    if ( v177 )
    {
      v165 = v177;
      sub_1623A60((__int64)&v165, (__int64)v177, 2);
      v67 = *(_QWORD *)(v33 + 48);
      v68 = v33 + 48;
      if ( v67 )
      {
        sub_161E7C0(v33 + 48, v67);
        v68 = v33 + 48;
      }
      v69 = (unsigned __int8 *)v165;
      *(_QWORD *)(v33 + 48) = v165;
      if ( v69 )
        sub_1623210((__int64)&v165, v69, v68);
    }
  }
  else
  {
    v33 = sub_15A46C0(37, (__int64 ***)a4, v32, 0);
  }
  if ( *(_BYTE *)(v33 + 16) <= 0x10u && *(_BYTE *)(v31 + 16) <= 0x10u )
  {
    v34 = sub_15A2C20((__int64 *)v33, v31, 0, 0, *(double *)a7.m128_u64, a8, a9);
    v35 = v177;
    v36 = v34;
LABEL_26:
    if ( v35 )
      sub_161E7C0((__int64)&v177, (__int64)v35);
    goto LABEL_30;
  }
  LOWORD(v176[0]) = 257;
  v36 = sub_15FB440(15, (__int64 *)v33, v31, (__int64)&v174, 0);
  if ( v178 )
  {
    v70 = v179;
    sub_157E9D0(v178 + 40, v36);
    v71 = *(_QWORD *)(v36 + 24);
    v72 = *v70;
    *(_QWORD *)(v36 + 32) = v70;
    v72 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v36 + 24) = v72 | v71 & 7;
    *(_QWORD *)(v72 + 8) = v36 + 24;
    *v70 = *v70 & 7 | (v36 + 24);
  }
  sub_164B780(v36, v166);
  if ( v177 )
  {
    v165 = v177;
    sub_1623A60((__int64)&v165, (__int64)v177, 2);
    v73 = *(_QWORD *)(v36 + 48);
    if ( v73 )
      sub_161E7C0(v36 + 48, v73);
    v74 = (unsigned __int8 *)v165;
    *(_QWORD *)(v36 + 48) = v165;
    if ( v74 )
      sub_1623210((__int64)&v165, v74, v36 + 48);
    v35 = v177;
    goto LABEL_26;
  }
LABEL_30:
  v151 = *(__int64 **)v36;
  v37 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v37 = **(_QWORD **)(v37 + 16);
  v38 = *(_DWORD *)(v37 + 8) >> 8;
  v39 = sub_1646BA0(v151, v38);
  v148 = *(_QWORD *)a3;
  if ( *(_BYTE *)(a3 + 16) != 13 )
  {
    v138 = v39;
    v177 = "dst";
    LOWORD(v179) = 259;
    v43 = sub_1648A60(56, 1u);
    v46 = v138;
    v47 = v43;
    if ( v43 )
    {
      v139 = v43;
      sub_15FD590((__int64)v43, (__int64)a2, v46, (__int64)&v177, (__int64)a1);
      v47 = v139;
    }
    sub_1C7F460(a1, (__int64)v151, v47, v145, v36, a6, a7, a8, a9, a10, v44, v45, a13, a14, (__int64)a15, a16);
    LOWORD(v179) = 259;
    v177 = "rem";
    v48 = sub_15A0680(v148, v22, 0);
    v152 = sub_15FB440(20, (__int64 *)a3, v48, (__int64)&v177, (__int64)a1);
    v177 = "offset";
    LOWORD(v179) = 259;
    v49 = sub_15A0680(v148, v22, 0);
    v50 = sub_15FB440(15, (__int64 *)v145, v49, (__int64)&v177, (__int64)a1);
    v174 = (unsigned __int8 *)v176;
    v176[0] = v50;
    v175 = 0x100000001LL;
    v177 = "rem.gep";
    LOWORD(v179) = 259;
    v51 = sub_1643330(a15);
    if ( !v51 )
    {
      v117 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v117 = **(_QWORD **)(v117 + 16);
      v51 = *(_QWORD *)(v117 + 24);
    }
    v52 = sub_1648A60(72, 2u);
    v55 = (__int64)v52;
    if ( v52 )
    {
      v56 = v52 - 6;
      v57 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v57 = **(_QWORD **)(v57 + 16);
      v146 = (__int64)v56;
      v149 = *(_DWORD *)(v57 + 8) >> 8;
      v58 = (__int64 *)sub_15F9F50(v51, (__int64)v176, 1);
      v59 = sub_1646BA0(v58, v149);
      v60 = v146;
      v61 = (__int64 *)v59;
      v62 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16
        || (v62 = *(_QWORD *)v176[0], *(_BYTE *)(*(_QWORD *)v176[0] + 8LL) == 16) )
      {
        v64 = sub_16463B0(v61, *(_QWORD *)(v62 + 32));
        v60 = v146;
        v61 = v64;
      }
      sub_15F1EA0(v55, (__int64)v61, 32, v60, 2, (__int64)a1);
      *(_QWORD *)(v55 + 56) = v51;
      *(_QWORD *)(v55 + 64) = sub_15F9F50(v51, (__int64)v176, 1);
      sub_15F9CE0(v55, (__int64)a2, v176, 1, (__int64)&v177);
    }
    sub_1C7F460(
      a1,
      *(_QWORD *)(v55 + 64),
      (_BYTE *)v55,
      v152,
      a4,
      a6,
      a7,
      a8,
      a9,
      a10,
      v53,
      v54,
      a13,
      a14,
      (__int64)a15,
      a16);
    v63 = v174;
    if ( v174 == (unsigned __int8 *)v176 )
      return;
LABEL_47:
    _libc_free((unsigned __int64)v63);
    return;
  }
  v40 = *(unsigned __int64 **)(a3 + 24);
  if ( *(_DWORD *)(a3 + 32) <= 0x40u )
  {
    v41 = *(_QWORD *)(a3 + 24);
    if ( !v40 )
      return;
  }
  else
  {
    v41 = *v40;
    if ( !*v40 )
      return;
  }
  v140 = v39;
  v177 = "dst";
  LOWORD(v179) = 259;
  v81 = sub_1648A60(56, 1u);
  v84 = v81;
  if ( v81 )
  {
    v85 = v140;
    v141 = v81;
    sub_15FD590((__int64)v81, (__int64)a2, v85, (__int64)&v177, (__int64)a1);
    v84 = v141;
  }
  v142 = a6;
  sub_1C7F460(a1, (__int64)v151, v84, v145, v36, a6, a7, a8, a9, a10, v82, v83, a13, a14, (__int64)a15, a16);
  v135 = v41 % v22;
  if ( v135 )
  {
    v174 = (unsigned __int8 *)v176;
    v175 = 0x100000000LL;
    v86 = *(_QWORD **)(v145 + 24);
    if ( *(_DWORD *)(v145 + 32) > 0x40u )
      v86 = (_QWORD *)*v86;
    v89 = sub_15A0680(v148, (_QWORD)v86 * v22, 0);
    v90 = (unsigned int)v175;
    if ( (unsigned int)v175 >= HIDWORD(v175) )
    {
      sub_16CD150((__int64)&v174, v176, 0, 8, v87, v88);
      v90 = (unsigned int)v175;
    }
    *(_QWORD *)&v174[8 * v90] = v89;
    v91 = v175;
    v154 = (__int64 *)v174;
    v177 = "rem.gep";
    LODWORD(v175) = v175 + 1;
    LOWORD(v179) = 259;
    v147 = (unsigned int)v175;
    v144 = sub_1643330(a15);
    if ( !v144 )
    {
      v125 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v125 = **(_QWORD **)(v125 + 16);
      v144 = *(_QWORD *)(v125 + 24);
    }
    v136 = v91 + 2;
    v92 = sub_1648A60(72, v91 + 2);
    v93 = (__int64)v92;
    if ( v92 )
    {
      v94 = v136;
      v137 = (__int64)&v92[-3 * v136];
      v95 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v95 = **(_QWORD **)(v95 + 16);
      v133 = v94;
      v134 = *(_DWORD *)(v95 + 8) >> 8;
      v96 = (__int64 *)sub_15F9F50(v144, (__int64)v154, v147);
      v97 = (__int64 *)sub_1646BA0(v96, v134);
      v98 = v133;
      v99 = v97;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      {
        v126 = sub_16463B0(v97, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
        v98 = v133;
        v99 = v126;
      }
      else
      {
        v100 = v154;
        v101 = &v154[v147];
        if ( v154 != v101 )
        {
          while ( 1 )
          {
            v102 = *(_QWORD *)*v100;
            if ( *(_BYTE *)(v102 + 8) == 16 )
              break;
            if ( v101 == ++v100 )
              goto LABEL_89;
          }
          v103 = sub_16463B0(v99, *(_QWORD *)(v102 + 32));
          v98 = v133;
          v99 = v103;
        }
      }
LABEL_89:
      sub_15F1EA0(v93, (__int64)v99, 32, v137, v98, (__int64)a1);
      *(_QWORD *)(v93 + 56) = v144;
      *(_QWORD *)(v93 + 64) = sub_15F9F50(v144, (__int64)v154, v147);
      sub_15F9CE0(v93, (__int64)a2, v154, v147, (__int64)&v177);
    }
    if ( v21 )
    {
      v104 = (__int64 *)sub_1643330(a15);
      v105 = sub_16463B0(v104, v135);
      v163 = sub_1646BA0(v105, v38);
      v177 = "rem.dst";
      LOWORD(v179) = 259;
      v106 = sub_1648A60(56, 1u);
      v107 = (__int64)v106;
      if ( v106 )
        sub_15FD590((__int64)v106, v93, v163, (__int64)&v177, (__int64)a1);
      v108 = *((_BYTE *)v105 + 8) == 11 ? sub_15A0680((__int64)v105, 0, 0) : sub_1598F00((__int64 **)v105);
      v109 = sub_1648A60(64, 2u);
      if ( v109 )
        sub_15F9630((__int64)v109, v108, v107, v142, v22, (__int64)a1);
    }
    else
    {
      v113 = sub_15A0680(v148, v135, 0);
      v114 = sub_1643330(a15);
      sub_1C7F460(a1, v114, (_BYTE *)v93, v113, a4, v142, a7, a8, a9, a10, v115, v116, a13, a14, (__int64)a15, a16);
    }
    v63 = v174;
    if ( v174 != (unsigned __int8 *)v176 )
      goto LABEL_47;
  }
}
