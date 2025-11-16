// Function: sub_1725880
// Address: 0x1725880
//
__int64 __fastcall sub_1725880(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // r12
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  char v14; // bl
  char v15; // al
  unsigned __int8 *v16; // rax
  __int64 v17; // r15
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // rcx
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rbx
  __int64 **v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rsi
  bool v40; // al
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 *v43; // r11
  bool v44; // al
  __int64 v45; // rax
  __int64 v46; // rax
  char v47; // al
  __int64 *v48; // r11
  char v49; // bl
  int v50; // eax
  char v51; // al
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 *v54; // r11
  __int64 v55; // rdi
  bool v56; // al
  char v57; // al
  int v58; // eax
  bool v59; // al
  __int64 v60; // rdi
  __int64 v61; // r12
  __int64 *v62; // rax
  __int64 v63; // rsi
  char v64; // al
  __int64 *v65; // r11
  _QWORD *v66; // rdi
  unsigned int v67; // ebx
  int v68; // eax
  int v69; // eax
  int v70; // eax
  char v71; // al
  __int64 v72; // rcx
  __int64 *v73; // r11
  char v74; // al
  char v75; // al
  __int64 *v76; // rsi
  bool v77; // al
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 *v80; // r11
  __int64 v81; // rax
  char v82; // al
  __int64 *v83; // r11
  char v84; // al
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 *v87; // r11
  __int64 **v88; // rbx
  int v89; // eax
  __int64 v90; // rdx
  __int64 v91; // rcx
  bool v92; // al
  double v93; // xmm4_8
  double v94; // xmm5_8
  char v95; // al
  __int64 v96; // rdx
  __int64 *v97; // r12
  __int64 *v98; // rax
  bool v99; // al
  __int64 v100; // rdi
  __int64 *v101; // rax
  int v102; // eax
  __int64 **v103; // rdx
  __int64 *v104; // rax
  __int64 v105; // rdi
  char v106; // al
  __int64 v107; // rax
  __int64 *v108; // rax
  __int64 v109; // rax
  int v110; // eax
  __int64 v111; // rdx
  __int64 v112; // rcx
  char v113; // al
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 *v116; // r11
  bool v117; // al
  char v118; // al
  bool v119; // al
  __int64 v120; // rax
  __int64 v121; // r11
  int v122; // eax
  __int64 v123; // rbx
  __int64 v124; // rax
  _QWORD *v125; // rax
  __int64 v126; // rax
  unsigned int v127; // edx
  int v128; // eax
  __int64 *v129; // rbx
  __int64 v130; // rdx
  char v131; // al
  __int64 v132; // rdx
  char v133; // al
  __int64 v134; // rax
  double v135; // xmm4_8
  double v136; // xmm5_8
  char v137; // al
  __int64 v138; // rax
  __int64 v139; // r12
  __int64 v140; // rdx
  unsigned __int8 *v141; // rax
  char v142; // al
  __int16 v143; // dx
  __int64 *v144; // rbx
  __int64 v145; // r8
  char v146; // al
  char v147; // al
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 *v150; // r11
  __int64 *v151; // rbx
  int v152; // eax
  char v153; // al
  __int64 v154; // rcx
  __int64 v155; // rdi
  unsigned __int8 *v156; // rdx
  __int64 v157; // rbx
  __int64 *v158; // rax
  char v159; // al
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 *v162; // rax
  __int64 v163; // rdi
  __int64 **v164; // rcx
  unsigned __int8 *v165; // rax
  bool v166; // al
  char v167; // al
  __int64 v168; // rdx
  __int64 v169; // rdi
  __int64 v170; // r12
  __int64 v171; // rax
  unsigned __int8 *v172; // rax
  __int64 **v173; // rax
  __int64 *v174; // rax
  __int64 v175; // r12
  __int64 v176; // rdx
  unsigned __int8 *v177; // rax
  __int64 v178; // r15
  __int64 v179; // rax
  unsigned __int8 *v180; // rax
  __int64 v181; // r13
  __int64 v182; // r15
  bool v183; // bl
  bool v184; // al
  unsigned __int8 *v185; // rax
  char v186; // al
  __int64 *v187; // rax
  __int64 v188; // rbx
  __int64 v189; // r12
  const char *v190; // rax
  __int64 v191; // rdx
  __int64 v192; // rax
  __int64 v193; // rdx
  __int64 v194; // rdx
  unsigned __int64 v195; // rax
  int v196; // edx
  __int64 v197; // r14
  __int64 v198; // rdx
  __int64 v199; // rax
  __int64 v200; // rax
  unsigned int v201; // r8d
  int v202; // eax
  bool v203; // al
  __int64 *v204; // [rsp+0h] [rbp-100h]
  __int64 *v205; // [rsp+8h] [rbp-F8h]
  int v206; // [rsp+8h] [rbp-F8h]
  __int64 *v207; // [rsp+10h] [rbp-F0h]
  __int64 *v208; // [rsp+10h] [rbp-F0h]
  __int64 *v209; // [rsp+10h] [rbp-F0h]
  __int64 *v210; // [rsp+18h] [rbp-E8h]
  int v211; // [rsp+18h] [rbp-E8h]
  __int64 *v212; // [rsp+18h] [rbp-E8h]
  __int64 *v213; // [rsp+18h] [rbp-E8h]
  __int64 *v214; // [rsp+18h] [rbp-E8h]
  int v215; // [rsp+18h] [rbp-E8h]
  bool v216; // [rsp+18h] [rbp-E8h]
  __int64 *v217; // [rsp+18h] [rbp-E8h]
  __int64 *v218; // [rsp+18h] [rbp-E8h]
  __int64 *v219; // [rsp+18h] [rbp-E8h]
  __int64 *v220; // [rsp+18h] [rbp-E8h]
  __int64 *v221; // [rsp+18h] [rbp-E8h]
  __int64 v222; // [rsp+18h] [rbp-E8h]
  __int64 *v223; // [rsp+20h] [rbp-E0h]
  __int64 *v224; // [rsp+20h] [rbp-E0h]
  __int64 *v225; // [rsp+20h] [rbp-E0h]
  __int64 v226; // [rsp+20h] [rbp-E0h]
  __int64 *v227; // [rsp+20h] [rbp-E0h]
  __int64 *v228; // [rsp+20h] [rbp-E0h]
  __int64 *v229; // [rsp+20h] [rbp-E0h]
  __int64 *v230; // [rsp+20h] [rbp-E0h]
  __int64 *v231; // [rsp+20h] [rbp-E0h]
  __int64 *v232; // [rsp+20h] [rbp-E0h]
  __int64 v233; // [rsp+20h] [rbp-E0h]
  unsigned int v234; // [rsp+20h] [rbp-E0h]
  __int64 *v235; // [rsp+20h] [rbp-E0h]
  __int64 v236; // [rsp+20h] [rbp-E0h]
  __int64 *v237; // [rsp+20h] [rbp-E0h]
  int v238; // [rsp+20h] [rbp-E0h]
  __int64 v239; // [rsp+28h] [rbp-D8h]
  __int64 v240; // [rsp+30h] [rbp-D0h] BYREF
  __int64 *v241; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v242; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v243; // [rsp+48h] [rbp-B8h] BYREF
  __int64 *v244; // [rsp+50h] [rbp-B0h] BYREF
  _DWORD v245[6]; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v246; // [rsp+70h] [rbp-90h] BYREF
  __int64 v247; // [rsp+78h] [rbp-88h]
  __int64 *v248; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v249; // [rsp+88h] [rbp-78h]
  __int16 v250; // [rsp+90h] [rbp-70h]
  __m128 v251; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v252; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v253; // [rsp+C0h] [rbp-40h]

  v11 = (__int64 *)a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v253 = a2;
  v13 = _mm_loadu_si128(a1 + 168);
  v251 = v12;
  v252 = v13;
  v14 = sub_15F2370(a2);
  v15 = sub_15F2380(a2);
  v16 = sub_13DF290(*(_QWORD **)(a2 - 48), *(_QWORD *)(a2 - 24), v15, v14, (__int64 *)&v251);
  if ( v16 )
  {
    if ( *(_QWORD *)(a2 + 8) )
    {
      v17 = (__int64)v16;
      sub_17205C0(a1->m128i_i64[0], a2);
      if ( a2 == v17 )
        v17 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v17, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v18, v19, a9, a10);
      return (__int64)v11;
    }
    return 0;
  }
  v21 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)a2,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
  if ( v21 )
    return v21;
  v22 = sub_1708300(a1, (unsigned __int8 *)a2, (__m128i)v12, v13, a5);
  if ( !v22 )
  {
    v26 = *(_QWORD *)(a2 - 24);
    v27 = *(_QWORD *)(a2 - 48);
    v239 = v27;
    v28 = sub_1705480(
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            (__int64)a1,
            v26,
            0,
            v23);
    if ( v28 )
    {
      v29 = v27;
      v252.m128i_i16[0] = 257;
      v32 = sub_15FB440(11, (__int64 *)v27, v28, (__int64)&v251, 0);
      if ( (unsigned __int8)(*(_BYTE *)(v26 + 16) - 35) > 0x11u )
      {
        if ( !(unsigned __int8)sub_1596730(v26, v29, v30, v31) )
          return v32;
      }
      else if ( !sub_15F2380(v26) )
      {
        return v32;
      }
      v36 = (__int64)v11;
      v11 = (__int64 *)v32;
      if ( sub_15F2380(v36) )
        sub_15F2330(v32, 1);
      return (__int64)v11;
    }
    v33 = *(__int64 ***)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v33 = (__int64 **)*v33[2];
    if ( sub_1642F90((__int64)v33, 1) )
    {
      v252.m128i_i16[0] = 257;
      return sub_15FB440(28, (__int64 *)v27, v26, (__int64)&v251, 0);
    }
    if ( (unsigned __int8)sub_17198D0((_BYTE *)v27, 1, v34, v35) )
    {
      v252.m128i_i16[0] = 257;
      return sub_15FB630((__int64 *)v26, (__int64)&v251, 0);
    }
    v39 = v27;
    v248 = &v240;
    v40 = sub_171DA10(&v248, v27, v37, v38);
    v43 = (__int64 *)&v248;
    if ( v40 )
    {
      v39 = v26;
      v251.m128_u64[0] = (unsigned __int64)&v241;
      v44 = sub_171DA10(&v251, v26, v41, v42);
      v43 = (__int64 *)&v248;
      if ( v44 )
      {
        v252.m128i_i16[0] = 257;
        return sub_15FB440(13, v241, v240, (__int64)&v251, 0);
      }
    }
    v45 = *(_QWORD *)(v27 + 8);
    if ( !v45 || *(_QWORD *)(v45 + 8) )
    {
LABEL_29:
      v251.m128_u64[0] = (unsigned __int64)&v240;
      v46 = *(_QWORD *)(v26 + 8);
      if ( v46 )
      {
        if ( !*(_QWORD *)(v46 + 8) )
        {
          v39 = v26;
          v230 = v43;
          v99 = sub_171CD50(&v251, v26, v41, v42);
          v43 = v230;
          if ( v99 )
          {
            v100 = a1->m128i_i64[1];
            v252.m128i_i16[0] = 257;
            v250 = 257;
            v101 = (__int64 *)sub_171CA90(
                                v100,
                                v240,
                                v230,
                                *(double *)v12.m128_u64,
                                *(double *)v13.m128i_i64,
                                *(double *)a5.m128i_i64);
            return sub_15FB440(11, v101, v27, (__int64)&v251, 0);
          }
        }
      }
      if ( *(_BYTE *)(v27 + 16) > 0x10u )
        goto LABEL_52;
      v223 = v43;
      v47 = sub_1719130((_BYTE *)v27, v39, v41, v42);
      v48 = v223;
      v49 = v47;
      v50 = *(unsigned __int8 *)(v26 + 16);
      if ( (unsigned __int8)v50 > 0x17u )
      {
        v102 = v50 - 24;
      }
      else
      {
        if ( (_BYTE)v50 != 5 )
          goto LABEL_34;
        v102 = *(unsigned __int16 *)(v26 + 18);
      }
      if ( v102 == 37 )
      {
        v103 = (*(_BYTE *)(v26 + 23) & 0x40) != 0
             ? *(__int64 ***)(v26 - 8)
             : (__int64 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
        v104 = *v103;
        if ( *v103 )
        {
          v248 = *v103;
          v105 = *v104;
          if ( *(_BYTE *)(*v104 + 8) == 16 )
            v105 = **(_QWORD **)(v105 + 16);
          if ( sub_1642F90(v105, 1) )
          {
            v252.m128i_i16[0] = 257;
            if ( v49 )
              return sub_15FDED0(v248, *v11, (__int64)&v251, 0);
            v123 = v239;
            v124 = sub_15A0680(*(_QWORD *)v239, 1, 0);
            v125 = (_QWORD *)sub_15A2B60(
                               (__int64 *)v239,
                               v124,
                               0,
                               0,
                               *(double *)v12.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)a5.m128i_i64);
            return sub_14EDD70((__int64)v248, v125, v123, (__int64)&v251, 0, 0);
          }
          v48 = v223;
        }
      }
LABEL_34:
      v251.m128_u64[0] = (unsigned __int64)v48;
      v224 = v48;
      v51 = sub_171E9E0(&v251, v26);
      v54 = v224;
      if ( v51 )
      {
        v55 = *v248;
        if ( *(_BYTE *)(*v248 + 8) == 16 )
          v55 = **(_QWORD **)(v55 + 16);
        v56 = sub_1642F90(v55, 1);
        v54 = v224;
        if ( v56 )
        {
          v252.m128i_i16[0] = 257;
          if ( v49 )
            return sub_15FDE70(v248, *v11, (__int64)&v251, 0);
          v123 = v239;
          v125 = (_QWORD *)sub_1718F30(
                             (__int64 *)v239,
                             *(double *)v12.m128_u64,
                             *(double *)v13.m128i_i64,
                             *(double *)a5.m128i_i64);
          return sub_14EDD70((__int64)v248, v125, v123, (__int64)&v251, 0, 0);
        }
      }
      v251.m128_u64[0] = (unsigned __int64)v54;
      v229 = v54;
      v92 = sub_171DA10(&v251, v26, v52, v53);
      v43 = v229;
      if ( v92 )
      {
        v252.m128i_i16[0] = 257;
        v109 = sub_1718F30((__int64 *)v239, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
        return sub_15FB440(11, v248, v109, (__int64)&v251, 0);
      }
      if ( *(_BYTE *)(v26 + 16) == 79 )
      {
        v21 = sub_1707470(
                (__int64)a1,
                v11,
                v26,
                *(double *)v12.m128_u64,
                *(double *)v13.m128i_i64,
                *(double *)a5.m128i_i64);
        if ( v21 )
          return v21;
        v43 = v229;
      }
      if ( *(_BYTE *)(v26 + 16) == 77 )
      {
        v235 = v43;
        v21 = sub_17127D0(
                a1->m128i_i64,
                (__int64)v11,
                v26,
                v12,
                *(double *)v13.m128i_i64,
                *(double *)a5.m128i_i64,
                a6,
                v93,
                v94,
                a9,
                a10);
        if ( v21 )
          return v21;
        v43 = v235;
      }
      v95 = *(_BYTE *)(v26 + 16);
      if ( v95 == 35 )
      {
        if ( *(_QWORD *)(v26 - 48) )
        {
          v248 = *(__int64 **)(v26 - 48);
          v63 = *(_QWORD *)(v26 - 24);
          if ( *(_BYTE *)(v63 + 16) <= 0x10u )
            goto LABEL_85;
        }
      }
      else if ( v95 == 5 && *(_WORD *)(v26 + 18) == 11 )
      {
        v96 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
        if ( *(_QWORD *)(v26 - 24 * v96) )
        {
          v248 = *(__int64 **)(v26 - 24 * v96);
          v63 = *(_QWORD *)(v26 + 24 * (1 - v96));
          if ( v63 )
          {
LABEL_85:
            v97 = v248;
            v252.m128i_i16[0] = 257;
            v98 = (__int64 *)sub_15A2B60(
                               (__int64 *)v239,
                               v63,
                               0,
                               0,
                               *(double *)v12.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)a5.m128i_i64);
            return sub_15FB440(13, v98, (__int64)v97, (__int64)&v251, 0);
          }
        }
      }
LABEL_52:
      v225 = v43;
      v251.m128_u64[0] = (unsigned __int64)&v242;
      v64 = sub_13D2630(&v251, (_BYTE *)v239);
      v65 = v225;
      if ( !v64 )
        goto LABEL_57;
      v66 = (_QWORD *)v242;
      v67 = *(_DWORD *)(v242 + 8);
      if ( v67 <= 0x40 )
      {
        if ( *(_QWORD *)v242 )
          goto LABEL_117;
      }
      else
      {
        v210 = v225;
        v226 = v242;
        v68 = sub_16A57B0(v242);
        v66 = (_QWORD *)v226;
        v65 = v210;
        if ( v67 != v68 )
          goto LABEL_55;
      }
      v207 = v65;
      v110 = sub_16431D0(*v11);
      v251.m128_u64[0] = (unsigned __int64)v245;
      v215 = v110;
      v251.m128_u64[1] = (unsigned __int64)&v246;
      v113 = sub_171EB90(&v251, v26, v111, v112);
      v116 = v207;
      if ( v113 )
      {
        v117 = sub_13A38F0(v246, (_QWORD *)(unsigned int)(v215 - 1));
        v116 = v207;
        if ( v117 )
        {
          v193 = *(_QWORD *)(sub_13CF970(v26) + 24);
          v252.m128i_i16[0] = 257;
          return sub_15FB440(25, *(__int64 **)v245, v193, (__int64)&v251, 0);
        }
      }
      v208 = v116;
      v251.m128_u64[1] = (unsigned __int64)&v246;
      v251.m128_u64[0] = (unsigned __int64)v245;
      v118 = sub_171FED0(&v251, v26, v114, v115);
      v65 = v208;
      if ( v118 )
      {
        v119 = sub_13A38F0(v246, (_QWORD *)(unsigned int)(v215 - 1));
        v65 = v208;
        if ( v119 )
        {
          v194 = *(_QWORD *)(sub_13CF970(v26) + 24);
          v252.m128i_i16[0] = 257;
          return sub_15FB440(24, *(__int64 **)v245, v194, (__int64)&v251, 0);
        }
      }
      v120 = *(_QWORD *)(v26 + 8);
      if ( v120 )
      {
        if ( !*(_QWORD *)(v120 + 8) )
        {
          v237 = v65;
          v195 = sub_14B2890(v26, v65, (__int64 *)&v251, 0, 0);
          v65 = v237;
          *(_QWORD *)&v245[3] = v195;
          v245[5] = v196;
          if ( (unsigned int)(v195 - 7) <= 1 )
          {
            v197 = *(_QWORD *)(v26 - 48);
            sub_1593B40((_QWORD *)(v26 - 48), *(_QWORD *)(v26 - 24));
            sub_1593B40((_QWORD *)(v26 - 24), v197);
            v85 = v26;
            return sub_170E100(
                     a1->m128i_i64,
                     (__int64)v11,
                     v85,
                     v12,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64,
                     a6,
                     v135,
                     v136,
                     a9,
                     a10);
          }
        }
      }
      v66 = (_QWORD *)v242;
      v67 = *(_DWORD *)(v242 + 8);
      if ( v67 > 0x40 )
      {
LABEL_55:
        v227 = v65;
        v69 = sub_16A58F0((__int64)v66);
        v65 = v227;
        v211 = v69;
        if ( !v69 )
          goto LABEL_57;
        v70 = sub_16A57B0((__int64)v66);
        v65 = v227;
        if ( v70 + v211 != v67 )
          goto LABEL_57;
LABEL_119:
        v231 = v65;
        sub_14C2530(
          (__int64)&v251,
          (__int64 *)v26,
          a1[166].m128i_i64[1],
          0,
          a1[165].m128i_i64[0],
          (__int64)v11,
          a1[166].m128i_i64[0],
          0);
        sub_13A38D0((__int64)&v246, v242);
        v121 = (__int64)v231;
        if ( (unsigned int)v247 > 0x40 )
        {
          sub_16A89F0(&v246, (__int64 *)&v251);
          v121 = (__int64)v231;
        }
        else
        {
          v246 |= v251.m128_u64[0];
        }
        v122 = v247;
        v232 = (__int64 *)v121;
        LODWORD(v247) = 0;
        LODWORD(v249) = v122;
        v248 = (__int64 *)v246;
        v216 = sub_1454FB0(v121);
        sub_135E100(v232);
        sub_135E100(&v246);
        if ( v216 )
        {
          v250 = 257;
          v11 = (__int64 *)sub_15FB440(28, (__int64 *)v26, v239, (__int64)v232, 0);
          sub_135E100(v252.m128i_i64);
          sub_135E100((__int64 *)&v251);
          return (__int64)v11;
        }
        sub_135E100(v252.m128i_i64);
        sub_135E100((__int64 *)&v251);
        v65 = v232;
LABEL_57:
        v251.m128_u64[1] = (unsigned __int64)v65;
        v228 = v65;
        v251.m128_u64[0] = v239;
        v71 = sub_13D60C0((__int64)&v251, v26);
        v73 = v228;
        if ( !v71 )
        {
          v74 = *(_BYTE *)(v239 + 16);
          if ( v74 == 37 )
          {
            v107 = *(_QWORD *)(v239 - 48);
            if ( !v107 || v26 != v107 || (v108 = *(__int64 **)(v239 - 24)) == 0 )
            {
LABEL_61:
              v75 = *(_BYTE *)(v26 + 16);
              if ( v75 == 52 )
              {
                v129 = *(__int64 **)(v26 - 48);
                if ( !v129 )
                  goto LABEL_64;
                v130 = *(_QWORD *)(v26 - 24);
                if ( !v130 )
                  goto LABEL_64;
              }
              else
              {
                if ( v75 != 5 )
                  goto LABEL_64;
                if ( *(_WORD *)(v26 + 18) != 28 )
                  goto LABEL_64;
                v132 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                v129 = *(__int64 **)(v26 - 24 * v132);
                if ( !v129 )
                  goto LABEL_64;
                v130 = *(_QWORD *)(v26 + 24 * (1 - v132));
                if ( !v130 )
                  goto LABEL_64;
              }
              v251.m128_u64[1] = v130;
              v217 = v228;
              v236 = v130;
              v251.m128_u64[0] = (unsigned __int64)v129;
              v131 = sub_1720000(&v251, v239);
              v73 = v217;
              if ( v131 )
              {
                v252.m128i_i16[0] = 257;
                return sub_15FB440(26, v129, v236, (__int64)&v251, 0);
              }
LABEL_64:
              v76 = (__int64 *)v239;
              v251.m128_u64[1] = v26;
              v212 = v73;
              v251.m128_u64[0] = (unsigned __int64)v245;
              v77 = sub_17200C0((__int64)&v251, v239);
              v80 = v212;
              if ( v77 )
              {
                v139 = a1->m128i_i64[1];
                v252.m128i_i16[0] = 257;
                v246 = (__int64)sub_1649960(v26);
                v247 = v140;
                v248 = &v246;
                v250 = 773;
                v249 = (__int64 *)".not";
                v141 = sub_171CA90(
                         v139,
                         v26,
                         v212,
                         *(double *)v12.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64);
                return sub_15FB440(26, *(__int64 **)v245, (__int64)v141, (__int64)&v251, 0);
              }
              v81 = *(_QWORD *)(v26 + 8);
              if ( !v81 || *(_QWORD *)(v81 + 8) )
              {
LABEL_67:
                v213 = v80;
                v248 = &v243;
                v82 = sub_17201D0((_QWORD **)v80, v239);
                v83 = v213;
                if ( !v82
                  || (v251.m128_u64[0] = (unsigned __int64)&v244, v137 = sub_17201D0(&v251, v26), v83 = v213, !v137)
                  || (v138 = sub_171D1F0(
                               (__int64)a1,
                               v243,
                               (__int64)v244,
                               (__int64 **)*v11,
                               *(double *)v12.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)a5.m128i_i64),
                      v83 = v213,
                      (v85 = v138) == 0) )
                {
                  v214 = v83;
                  v248 = &v243;
                  v84 = sub_1720250((_QWORD **)v83, v239);
                  v87 = v214;
                  if ( !v84
                    || (v251.m128_u64[0] = (unsigned __int64)&v244, v133 = sub_1720250(&v251, v26), v87 = v214, !v133)
                    || (v134 = sub_171D1F0(
                                 (__int64)a1,
                                 v243,
                                 (__int64)v244,
                                 (__int64 **)*v11,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64),
                        v87 = v214,
                        (v85 = v134) == 0) )
                  {
                    v88 = (__int64 **)*v11;
                    v248 = (__int64 *)v245;
                    v249 = &v246;
                    if ( (unsigned __int8)sub_171FED0((_QWORD **)v87, v26, v85, v86) )
                    {
                      if ( sub_1648CD0(v26, 2) )
                      {
                        v89 = sub_16431D0((__int64)v88);
                        if ( sub_13A38F0(v246, (_QWORD *)(unsigned int)(v89 - 1)) )
                        {
                          v251.m128_u64[1] = v26;
                          v251.m128_u64[0] = *(_QWORD *)v245;
                          if ( (unsigned __int8)sub_1720310(&v251, v239) )
                          {
                            v178 = a1->m128i_i64[1];
                            v252.m128i_i16[0] = 257;
                            v179 = sub_15A06D0(v88, v239, v90, v91);
                            v180 = sub_17203D0(v178, 40, *(__int64 *)v245, v179, (__int64 *)&v251);
                            v181 = a1->m128i_i64[1];
                            v182 = (__int64)v180;
                            v183 = sub_15F2380((__int64)v11);
                            v184 = sub_15F2370((__int64)v11);
                            v252.m128i_i16[0] = 257;
                            v185 = sub_171CBD0(
                                     v181,
                                     *(__int64 *)v245,
                                     (__int64 *)&v251,
                                     v184,
                                     v183,
                                     *(double *)v12.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)a5.m128i_i64);
                            v252.m128i_i16[0] = 257;
                            return sub_14EDD70(v182, v185, *(__int64 *)v245, (__int64)&v251, 0, 0);
                          }
                        }
                      }
                    }
                    if ( sub_15F2380((__int64)v11)
                      || (unsigned int)sub_14C2E40(
                                         v239,
                                         (__int64 *)v26,
                                         a1[166].m128i_i64[1],
                                         a1[165].m128i_i64[0],
                                         (__int64)v11,
                                         a1[166].m128i_i64[0]) != 2 )
                    {
                      if ( sub_15F2370((__int64)v11)
                        || (unsigned int)sub_14C2B30(
                                           (__int64 *)v239,
                                           (__int64 *)v26,
                                           a1[166].m128i_i64[1],
                                           a1[165].m128i_i64[0],
                                           (__int64)v11,
                                           a1[166].m128i_i64[0]) != 2 )
                      {
                        return 0;
                      }
                    }
                    else
                    {
                      sub_15F2330((__int64)v11, 1);
                      if ( sub_15F2370((__int64)v11)
                        || (unsigned int)sub_14C2B30(
                                           (__int64 *)v239,
                                           (__int64 *)v26,
                                           a1[166].m128i_i64[1],
                                           a1[165].m128i_i64[0],
                                           (__int64)v11,
                                           a1[166].m128i_i64[0]) != 2 )
                      {
                        return (__int64)v11;
                      }
                    }
                    sub_15F2310((__int64)v11, 1);
                    return (__int64)v11;
                  }
                }
                return sub_170E100(
                         a1->m128i_i64,
                         (__int64)v11,
                         v85,
                         v12,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64,
                         a6,
                         v135,
                         v136,
                         a9,
                         a10);
              }
              v244 = 0;
              v142 = *(_BYTE *)(v26 + 16);
              switch ( v142 )
              {
                case 37:
                  v78 = *(_QWORD *)(v26 - 48);
                  if ( !v78 )
                    goto LABEL_165;
                  v244 = *(__int64 **)(v26 - 48);
                  v188 = *(_QWORD *)(v26 - 24);
                  if ( !v188 )
                    goto LABEL_163;
                  break;
                case 5:
                  v143 = *(_WORD *)(v26 + 18);
                  if ( v143 != 13 )
                  {
                    if ( v143 != 26 )
                    {
                      if ( v143 != 18 )
                        goto LABEL_165;
                      v79 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                      v144 = *(__int64 **)(v26 - 24 * v79);
                      if ( !v144 )
                        goto LABEL_165;
                      v78 = 24 * (1 - v79);
                      v145 = *(_QWORD *)(v26 + v78);
                      if ( !v145 )
                        goto LABEL_165;
                      goto LABEL_162;
                    }
                    v79 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                    v78 = v26 - 24 * v79;
                    v76 = *(__int64 **)v78;
                    v173 = (__int64 **)(v26 + 24 * (1 - v79));
                    if ( *(_QWORD *)v78 )
                    {
                      v244 = *(__int64 **)v78;
                      v174 = *v173;
                      if ( (__int64 *)v239 == v174 )
                        goto LABEL_186;
                    }
                    else
                    {
                      v174 = *v173;
                    }
                    if ( !v174 )
                      goto LABEL_163;
                    v244 = v174;
                    if ( !*(_QWORD *)v78 || v239 != *(_QWORD *)v78 )
                      goto LABEL_163;
                    goto LABEL_187;
                  }
                  v78 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                  v79 = *(_QWORD *)(v26 - 24 * v78);
                  if ( !v79 )
                    goto LABEL_165;
                  v244 = *(__int64 **)(v26 - 24 * v78);
                  v188 = *(_QWORD *)(v26 + 24 * (1 - v78));
                  if ( !v188 )
                    goto LABEL_163;
                  break;
                case 50:
                  if ( *(_QWORD *)(v26 - 48) )
                  {
                    v244 = *(__int64 **)(v26 - 48);
                    v187 = *(__int64 **)(v26 - 24);
                    if ( !v187 )
                      goto LABEL_163;
                    if ( (__int64 *)v239 == v187 )
                      goto LABEL_187;
                  }
                  else
                  {
                    v187 = *(__int64 **)(v26 - 24);
                  }
                  if ( !v187 )
                    goto LABEL_163;
                  v244 = v187;
                  v174 = *(__int64 **)(v26 - 48);
                  if ( (__int64 *)v239 != v174 )
                    goto LABEL_163;
LABEL_186:
                  if ( !v174 )
                    goto LABEL_163;
LABEL_187:
                  v175 = a1->m128i_i64[1];
                  v252.m128i_i16[0] = 257;
                  v246 = (__int64)sub_1649960((__int64)v244);
                  v247 = v176;
                  v248 = &v246;
                  v250 = 773;
                  v249 = (__int64 *)".not";
                  v177 = sub_171CA90(
                           v175,
                           (__int64)v244,
                           v212,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
                  return sub_15FB440(26, (__int64 *)v239, (__int64)v177, (__int64)&v251, 0);
                case 42:
                  v144 = *(__int64 **)(v26 - 48);
                  if ( !v144 )
                    goto LABEL_165;
                  v145 = *(_QWORD *)(v26 - 24);
                  if ( *(_BYTE *)(v145 + 16) > 0x10u )
                    goto LABEL_165;
LABEL_162:
                  v209 = v212;
                  v218 = (__int64 *)v145;
                  v146 = sub_1719A40((_BYTE *)v239, v239, v78, v79);
                  v80 = v209;
                  if ( v146 )
                  {
                    v186 = sub_1596730((__int64)v218, v239, v78, v79);
                    v80 = v209;
                    if ( v186 )
                    {
                      if ( !sub_15962C0((__int64)v218, v239, v78, v79) )
                      {
                        v252.m128i_i16[0] = 257;
                        v192 = sub_15A2B90(
                                 v218,
                                 0,
                                 0,
                                 v79,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64);
                        return sub_15FB440(18, v144, v192, (__int64)&v251, 0);
                      }
                      v142 = *(_BYTE *)(v26 + 16);
                      v80 = v209;
LABEL_164:
                      if ( v142 == 47 )
                      {
                        v157 = *(_QWORD *)(v26 - 48);
                        if ( !v157 )
                          goto LABEL_167;
                        v158 = *(__int64 **)(v26 - 24);
                        if ( !v158 )
                          goto LABEL_167;
                        goto LABEL_174;
                      }
LABEL_165:
                      if ( v142 != 5 )
                        goto LABEL_167;
                      if ( *(_WORD *)(v26 + 18) != 23 )
                        goto LABEL_167;
                      v78 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                      v157 = *(_QWORD *)(v26 - 24 * v78);
                      if ( !v157 )
                        goto LABEL_167;
                      v158 = *(__int64 **)(v26 + 24 * (1 - v78));
                      if ( !v158 )
                        goto LABEL_167;
LABEL_174:
                      v221 = v80;
                      v244 = v158;
                      v159 = sub_1719A40((_BYTE *)v239, (__int64)v76, v78, v79);
                      v80 = v221;
                      if ( v159 )
                      {
                        v162 = (__int64 *)sub_1705480(
                                            *(double *)v12.m128_u64,
                                            *(double *)v13.m128i_i64,
                                            *(double *)a5.m128i_i64,
                                            (__int64)a1,
                                            v157,
                                            v160,
                                            v161);
                        v80 = v221;
                        if ( v162 )
                        {
                          v252.m128i_i16[0] = 257;
                          return sub_15FB440(23, v162, (__int64)v244, (__int64)&v251, 0);
                        }
                      }
LABEL_167:
                      v219 = v80;
                      v251.m128_u64[0] = (unsigned __int64)&v244;
                      v147 = sub_171E9E0(&v251, v26);
                      v150 = v219;
                      if ( v147 )
                      {
                        v151 = v244;
                        v152 = sub_16431D0(*v244);
                        v150 = v219;
                        if ( v152 == 1 )
                        {
                          v163 = a1->m128i_i64[1];
                          v164 = (__int64 **)*v11;
                          v252.m128i_i16[0] = 257;
                          v165 = sub_1708970(v163, 37, (__int64)v151, v164, (__int64 *)&v251);
                          v252.m128i_i16[0] = 257;
                          v32 = sub_15FB440(11, (__int64 *)v239, (__int64)v165, (__int64)&v251, 0);
                          v166 = sub_15F2380((__int64)v11);
                          sub_15F2330(v32, v166);
                          return v32;
                        }
                      }
                      v220 = v150;
                      v251.m128_u64[0] = (unsigned __int64)v245;
                      v252.m128i_i64[0] = (__int64)&v246;
                      v153 = sub_17255B0(&v251, v26, v148, v149);
                      v80 = v220;
                      if ( v153 )
                      {
                        v155 = a1->m128i_i64[1];
                        v252.m128i_i16[0] = 257;
                        v250 = 257;
                        v156 = sub_171D160(
                                 v155,
                                 *(__int64 *)v245,
                                 v246,
                                 v220,
                                 0,
                                 0,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64);
                      }
                      else
                      {
                        v167 = *(_BYTE *)(v26 + 16);
                        if ( v167 == 39 )
                        {
                          if ( !*(_QWORD *)(v26 - 48) )
                            goto LABEL_67;
                          *(_QWORD *)v245 = *(_QWORD *)(v26 - 48);
                          v169 = *(_QWORD *)(v26 - 24);
                          if ( *(_BYTE *)(v169 + 16) > 0x10u )
                            goto LABEL_67;
                        }
                        else
                        {
                          if ( v167 != 5 )
                            goto LABEL_67;
                          if ( *(_WORD *)(v26 + 18) != 15 )
                            goto LABEL_67;
                          v168 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
                          if ( !*(_QWORD *)(v26 - 24 * v168) )
                            goto LABEL_67;
                          *(_QWORD *)v245 = *(_QWORD *)(v26 - 24 * v168);
                          v169 = *(_QWORD *)(v26 + 24 * (1 - v168));
                          if ( !v169 )
                            goto LABEL_67;
                        }
                        v170 = a1->m128i_i64[1];
                        v252.m128i_i16[0] = 257;
                        v171 = sub_15A2B90(
                                 (__int64 *)v169,
                                 0,
                                 0,
                                 v154,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64);
                        v172 = sub_171D160(
                                 v170,
                                 *(__int64 *)v245,
                                 v171,
                                 (__int64 *)&v251,
                                 0,
                                 0,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64);
                        v252.m128i_i16[0] = 257;
                        v156 = v172;
                      }
                      return sub_15FB440(11, (__int64 *)v239, (__int64)v156, (__int64)&v251, 0);
                    }
                  }
LABEL_163:
                  v142 = *(_BYTE *)(v26 + 16);
                  goto LABEL_164;
                default:
                  goto LABEL_164;
              }
              v189 = a1->m128i_i64[1];
              v252.m128i_i16[0] = 257;
              v190 = sub_1649960(v26);
              v247 = v191;
              v246 = (__int64)v190;
              v250 = 261;
              v248 = &v246;
              v156 = sub_171D0D0(
                       v189,
                       v188,
                       (__int64)v244,
                       v212,
                       0,
                       0,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              return sub_15FB440(11, (__int64 *)v239, (__int64)v156, (__int64)&v251, 0);
            }
          }
          else
          {
            if ( v74 != 5 )
              goto LABEL_61;
            if ( *(_WORD *)(v239 + 18) != 13 )
              goto LABEL_61;
            v198 = *(_DWORD *)(v239 + 20) & 0xFFFFFFF;
            v199 = *(_QWORD *)(v239 - 24 * v198);
            if ( v26 != v199 )
              goto LABEL_61;
            if ( !v199 )
              goto LABEL_61;
            v108 = *(__int64 **)(v239 + 24 * (1 - v198));
            if ( !v108 )
              goto LABEL_61;
          }
          v248 = v108;
        }
        v252.m128i_i16[0] = 257;
        return sub_15FB530(v248, (__int64)&v251, 0, v72);
      }
LABEL_117:
      if ( !*v66 || (*v66 & (*v66 + 1LL)) != 0 )
        goto LABEL_57;
      goto LABEL_119;
    }
    v57 = *(_BYTE *)(v27 + 16);
    if ( v57 == 35 )
    {
      v42 = v27;
      if ( !*(_QWORD *)(v27 - 48) )
        goto LABEL_29;
      v240 = *(_QWORD *)(v27 - 48);
      v106 = sub_17198D0(*(_BYTE **)(v27 - 24), v39, v41, v27);
      v43 = (__int64 *)&v248;
      if ( !v106 )
        goto LABEL_29;
    }
    else
    {
      if ( v57 != 5 )
        goto LABEL_29;
      if ( *(_WORD *)(v27 + 18) != 11 )
        goto LABEL_29;
      v42 = v27;
      v41 = *(_DWORD *)(v27 + 20) & 0xFFFFFFF;
      if ( !*(_QWORD *)(v27 - 24 * v41) )
        goto LABEL_29;
      v240 = *(_QWORD *)(v27 - 24 * v41);
      v41 = *(_QWORD *)(v27 + 24 * (1 - v41));
      if ( *(_BYTE *)(v41 + 16) == 13 )
      {
        v39 = *(unsigned int *)(v41 + 32);
        if ( (unsigned int)v39 <= 0x40 )
        {
          v42 = (unsigned int)(64 - v39);
          if ( *(_QWORD *)(v41 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) )
            goto LABEL_29;
          goto LABEL_48;
        }
        v58 = sub_16A58F0(v41 + 24);
        v39 = (unsigned int)v39;
        v43 = (__int64 *)&v248;
        v59 = (_DWORD)v39 == v58;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) != 16 )
          goto LABEL_29;
        v233 = v41;
        v126 = sub_15A1020((_BYTE *)v41, v39, v41, v27);
        v41 = v233;
        v43 = (__int64 *)&v248;
        if ( !v126 || *(_BYTE *)(v126 + 16) != 13 )
        {
          v39 = 0;
          v238 = *(_DWORD *)(*(_QWORD *)v233 + 32LL);
          while ( v238 != (_DWORD)v39 )
          {
            v205 = v43;
            v222 = v41;
            v200 = sub_15A0A60(v41, v39);
            v43 = v205;
            if ( !v200 )
              goto LABEL_29;
            v42 = *(unsigned __int8 *)(v200 + 16);
            v41 = v222;
            v39 = (unsigned int)v39;
            if ( (_BYTE)v42 != 9 )
            {
              if ( (_BYTE)v42 != 13 )
                goto LABEL_29;
              v201 = *(_DWORD *)(v200 + 32);
              if ( v201 <= 0x40 )
              {
                v42 = 64 - v201;
                v203 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v201) == *(_QWORD *)(v200 + 24);
              }
              else
              {
                v204 = v205;
                v206 = *(_DWORD *)(v200 + 32);
                v202 = sub_16A58F0(v200 + 24);
                v41 = v222;
                v39 = (unsigned int)v39;
                v43 = v204;
                v203 = v206 == v202;
              }
              if ( !v203 )
                goto LABEL_29;
            }
            v39 = (unsigned int)(v39 + 1);
          }
          goto LABEL_48;
        }
        v127 = *(_DWORD *)(v126 + 32);
        if ( v127 <= 0x40 )
        {
          v42 = 64 - v127;
          v41 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v127);
          v59 = v41 == *(_QWORD *)(v126 + 24);
        }
        else
        {
          v234 = *(_DWORD *)(v126 + 32);
          v128 = sub_16A58F0(v126 + 24);
          v41 = v234;
          v43 = (__int64 *)&v248;
          v59 = v234 == v128;
        }
      }
      if ( !v59 )
        goto LABEL_29;
    }
LABEL_48:
    v60 = a1->m128i_i64[1];
    v61 = v240;
    v250 = 257;
    v252.m128i_i16[0] = 257;
    v62 = (__int64 *)sub_171CA90(
                       v60,
                       v26,
                       v43,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
    return sub_15FB440(11, v62, v61, (__int64)&v251, 0);
  }
  return sub_170E100(
           a1->m128i_i64,
           a2,
           (__int64)v22,
           v12,
           *(double *)v13.m128i_i64,
           *(double *)a5.m128i_i64,
           a6,
           v24,
           v25,
           a9,
           a10);
}
