// Function: sub_17233C0
// Address: 0x17233c0
//
__int64 __fastcall sub_17233C0(
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
  __int64 v10; // r14
  __int64 *v11; // rbx
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  char v14; // r13
  char v15; // al
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  char v26; // al
  _QWORD *v27; // rdx
  double v28; // xmm4_8
  double v29; // xmm5_8
  unsigned __int64 v30; // r13
  __int64 v31; // r15
  char v32; // al
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rcx
  double v45; // xmm4_8
  double v46; // xmm5_8
  __int32 v47; // eax
  int v48; // r10d
  __int64 v49; // rax
  unsigned int v50; // eax
  __int64 v51; // rsi
  unsigned int v52; // ecx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rbx
  bool v56; // al
  __int64 v57; // rdi
  bool v58; // al
  __int64 v59; // rdx
  __int64 v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int32 v63; // eax
  int v64; // eax
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 *v68; // rsi
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // r14
  __int64 v72; // r15
  __int64 *v73; // r13
  __int64 v74; // rax
  __int32 v75; // eax
  int v76; // eax
  __int32 v77; // eax
  __int64 v78; // r14
  const char *v79; // rax
  __int64 *v80; // rdx
  __int64 *v81; // rax
  __int64 v82; // rax
  char v83; // al
  __int64 v84; // rdx
  __int64 v85; // rcx
  _BYTE *v86; // r8
  char v87; // al
  __int64 v88; // rdx
  __int64 v89; // rcx
  unsigned int v90; // r8d
  _QWORD *v91; // rax
  _QWORD *v92; // rdx
  __int64 v93; // rdi
  void *v94; // rax
  __int64 v95; // r15
  __int64 v96; // rax
  char v97; // al
  __int64 v98; // r13
  bool v99; // al
  __int64 v100; // r15
  __int64 v101; // rdx
  _BYTE *v102; // r13
  unsigned int v103; // edx
  bool v104; // al
  __int64 v105; // rdi
  __int64 v106; // r13
  __int64 v107; // rax
  bool v108; // al
  const char *v109; // rax
  __int64 *v110; // rdx
  __int64 v111; // rax
  unsigned int v112; // r13d
  unsigned int v113; // ebx
  __int64 v114; // rax
  char v115; // cl
  unsigned int v116; // esi
  __int64 v117; // rax
  unsigned int v118; // r13d
  __int64 *v119; // r13
  unsigned __int8 v120; // al
  bool v121; // al
  unsigned int v122; // r15d
  __int64 v123; // rax
  int v124; // eax
  bool v125; // al
  __int64 v126; // r15
  __int64 v127; // rax
  char v128; // al
  __int64 v129; // rdx
  __int64 v130; // r13
  int v131; // eax
  bool v132; // al
  _QWORD *v133; // rax
  _QWORD *v134; // rdx
  __int64 v135; // rdi
  unsigned __int8 *v136; // rax
  __int64 v137; // rdi
  _QWORD *v138; // rax
  __int64 v139; // rdi
  __int64 v140; // rax
  unsigned int v141; // r13d
  bool v142; // al
  __int64 *v143; // r13
  unsigned __int8 v144; // al
  int v145; // eax
  bool v146; // al
  unsigned int v147; // r15d
  __int64 v148; // rax
  int v149; // eax
  bool v150; // al
  __int64 v151; // rax
  unsigned int v152; // r13d
  int v153; // eax
  char v154; // al
  __int64 v155; // rax
  _QWORD *v156; // rsi
  __int64 *v157; // rdx
  __int64 *v158; // rsi
  __int64 *v159; // rdx
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rdi
  __int64 v163; // rdx
  __int64 v164; // rsi
  unsigned __int8 *v165; // rax
  __int64 v166; // r13
  _QWORD *v167; // rax
  __int64 v168; // rax
  __int64 v169; // rax
  __int64 v170; // rdi
  __int64 v171; // rdx
  __int64 v172; // rsi
  unsigned __int8 *v173; // rax
  __int64 v174; // r13
  _QWORD *v175; // rax
  __int64 v176; // rax
  __int64 v177; // rax
  unsigned int v178; // r13d
  unsigned int v179; // r15d
  __int64 v180; // rax
  int v181; // eax
  bool v182; // al
  unsigned int v183; // r15d
  __int64 v184; // rax
  int v185; // eax
  bool v186; // al
  char v187; // [rsp+8h] [rbp-178h]
  char v188; // [rsp+10h] [rbp-170h]
  __int64 v189; // [rsp+10h] [rbp-170h]
  __int64 v190; // [rsp+18h] [rbp-168h]
  unsigned int v191; // [rsp+18h] [rbp-168h]
  __int64 v192; // [rsp+18h] [rbp-168h]
  unsigned int v193; // [rsp+20h] [rbp-160h]
  bool v194; // [rsp+20h] [rbp-160h]
  bool v195; // [rsp+20h] [rbp-160h]
  __int64 v196; // [rsp+28h] [rbp-158h]
  __int64 *v197; // [rsp+28h] [rbp-158h]
  _BYTE *v198; // [rsp+28h] [rbp-158h]
  int v199; // [rsp+28h] [rbp-158h]
  int v200; // [rsp+28h] [rbp-158h]
  int v201; // [rsp+28h] [rbp-158h]
  int v202; // [rsp+28h] [rbp-158h]
  int v203; // [rsp+28h] [rbp-158h]
  __int64 v204; // [rsp+30h] [rbp-150h]
  int v205; // [rsp+30h] [rbp-150h]
  __int64 v206; // [rsp+30h] [rbp-150h]
  int v207; // [rsp+30h] [rbp-150h]
  __int64 v208; // [rsp+30h] [rbp-150h]
  __int64 v209; // [rsp+30h] [rbp-150h]
  __int64 v210; // [rsp+30h] [rbp-150h]
  char v211; // [rsp+38h] [rbp-148h]
  __int64 v212; // [rsp+38h] [rbp-148h]
  __int64 v213; // [rsp+38h] [rbp-148h]
  unsigned __int64 v214; // [rsp+38h] [rbp-148h]
  __int64 v215; // [rsp+40h] [rbp-140h]
  _QWORD *v216; // [rsp+40h] [rbp-140h]
  __int64 *v217; // [rsp+40h] [rbp-140h]
  __int64 *v218; // [rsp+40h] [rbp-140h]
  __int64 **v219; // [rsp+48h] [rbp-138h]
  int v220; // [rsp+48h] [rbp-138h]
  int v221; // [rsp+48h] [rbp-138h]
  __int64 *v222; // [rsp+48h] [rbp-138h]
  int v223; // [rsp+48h] [rbp-138h]
  int v224; // [rsp+48h] [rbp-138h]
  __int64 v225; // [rsp+48h] [rbp-138h]
  int v226; // [rsp+48h] [rbp-138h]
  int v227; // [rsp+48h] [rbp-138h]
  int v228; // [rsp+48h] [rbp-138h]
  int v229; // [rsp+48h] [rbp-138h]
  int v230; // [rsp+58h] [rbp-128h] BYREF
  int v231; // [rsp+5Ch] [rbp-124h] BYREF
  __int64 *v232; // [rsp+60h] [rbp-120h] BYREF
  __int64 v233; // [rsp+68h] [rbp-118h] BYREF
  __int64 v234; // [rsp+70h] [rbp-110h] BYREF
  char v235; // [rsp+78h] [rbp-108h] BYREF
  __int64 v236; // [rsp+80h] [rbp-100h] BYREF
  __int64 v237; // [rsp+88h] [rbp-F8h] BYREF
  __int64 v238; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v239; // [rsp+98h] [rbp-E8h] BYREF
  __int64 v240; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v241; // [rsp+A8h] [rbp-D8h] BYREF
  unsigned __int64 v242; // [rsp+B0h] [rbp-D0h] BYREF
  __int32 v243; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v244; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned int v245; // [rsp+C8h] [rbp-B8h]
  unsigned __int64 v246; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 *v247; // [rsp+D8h] [rbp-A8h]
  __int64 *v248; // [rsp+E0h] [rbp-A0h]
  signed __int64 v249; // [rsp+F0h] [rbp-90h] BYREF
  __int64 *v250; // [rsp+F8h] [rbp-88h]
  __int64 *v251; // [rsp+100h] [rbp-80h]
  __int64 *v252; // [rsp+108h] [rbp-78h]
  __int64 *v253; // [rsp+110h] [rbp-70h]
  __m128 v254; // [rsp+120h] [rbp-60h] BYREF
  __m128i v255; // [rsp+130h] [rbp-50h] BYREF
  __int64 v256; // [rsp+140h] [rbp-40h]

  v10 = a2;
  v11 = (__int64 *)a1;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v256 = a2;
  v13 = _mm_loadu_si128(a1 + 168);
  v254 = v12;
  v255 = v13;
  v14 = sub_15F2370(a2);
  v15 = sub_15F2380(a2);
  v16 = sub_13DEB20(*(_QWORD *)(a2 - 48), *(_QWORD *)(a2 - 24), v15, v14, &v254);
  if ( v16 )
  {
    if ( *(_QWORD *)(a2 + 8) )
    {
      v18 = (__int64)v16;
      sub_17205C0(a1->m128i_i64[0], a2);
      if ( a2 == v18 )
        v18 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v18, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v19, v20, a9, a10);
      return v10;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_170D400(a1, a2, v17, (__m128i)v12, *(double *)v13.m128i_i64, a5) )
    return v10;
  v22 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)a2,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
  if ( v22 )
    return v22;
  v27 = sub_1708300(a1, (unsigned __int8 *)a2, (__m128i)v12, v13, a5);
  if ( !v27 )
  {
    v22 = sub_1722490(
            a1->m128i_i64,
            (_BYTE *)a2,
            v12,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            a6,
            v28,
            v29,
            a9,
            a10);
    if ( v22 )
      return v22;
    v30 = *(_QWORD *)(a2 - 24);
    v31 = *(_QWORD *)(a2 - 48);
    v219 = *(__int64 ***)a2;
    if ( *(_BYTE *)(v30 + 16) == 13 )
    {
      v32 = *(_BYTE *)(v31 + 16);
      if ( v32 != 52 )
      {
        if ( v32 != 5 )
          goto LABEL_29;
        if ( *(_WORD *)(v31 + 18) != 28 )
          goto LABEL_29;
        v59 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        v212 = *(_QWORD *)(v31 - 24 * v59);
        if ( !v212 )
          goto LABEL_29;
        v204 = *(_QWORD *)(v31 + 24 * (1 - v59));
        if ( *(_BYTE *)(v204 + 16) != 13 )
          goto LABEL_29;
LABEL_42:
        v193 = sub_16431D0((__int64)v219);
        v190 = v30 + 24;
        sub_13A38D0((__int64)&v249, v30 + 24);
        if ( (unsigned int)v250 > 0x40 )
          sub_16A8F40(&v249);
        else
          v249 = ~v249 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v250);
        sub_16A7400((__int64)&v249);
        v47 = (int)v250;
        LODWORD(v250) = 0;
        v254.m128_i32[2] = v47;
        v254.m128_u64[0] = v249;
        v196 = v204 + 24;
        v188 = sub_1455820(v204 + 24, &v254);
        sub_135E100((__int64 *)&v254);
        sub_135E100(&v249);
        if ( v188 )
        {
          if ( sub_14A9C60(v190) )
          {
            v48 = v193 - *(_DWORD *)(v30 + 32) + sub_1455840(v190);
          }
          else
          {
            if ( !sub_14A9C60(v196) )
              goto LABEL_48;
            v48 = v193 - *(_DWORD *)(v204 + 32) + sub_1455840(v196);
          }
          if ( v48 )
          {
            v191 = v48;
            sub_171A350((__int64)&v254, v193, v48);
            if ( (unsigned __int8)sub_14C1670(
                                    v212,
                                    (__int64)&v254,
                                    a1[166].m128i_i64[1],
                                    0,
                                    a1[165].m128i_i64[0],
                                    a2,
                                    a1[166].m128i_i64[0]) )
            {
              sub_135E100((__int64 *)&v254);
              v71 = sub_15A0680((__int64)v219, v191, 0);
              v255.m128i_i16[0] = 259;
              v72 = a1->m128i_i64[1];
              v254.m128_u64[0] = (unsigned __int64)"sext";
              if ( *(_BYTE *)(v212 + 16) > 0x10u || *(_BYTE *)(v71 + 16) > 0x10u )
              {
                v73 = (__int64 *)sub_170A2B0(v72, 23, (__int64 *)v212, v71, (__int64 *)&v254, 0, 0);
              }
              else
              {
                v73 = (__int64 *)sub_15A2D50(
                                   (__int64 *)v212,
                                   v71,
                                   0,
                                   0,
                                   *(double *)v12.m128_u64,
                                   *(double *)v13.m128i_i64,
                                   *(double *)a5.m128i_i64);
                v74 = sub_14DBA30((__int64)v73, *(_QWORD *)(v72 + 96), 0);
                if ( v74 )
                  v73 = (__int64 *)v74;
              }
              v255.m128i_i16[0] = 257;
              return sub_15FB440(25, v73, v71, (__int64)&v254, 0);
            }
            sub_135E100((__int64 *)&v254);
          }
        }
LABEL_48:
        v49 = *(_QWORD *)(v31 + 8);
        if ( v49 )
        {
          if ( !*(_QWORD *)(v49 + 8) )
          {
            sub_13A38D0((__int64)&v249, v196);
            sub_16A7490((__int64)&v249, 1);
            v63 = (int)v250;
            LODWORD(v250) = 0;
            v254.m128_i32[2] = v63;
            v254.m128_u64[0] = v249;
            v194 = sub_14A9C60((__int64)&v254);
            sub_135E100((__int64 *)&v254);
            sub_135E100(&v249);
            if ( v194 )
            {
              sub_14C2530(
                (__int64)&v254,
                (__int64 *)v212,
                a1[166].m128i_i64[1],
                0,
                a1[165].m128i_i64[0],
                a2,
                a1[166].m128i_i64[0],
                0);
              sub_13A38D0((__int64)&v246, v196);
              if ( (unsigned int)v247 > 0x40 )
                sub_16A89F0((__int64 *)&v246, (__int64 *)&v254);
              else
                v246 |= v254.m128_u64[0];
              v64 = (int)v247;
              LODWORD(v247) = 0;
              LODWORD(v250) = v64;
              v249 = v246;
              v195 = sub_1454FB0((__int64)&v249);
              sub_135E100(&v249);
              sub_135E100((__int64 *)&v246);
              if ( v195 )
              {
                LOWORD(v251) = 257;
                v65 = (__int64 *)sub_15A2B30(
                                   (__int64 *)v204,
                                   v30,
                                   0,
                                   0,
                                   *(double *)v12.m128_u64,
                                   *(double *)v13.m128i_i64,
                                   *(double *)a5.m128i_i64);
                v10 = sub_15FB440(13, v65, v212, (__int64)&v249, 0);
                sub_135E100(v255.m128i_i64);
                sub_135E100((__int64 *)&v254);
                return v10;
              }
              sub_135E100(v255.m128i_i64);
              sub_135E100((__int64 *)&v254);
            }
          }
        }
        v50 = *(_DWORD *)(v204 + 32);
        v51 = *(_QWORD *)(v204 + 24);
        v52 = v50 - 1;
        if ( v50 <= 0x40 )
        {
          if ( v51 != 1LL << v52 )
            goto LABEL_29;
        }
        else if ( (*(_QWORD *)(v51 + 8LL * (v52 >> 6)) & (1LL << v52)) == 0 || (unsigned int)sub_16A58A0(v196) != v52 )
        {
          goto LABEL_29;
        }
        v255.m128i_i16[0] = 257;
        v53 = sub_15A2D30(
                (__int64 *)v204,
                v30,
                *(double *)v12.m128_u64,
                *(double *)v13.m128i_i64,
                *(double *)a5.m128i_i64);
        return sub_15FB440(11, (__int64 *)v212, v53, (__int64)&v254, 0);
      }
      v212 = *(_QWORD *)(v31 - 48);
      if ( v212 )
      {
        v204 = *(_QWORD *)(v31 - 24);
        if ( *(_BYTE *)(v204 + 16) == 13 )
          goto LABEL_42;
      }
    }
LABEL_29:
    v33 = (__int64)v219;
    if ( *((_BYTE *)v219 + 8) == 16 )
      v33 = *v219[2];
    if ( sub_1642F90(v33, 1) )
    {
      v255.m128i_i16[0] = 257;
      return sub_15FB440(28, (__int64 *)v31, v30, (__int64)&v254, 0);
    }
    if ( v31 == v30 )
    {
      v255.m128i_i16[0] = 257;
      v54 = sub_15A0680((__int64)v219, 1, 0);
      v55 = sub_15FB440(23, (__int64 *)v31, v54, (__int64)&v254, 0);
      v56 = sub_15F2380(v10);
      sub_15F2330(v55, v56);
      v57 = v10;
      v10 = v55;
      v58 = sub_15F2370(v57);
      sub_15F2310(v55, v58);
      return v10;
    }
    v254.m128_u64[1] = (unsigned __int64)&v232;
    if ( (unsigned __int8)sub_171ECC0((__int64)&v254, v31, v34, v35) )
    {
      v254.m128_u64[1] = (unsigned __int64)&v233;
      if ( (unsigned __int8)sub_171ECC0((__int64)&v254, v30, v36, v37) )
      {
        v38 = v11[1];
        v255.m128i_i16[0] = 257;
        v39 = (__int64 *)sub_17094A0(
                           v38,
                           (__int64)v232,
                           v233,
                           (__int64 *)&v254,
                           0,
                           0,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
        v255.m128i_i16[0] = 257;
        return sub_15FB530(v39, (__int64)&v254, 0, v40);
      }
      else
      {
        v255.m128i_i16[0] = 257;
        return sub_15FB440(13, (__int64 *)v30, (__int64)v232, (__int64)&v254, 0);
      }
    }
    v254.m128_u64[1] = (unsigned __int64)&v233;
    if ( (unsigned __int8)sub_171ECC0((__int64)&v254, v30, v36, v37) )
    {
      v255.m128i_i16[0] = 257;
      return sub_15FB440(13, (__int64 *)v31, v233, (__int64)&v254, 0);
    }
    v43 = (__int64)sub_1721670(
                     v10,
                     v11[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64,
                     v41,
                     v42);
    if ( v43 )
      return sub_170E100(v11, v10, v43, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v45, v46, a9, a10);
    v60 = *(_QWORD *)(v10 - 48);
    v254.m128_u64[0] = (unsigned __int64)&v232;
    v255.m128i_i64[0] = (__int64)&v233;
    if ( sub_171CD50(&v254, v60, 0, v44) && sub_171DA10(&v255, *(_QWORD *)(v10 - 24), v61, v62)
      || sub_171CD50(&v254, *(_QWORD *)(v10 - 24), v61, v62) && sub_171DA10(&v255, *(_QWORD *)(v10 - 48), v66, v67) )
    {
      v255.m128i_i16[0] = 257;
      return sub_15FB440(13, v232, v233, (__int64)&v254, 0);
    }
    v68 = (__int64 *)v10;
    v43 = sub_171DFC0((__int64)v11, v10, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
    if ( v43 )
      return sub_170E100(v11, v10, v43, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v45, v46, a9, a10);
    if ( !unk_4FA21A0 )
    {
      v68 = (__int64 *)v30;
      if ( (unsigned __int8)sub_14BB210(v31, v30, v11[333], v11[330], v10, v11[332]) )
      {
        v255.m128i_i16[0] = 257;
        return sub_15FB440(27, (__int64 *)v31, v30, (__int64)&v254, 0);
      }
    }
    if ( *(_BYTE *)(v30 + 16) == 13 )
    {
      v70 = *(_QWORD *)(v31 + 8);
      if ( !v70 || *(_QWORD *)(v70 + 8) )
      {
LABEL_80:
        if ( *(_BYTE *)(v31 + 16) != 79 )
          goto LABEL_81;
        goto LABEL_117;
      }
      v43 = *(unsigned __int8 *)(v31 + 16);
      if ( (_BYTE)v43 == 50 )
      {
        v69 = *(_QWORD *)(v31 - 48);
        v189 = v69;
        if ( !v69 )
          goto LABEL_80;
        v69 = *(_QWORD *)(v31 - 24);
        v213 = v69;
        if ( *(_BYTE *)(v69 + 16) != 13 )
          goto LABEL_80;
      }
      else
      {
        if ( (_BYTE)v43 != 5 )
          goto LABEL_80;
        if ( *(_WORD *)(v31 + 18) != 26 )
          goto LABEL_80;
        v69 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        v43 = -24 * v69;
        v68 = *(__int64 **)(v31 - 24 * v69);
        v189 = (__int64)v68;
        if ( !v68 )
          goto LABEL_80;
        v43 = 24 * (1 - v69);
        v69 = *(_QWORD *)(v31 + v43);
        v213 = v69;
        if ( *(_BYTE *)(v69 + 16) != 13 )
          goto LABEL_80;
      }
      v197 = (__int64 *)(v30 + 24);
      sub_13A38D0((__int64)&v249, v30 + 24);
      sub_1718FB0(&v249, (__int64 *)(v213 + 24));
      v75 = (int)v250;
      v68 = (__int64 *)&v254;
      LODWORD(v250) = 0;
      v254.m128_i32[2] = v75;
      v254.m128_u64[0] = v249;
      v187 = sub_1455820(v30 + 24, &v254);
      sub_135E100((__int64 *)&v254);
      sub_135E100(&v249);
      if ( v187 )
      {
        sub_13A38D0((__int64)&v244, (__int64)v197);
        if ( v245 > 0x40 )
          sub_16A8F40((__int64 *)&v244);
        else
          v244 = ~v244 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v245);
        sub_16A7400((__int64)&v244);
        LODWORD(v247) = v245;
        v245 = 0;
        v246 = v244;
        sub_1718FB0((__int64 *)&v246, v197);
        v76 = (int)v247;
        LODWORD(v247) = 0;
        LODWORD(v250) = v76;
        v249 = v246;
        sub_16A7800((__int64)&v249, 1u);
        v77 = (int)v250;
        LODWORD(v250) = 0;
        v254.m128_i32[2] = v77;
        v254.m128_u64[0] = v249;
        sub_13D0570((__int64)&v254);
        v243 = v254.m128_i32[2];
        v242 = v254.m128_u64[0];
        sub_135E100(&v249);
        sub_135E100((__int64 *)&v246);
        sub_135E100((__int64 *)&v244);
        sub_13A38D0((__int64)&v254, (__int64)&v242);
        sub_1718FB0((__int64 *)&v254, (__int64 *)(v213 + 24));
        LODWORD(v247) = v254.m128_i32[2];
        v68 = (__int64 *)&v246;
        v246 = v254.m128_u64[0];
        if ( sub_1455820((__int64)&v242, &v246) )
        {
          v78 = v11[1];
          v79 = sub_1649960(v31);
          v250 = v80;
          v254.m128_u64[0] = (unsigned __int64)&v249;
          v249 = (signed __int64)v79;
          v255.m128i_i16[0] = 261;
          v81 = (__int64 *)sub_17094A0(
                             v78,
                             v189,
                             v30,
                             (__int64 *)&v254,
                             0,
                             0,
                             *(double *)v12.m128_u64,
                             *(double *)v13.m128i_i64,
                             *(double *)a5.m128i_i64);
          v255.m128i_i16[0] = 257;
          v10 = sub_15FB440(26, v81, v213, (__int64)&v254, 0);
          sub_135E100((__int64 *)&v246);
          sub_135E100((__int64 *)&v242);
          return v10;
        }
        sub_135E100((__int64 *)&v246);
        sub_135E100((__int64 *)&v242);
      }
    }
    if ( *(_BYTE *)(v31 + 16) != 79 )
    {
      if ( *(_BYTE *)(v30 + 16) != 79 )
        goto LABEL_81;
      v82 = *(_QWORD *)(v30 + 8);
      if ( !v82 || *(_QWORD *)(v82 + 8) )
        goto LABEL_81;
      v216 = (_QWORD *)v31;
      v214 = v30;
LABEL_111:
      v192 = *(_QWORD *)(v214 - 48);
      v198 = *(_BYTE **)(v214 - 24);
      v83 = sub_1719A40(v198, (__int64)v68, v43, v69);
      v86 = (_BYTE *)v192;
      if ( v83 )
      {
        v68 = (__int64 *)v192;
        v254.m128_u64[0] = (unsigned __int64)&v249;
        v254.m128_u64[1] = (unsigned __int64)v216;
        v87 = sub_171EF90((__int64)&v254, v192, v84, v85, v192);
        v86 = (_BYTE *)v192;
        if ( v87 )
        {
          v255.m128i_i16[0] = 257;
          return sub_14EDD70(*(_QWORD *)(v214 - 72), (_QWORD *)v249, (__int64)v216, (__int64)&v254, 0, 0);
        }
      }
      if ( sub_1719A40(v86, (__int64)v68, v84, v85) )
      {
        v254.m128_u64[0] = (unsigned __int64)&v249;
        v254.m128_u64[1] = (unsigned __int64)v216;
        if ( (unsigned __int8)sub_171EF90((__int64)&v254, (__int64)v198, v88, v89, v90) )
        {
          v255.m128i_i16[0] = 257;
          return sub_14EDD70(*(_QWORD *)(v214 - 72), v216, v249, (__int64)&v254, 0, 0);
        }
      }
LABEL_81:
      if ( *(_BYTE *)(v31 + 16) != 62 )
        goto LABEL_82;
      v154 = *(_BYTE *)(v30 + 16);
      if ( LOBYTE(qword_4FA1CA0[20]) || v154 != 13 )
      {
        if ( v154 != 62 )
          goto LABEL_83;
        v156 = *(_QWORD **)(v31 - 24);
        v157 = *(__int64 **)(v30 - 24);
        if ( *v157 != *v156 )
          goto LABEL_83;
      }
      else
      {
        v155 = *(_QWORD *)(v31 + 8);
        if ( !v155 || *(_QWORD *)(v155 + 8) )
          goto LABEL_83;
        v217 = (__int64 *)sub_15A43B0(v30, **(__int64 ****)(v31 - 24), 0);
        if ( v30 == sub_15A4460((unsigned __int64)v217, v219, 0)
          && (unsigned int)sub_171CA60(v11, *(_QWORD *)(v31 - 24), v217, v10) == 2 )
        {
          v170 = v11[1];
          v255.m128i_i16[0] = 259;
          v171 = (__int64)v217;
          v254.m128_u64[0] = (unsigned __int64)"addconv";
          v172 = *(_QWORD *)(v31 - 24);
LABEL_290:
          v173 = sub_17094A0(
                   v170,
                   v172,
                   v171,
                   (__int64 *)&v254,
                   0,
                   1,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          v255.m128i_i16[0] = 257;
          v174 = (__int64)v173;
          v175 = sub_1648A60(56, 1u);
          v10 = (__int64)v175;
          if ( v175 )
            sub_15FC810((__int64)v175, v174, (__int64)v219, (__int64)&v254, 0);
          return v10;
        }
        if ( *(_BYTE *)(v30 + 16) != 62 )
          goto LABEL_82;
        v156 = *(_QWORD **)(v31 - 24);
        v157 = *(__int64 **)(v30 - 24);
        if ( *v157 != *v156 )
          goto LABEL_271;
      }
      v168 = *(_QWORD *)(v31 + 8);
      if ( !v168 || *(_QWORD *)(v168 + 8) )
      {
        v169 = *(_QWORD *)(v30 + 8);
        if ( !v169 || *(_QWORD *)(v169 + 8) )
        {
LABEL_271:
          if ( *(_BYTE *)(v31 + 16) != 61 )
            goto LABEL_83;
LABEL_272:
          if ( *(_BYTE *)(v30 + 16) == 61 )
          {
            v158 = *(__int64 **)(v31 - 24);
            v159 = *(__int64 **)(v30 - 24);
            if ( *v159 == *v158
              && ((v160 = *(_QWORD *)(v31 + 8)) != 0 && !*(_QWORD *)(v160 + 8)
               || (v161 = *(_QWORD *)(v30 + 8)) != 0 && !*(_QWORD *)(v161 + 8))
              && (unsigned int)sub_171CA30(v11, v158, v159, v10) == 2 )
            {
              v162 = v11[1];
              v255.m128i_i16[0] = 259;
              v254.m128_u64[0] = (unsigned __int64)"addconv";
              v163 = *(_QWORD *)(v30 - 24);
              v164 = *(_QWORD *)(v31 - 24);
              goto LABEL_280;
            }
          }
LABEL_83:
          v254.m128_u64[0] = (unsigned __int64)&v232;
          v254.m128_u64[1] = (unsigned __int64)&v233;
          v255.m128i_i64[0] = (__int64)&v232;
          v255.m128i_i64[1] = (__int64)&v233;
          if ( sub_171F030(&v254, v10) )
          {
            v255.m128i_i16[0] = 257;
            return sub_15FB440(27, v232, v233, (__int64)&v254, 0);
          }
          v23 = v10;
          v254.m128_u64[0] = (unsigned __int64)&v232;
          v254.m128_u64[1] = (unsigned __int64)&v233;
          v255.m128i_i64[0] = (__int64)&v232;
          v255.m128i_i64[1] = (__int64)&v233;
          v211 = sub_171F330(&v254, v10);
          if ( v211 )
          {
            sub_1593B40((_QWORD *)(v10 - 48), (__int64)v232);
            sub_1593B40((_QWORD *)(v10 - 24), v233);
            return v10;
          }
          if ( !sub_15F2380(v10) )
          {
            v23 = v31;
            if ( (unsigned int)sub_171CA60(v11, v31, (__int64 *)v30, v10) == 2 )
            {
              v23 = 1;
              sub_15F2330(v10, 1);
              v211 = 1;
            }
          }
          if ( !sub_15F2370(v10) )
          {
            v23 = v31;
            if ( (unsigned int)sub_171CA30(v11, (__int64 *)v31, (__int64 *)v30, v10) == 2 )
            {
              v23 = 1;
              sub_15F2310(v10, 1);
              v211 = 1;
            }
          }
          v215 = v11[1];
          v26 = *(_BYTE *)(v10 + 16);
          if ( v26 != 35 )
          {
            if ( v26 != 5 )
              goto LABEL_16;
            if ( *(_WORD *)(v10 + 18) != 11 )
              goto LABEL_16;
            v95 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
            v96 = *(_QWORD *)(v95 + 8);
            if ( !v96 || *(_QWORD *)(v96 + 8) )
              goto LABEL_16;
            v97 = *(_BYTE *)(v95 + 16);
            if ( v97 != 47 )
            {
              if ( v97 != 5 || *(_WORD *)(v95 + 18) != 23 )
                goto LABEL_16;
              v98 = *(_QWORD *)(v95 - 24LL * (*(_DWORD *)(v95 + 20) & 0xFFFFFFF));
              if ( *(_BYTE *)(v98 + 16) == 13 )
              {
                if ( *(_DWORD *)(v98 + 32) <= 0x40u )
                {
                  v99 = *(_QWORD *)(v98 + 24) == 1;
                }
                else
                {
                  v220 = *(_DWORD *)(v98 + 32);
                  v99 = v220 - 1 == (unsigned int)sub_16A57B0(v98 + 24);
                }
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v98 + 8LL) != 16 )
                  goto LABEL_16;
                v117 = sub_15A1020(*(_BYTE **)(v95 - 24LL * (*(_DWORD *)(v95 + 20) & 0xFFFFFFF)), v23, v24, v25);
                if ( !v117 || *(_BYTE *)(v117 + 16) != 13 )
                {
                  v206 = v95;
                  v122 = 0;
                  v224 = *(_DWORD *)(*(_QWORD *)v98 + 32LL);
                  while ( v224 != v122 )
                  {
                    v23 = v122;
                    v123 = sub_15A0A60(v98, v122);
                    if ( !v123 )
                      goto LABEL_16;
                    v25 = *(unsigned __int8 *)(v123 + 16);
                    if ( (_BYTE)v25 != 9 )
                    {
                      if ( (_BYTE)v25 != 13 )
                        goto LABEL_16;
                      v25 = *(unsigned int *)(v123 + 32);
                      if ( (unsigned int)v25 <= 0x40 )
                      {
                        v125 = *(_QWORD *)(v123 + 24) == 1;
                      }
                      else
                      {
                        v200 = *(_DWORD *)(v123 + 32);
                        v124 = sub_16A57B0(v123 + 24);
                        v25 = (unsigned int)(v200 - 1);
                        v125 = (_DWORD)v25 == v124;
                      }
                      if ( !v125 )
                        goto LABEL_16;
                    }
                    ++v122;
                  }
                  v95 = v206;
LABEL_149:
                  v100 = *(_QWORD *)(v95 + 24 * (1LL - (*(_DWORD *)(v95 + 20) & 0xFFFFFFF)));
                  if ( !v100 )
                    goto LABEL_16;
                  goto LABEL_150;
                }
                v118 = *(_DWORD *)(v117 + 32);
                if ( v118 <= 0x40 )
                  v99 = *(_QWORD *)(v117 + 24) == 1;
                else
                  v99 = v118 - 1 == (unsigned int)sub_16A57B0(v117 + 24);
              }
              if ( !v99 )
                goto LABEL_16;
              goto LABEL_149;
            }
            v119 = *(__int64 **)(v95 - 48);
            v120 = *((_BYTE *)v119 + 16);
            if ( v120 == 13 )
            {
              if ( *((_DWORD *)v119 + 8) <= 0x40u )
              {
                v121 = v119[3] == 1;
              }
              else
              {
                v223 = *((_DWORD *)v119 + 8);
                v121 = v223 - 1 == (unsigned int)sub_16A57B0((__int64)(v119 + 3));
              }
            }
            else
            {
              if ( *(_BYTE *)(*v119 + 8) != 16 || v120 > 0x10u )
                goto LABEL_16;
              v177 = sub_15A1020(*(_BYTE **)(v95 - 48), v23, v24, *v119);
              if ( !v177 || *(_BYTE *)(v177 + 16) != 13 )
              {
                v210 = v95;
                v183 = 0;
                v229 = *(_DWORD *)(*v119 + 32);
                while ( v229 != v183 )
                {
                  v23 = v183;
                  v184 = sub_15A0A60((__int64)v119, v183);
                  if ( !v184 )
                    goto LABEL_16;
                  v25 = *(unsigned __int8 *)(v184 + 16);
                  if ( (_BYTE)v25 != 9 )
                  {
                    if ( (_BYTE)v25 != 13 )
                      goto LABEL_16;
                    v25 = *(unsigned int *)(v184 + 32);
                    if ( (unsigned int)v25 <= 0x40 )
                    {
                      v186 = *(_QWORD *)(v184 + 24) == 1;
                    }
                    else
                    {
                      v203 = *(_DWORD *)(v184 + 32);
                      v185 = sub_16A57B0(v184 + 24);
                      v25 = (unsigned int)(v203 - 1);
                      v186 = (_DWORD)v25 == v185;
                    }
                    if ( !v186 )
                      goto LABEL_16;
                  }
                  ++v183;
                }
                v95 = v210;
LABEL_191:
                v100 = *(_QWORD *)(v95 - 24);
                if ( !v100 )
                  goto LABEL_16;
LABEL_150:
                v101 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
                v102 = *(_BYTE **)(v10 + 24 * (1 - v101));
                if ( v102[16] == 13 )
                {
                  v103 = *((_DWORD *)v102 + 8);
                  if ( v103 <= 0x40 )
                  {
                    v104 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v103) == *((_QWORD *)v102 + 3);
                  }
                  else
                  {
                    v221 = *((_DWORD *)v102 + 8);
                    v104 = v221 == (unsigned int)sub_16A58F0((__int64)(v102 + 24));
                  }
                }
                else
                {
                  if ( *(_BYTE *)(*(_QWORD *)v102 + 8LL) != 16 )
                    goto LABEL_16;
                  v111 = sub_15A1020(v102, v23, v101, v25);
                  if ( !v111 || *(_BYTE *)(v111 + 16) != 13 )
                  {
                    v222 = v11;
                    v113 = 0;
                    v205 = *(_DWORD *)(*(_QWORD *)v102 + 32LL);
                    while ( v205 != v113 )
                    {
                      v114 = sub_15A0A60((__int64)v102, v113);
                      if ( !v114 )
                        goto LABEL_180;
                      v115 = *(_BYTE *)(v114 + 16);
                      if ( v115 != 9 )
                      {
                        if ( v115 != 13 )
                          goto LABEL_180;
                        v116 = *(_DWORD *)(v114 + 32);
                        if ( v116 <= 0x40 )
                        {
                          if ( *(_QWORD *)(v114 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v116) )
                          {
LABEL_180:
                            v11 = v222;
                            goto LABEL_16;
                          }
                        }
                        else
                        {
                          v199 = *(_DWORD *)(v114 + 32);
                          if ( v199 != (unsigned int)sub_16A58F0(v114 + 24) )
                            goto LABEL_180;
                        }
                      }
                      ++v113;
                    }
                    v11 = v222;
LABEL_154:
                    v105 = sub_15A04A0(*(_QWORD ***)v100);
                    v255.m128i_i16[0] = 259;
                    v254.m128_u64[0] = (unsigned __int64)"notmask";
                    if ( *(_BYTE *)(v105 + 16) > 0x10u || *(_BYTE *)(v100 + 16) > 0x10u )
                    {
                      v106 = (__int64)sub_170A2B0(v215, 23, (__int64 *)v105, v100, (__int64 *)&v254, 0, 0);
                    }
                    else
                    {
                      v106 = sub_15A2D50(
                               (__int64 *)v105,
                               v100,
                               0,
                               0,
                               *(double *)v12.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)a5.m128i_i64);
                      v107 = sub_14DBA30(v106, *(_QWORD *)(v215 + 96), 0);
                      if ( v107 )
                        v106 = v107;
                    }
                    if ( (unsigned __int8)(*(_BYTE *)(v106 + 16) - 35) <= 0x11u )
                    {
                      sub_15F2330(v106, 1);
                      v108 = sub_15F2370(v10);
                      sub_15F2310(v106, v108);
                    }
                    v109 = sub_1649960(v10);
                    v255.m128i_i16[0] = 261;
                    v249 = (signed __int64)v109;
                    v250 = v110;
                    v254.m128_u64[0] = (unsigned __int64)&v249;
                    v22 = sub_15FB630((__int64 *)v106, (__int64)&v254, 0);
                    if ( v22 )
                      return v22;
                    goto LABEL_16;
                  }
                  v112 = *(_DWORD *)(v111 + 32);
                  if ( v112 <= 0x40 )
                    v104 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v112) == *(_QWORD *)(v111 + 24);
                  else
                    v104 = v112 == (unsigned int)sub_16A58F0(v111 + 24);
                }
                if ( v104 )
                  goto LABEL_154;
LABEL_16:
                if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
                {
                  v249 = (signed __int64)&v238;
                  v250 = &v240;
                  v252 = &v239;
                  v251 = &v234;
                  v253 = &v241;
                  if ( (unsigned __int8)sub_17232A0((_QWORD **)&v249, v10) )
                  {
                    v254.m128_u64[0] = (unsigned __int64)&v231;
                    v254.m128_u64[1] = (unsigned __int64)&v237;
                    v255.m128i_i64[0] = (__int64)&v244;
                    v255.m128i_i64[1] = (__int64)&v235;
                    if ( (unsigned __int8)sub_171F630((__int64)&v254, v241) )
                    {
                      v246 = (unsigned __int64)&v230;
                      v247 = &v236;
                      v248 = (__int64 *)&v242;
                      if ( (unsigned __int8)sub_171F7C0((__int64)&v246, v240) )
                      {
                        if ( v236 == v237 && v230 == 32 && v231 == 32 )
                        {
                          v133 = *(_QWORD **)(v242 + 24);
                          if ( *(_DWORD *)(v242 + 32) > 0x40u )
                            v133 = (_QWORD *)*v133;
                          v134 = *(_QWORD **)(v244 + 24);
                          if ( *(_DWORD *)(v244 + 32) > 0x40u )
                            v134 = (_QWORD *)*v134;
                          if ( v134 != v133 )
                          {
                            v135 = v11[1];
                            v255.m128i_i16[0] = 257;
                            v136 = sub_17094A0(
                                     v135,
                                     v239,
                                     v241,
                                     (__int64 *)&v254,
                                     0,
                                     0,
                                     *(double *)v12.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)a5.m128i_i64);
                            v137 = v11[1];
                            v255.m128i_i16[0] = 257;
                            v138 = sub_1707C10(v137, v240, v234, (__int64)v136, (__int64 *)&v254, 0);
                            v139 = v11[1];
                            v255.m128i_i16[0] = 257;
                            v94 = sub_17094A0(
                                    v139,
                                    v238,
                                    (__int64)v138,
                                    (__int64 *)&v254,
                                    0,
                                    0,
                                    *(double *)v12.m128_u64,
                                    *(double *)v13.m128i_i64,
                                    *(double *)a5.m128i_i64);
                            goto LABEL_138;
                          }
                        }
                      }
                    }
                  }
                  v252 = &v241;
                  v250 = &v234;
                  v249 = (signed __int64)&v240;
                  if ( (unsigned __int8)sub_171F810((_QWORD **)&v249, v10) )
                  {
                    v254.m128_u64[0] = (unsigned __int64)&v231;
                    v254.m128_u64[1] = (unsigned __int64)&v237;
                    v255.m128i_i64[0] = (__int64)&v244;
                    v255.m128i_i64[1] = (__int64)&v235;
                    if ( (unsigned __int8)sub_171F630((__int64)&v254, v241) )
                    {
                      v246 = (unsigned __int64)&v230;
                      v247 = &v236;
                      v248 = (__int64 *)&v242;
                      if ( (unsigned __int8)sub_171F7C0((__int64)&v246, v240) )
                      {
                        if ( v236 == v237 && v230 == 32 && v231 == 32 )
                        {
                          v91 = *(_QWORD **)(v242 + 24);
                          if ( *(_DWORD *)(v242 + 32) > 0x40u )
                            v91 = (_QWORD *)*v91;
                          v92 = *(_QWORD **)(v244 + 24);
                          if ( *(_DWORD *)(v244 + 32) > 0x40u )
                            v92 = (_QWORD *)*v92;
                          if ( v92 != v91 )
                          {
                            v93 = v11[1];
                            v255.m128i_i16[0] = 257;
                            v94 = sub_1707C10(v93, v240, v234, v241, (__int64 *)&v254, 0);
LABEL_138:
                            v43 = (__int64)v94;
                            return sub_170E100(
                                     v11,
                                     v10,
                                     v43,
                                     v12,
                                     *(double *)v13.m128i_i64,
                                     *(double *)a5.m128i_i64,
                                     a6,
                                     v45,
                                     v46,
                                     a9,
                                     a10);
                          }
                        }
                      }
                    }
                  }
                }
                if ( !v211 )
                  return 0;
                return v10;
              }
              v178 = *(_DWORD *)(v177 + 32);
              if ( v178 <= 0x40 )
                v121 = *(_QWORD *)(v177 + 24) == 1;
              else
                v121 = v178 - 1 == (unsigned int)sub_16A57B0(v177 + 24);
            }
            if ( !v121 )
              goto LABEL_16;
            goto LABEL_191;
          }
          v126 = *(_QWORD *)(v10 - 48);
          v127 = *(_QWORD *)(v126 + 8);
          if ( !v127 || *(_QWORD *)(v127 + 8) )
            goto LABEL_16;
          v128 = *(_BYTE *)(v126 + 16);
          if ( v128 != 47 )
          {
            if ( v128 != 5 || *(_WORD *)(v126 + 18) != 23 )
              goto LABEL_16;
            v129 = *(_DWORD *)(v126 + 20) & 0xFFFFFFF;
            v130 = *(_QWORD *)(v126 - 24 * v129);
            if ( *(_BYTE *)(v130 + 16) == 13 )
            {
              v25 = *(unsigned int *)(v130 + 32);
              if ( (unsigned int)v25 <= 0x40 )
              {
                v132 = *(_QWORD *)(v130 + 24) == 1;
              }
              else
              {
                v207 = *(_DWORD *)(v130 + 32);
                v225 = *(_DWORD *)(v126 + 20) & 0xFFFFFFF;
                v131 = sub_16A57B0(v130 + 24);
                v129 = v225;
                v25 = (unsigned int)(v207 - 1);
                v132 = (_DWORD)v25 == v131;
              }
              if ( !v132 )
                goto LABEL_16;
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v130 + 8LL) != 16 )
                goto LABEL_16;
              v140 = sub_15A1020(*(_BYTE **)(v126 - 24 * v129), v23, v129, v25);
              if ( v140 && *(_BYTE *)(v140 + 16) == 13 )
              {
                v141 = *(_DWORD *)(v140 + 32);
                if ( v141 <= 0x40 )
                  v142 = *(_QWORD *)(v140 + 24) == 1;
                else
                  v142 = v141 - 1 == (unsigned int)sub_16A57B0(v140 + 24);
                if ( !v142 )
                  goto LABEL_16;
              }
              else
              {
                v208 = v126;
                v147 = 0;
                v227 = *(_DWORD *)(*(_QWORD *)v130 + 32LL);
                while ( v227 != v147 )
                {
                  v23 = v147;
                  v148 = sub_15A0A60(v130, v147);
                  if ( !v148 )
                    goto LABEL_16;
                  v25 = *(unsigned __int8 *)(v148 + 16);
                  if ( (_BYTE)v25 != 9 )
                  {
                    if ( (_BYTE)v25 != 13 )
                      goto LABEL_16;
                    v25 = *(unsigned int *)(v148 + 32);
                    if ( (unsigned int)v25 <= 0x40 )
                    {
                      v150 = *(_QWORD *)(v148 + 24) == 1;
                    }
                    else
                    {
                      v201 = *(_DWORD *)(v148 + 32);
                      v149 = sub_16A57B0(v148 + 24);
                      v25 = (unsigned int)(v201 - 1);
                      v150 = (_DWORD)v25 == v149;
                    }
                    if ( !v150 )
                      goto LABEL_16;
                  }
                  ++v147;
                }
                v126 = v208;
              }
              v129 = *(_DWORD *)(v126 + 20) & 0xFFFFFFF;
            }
            v100 = *(_QWORD *)(v126 + 24 * (1 - v129));
            if ( !v100 )
              goto LABEL_16;
LABEL_219:
            if ( (unsigned __int8)sub_17198D0(*(_BYTE **)(v10 - 24), v23, v129, v25) )
              goto LABEL_154;
            goto LABEL_16;
          }
          v143 = *(__int64 **)(v126 - 48);
          v144 = *((_BYTE *)v143 + 16);
          if ( v144 == 13 )
          {
            v129 = *((unsigned int *)v143 + 8);
            if ( (unsigned int)v129 <= 0x40 )
            {
              v146 = v143[3] == 1;
            }
            else
            {
              v226 = *((_DWORD *)v143 + 8);
              v145 = sub_16A57B0((__int64)(v143 + 3));
              v129 = (unsigned int)(v226 - 1);
              v146 = (_DWORD)v129 == v145;
            }
          }
          else
          {
            if ( *(_BYTE *)(*v143 + 8) != 16 || v144 > 0x10u )
              goto LABEL_16;
            v151 = sub_15A1020(*(_BYTE **)(v126 - 48), v23, v24, *v143);
            if ( !v151 || *(_BYTE *)(v151 + 16) != 13 )
            {
              v129 = 0;
              v209 = v126;
              v179 = 0;
              v228 = *(_DWORD *)(*v143 + 32);
              while ( v228 != v179 )
              {
                v23 = v179;
                v180 = sub_15A0A60((__int64)v143, v179);
                if ( !v180 )
                  goto LABEL_16;
                v25 = *(unsigned __int8 *)(v180 + 16);
                if ( (_BYTE)v25 != 9 )
                {
                  if ( (_BYTE)v25 != 13 )
                    goto LABEL_16;
                  v25 = *(unsigned int *)(v180 + 32);
                  if ( (unsigned int)v25 <= 0x40 )
                  {
                    v182 = *(_QWORD *)(v180 + 24) == 1;
                  }
                  else
                  {
                    v202 = *(_DWORD *)(v180 + 32);
                    v181 = sub_16A57B0(v180 + 24);
                    v25 = (unsigned int)(v202 - 1);
                    v182 = (_DWORD)v25 == v181;
                  }
                  if ( !v182 )
                    goto LABEL_16;
                }
                ++v179;
              }
              v126 = v209;
LABEL_243:
              v100 = *(_QWORD *)(v126 - 24);
              if ( !v100 )
                goto LABEL_16;
              goto LABEL_219;
            }
            v152 = *(_DWORD *)(v151 + 32);
            if ( v152 <= 0x40 )
            {
              v146 = *(_QWORD *)(v151 + 24) == 1;
            }
            else
            {
              v153 = sub_16A57B0(v151 + 24);
              v129 = v152 - 1;
              v146 = (_DWORD)v129 == v153;
            }
          }
          if ( !v146 )
            goto LABEL_16;
          goto LABEL_243;
        }
      }
      if ( (unsigned int)sub_171CA60(v11, (__int64)v156, v157, v10) == 2 )
      {
        v170 = v11[1];
        v255.m128i_i16[0] = 259;
        v254.m128_u64[0] = (unsigned __int64)"addconv";
        v171 = *(_QWORD *)(v30 - 24);
        v172 = *(_QWORD *)(v31 - 24);
        goto LABEL_290;
      }
LABEL_82:
      if ( *(_BYTE *)(v31 + 16) != 61 )
        goto LABEL_83;
      if ( *(_BYTE *)(v30 + 16) == 13 )
      {
        v176 = *(_QWORD *)(v31 + 8);
        if ( !v176 || *(_QWORD *)(v176 + 8) )
          goto LABEL_83;
        v218 = (__int64 *)sub_15A43B0(v30, **(__int64 ****)(v31 - 24), 0);
        if ( v30 == sub_15A3CB0((unsigned __int64)v218, v219, 0)
          && (unsigned int)sub_171CA30(v11, *(__int64 **)(v31 - 24), v218, v10) == 2 )
        {
          v162 = v11[1];
          v255.m128i_i16[0] = 259;
          v163 = (__int64)v218;
          v254.m128_u64[0] = (unsigned __int64)"addconv";
          v164 = *(_QWORD *)(v31 - 24);
LABEL_280:
          v165 = sub_17094A0(
                   v162,
                   v164,
                   v163,
                   (__int64 *)&v254,
                   1u,
                   0,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          v255.m128i_i16[0] = 257;
          v166 = (__int64)v165;
          v167 = sub_1648A60(56, 1u);
          v10 = (__int64)v167;
          if ( v167 )
            sub_15FC690((__int64)v167, v166, (__int64)v219, (__int64)&v254, 0);
          return v10;
        }
      }
      goto LABEL_272;
    }
    v70 = *(_QWORD *)(v31 + 8);
LABEL_117:
    if ( !v70 )
      goto LABEL_83;
    if ( *(_QWORD *)(v70 + 8) )
      goto LABEL_82;
    v216 = (_QWORD *)v30;
    v214 = v31;
    goto LABEL_111;
  }
  return sub_170E100(
           a1->m128i_i64,
           a2,
           (__int64)v27,
           v12,
           *(double *)v13.m128i_i64,
           *(double *)a5.m128i_i64,
           a6,
           v28,
           v29,
           a9,
           a10);
}
