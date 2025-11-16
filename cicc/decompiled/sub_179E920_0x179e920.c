// Function: sub_179E920
// Address: 0x179e920
//
__int64 __fastcall sub_179E920(
        __int64 *a1,
        __int64 a2,
        char *a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _DWORD *v12; // r11
  char v17; // al
  _QWORD *v18; // rsi
  __int64 v19; // r14
  char v20; // al
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 v25; // rbx
  _QWORD *v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rsi
  double v30; // xmm4_8
  double v31; // xmm5_8
  unsigned __int8 v33; // al
  _DWORD *v34; // r11
  __int64 v35; // rdx
  int v36; // edx
  unsigned int v37; // eax
  _BYTE *v38; // rdi
  unsigned __int8 v39; // al
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdx
  int v43; // esi
  __int64 v44; // r14
  __int64 *v45; // r12
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // r11
  __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // r14
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rax
  _BYTE *v55; // rdi
  unsigned __int8 v56; // al
  __int64 v57; // rax
  __int64 v58; // rdi
  int v59; // esi
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 *v62; // r12
  __int64 v63; // rax
  __int64 v64; // rbx
  _QWORD *v65; // rax
  _QWORD *v66; // r13
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r14
  int v77; // edx
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // r15
  const char *v81; // rax
  int v82; // esi
  __int64 v83; // rdx
  unsigned int v84; // r14d
  unsigned int v85; // eax
  _DWORD *v86; // r11
  unsigned int v87; // r15d
  unsigned __int64 v88; // rdx
  _QWORD *v89; // rsi
  unsigned int v90; // eax
  __int64 v91; // r15
  __int64 v92; // rdx
  __int64 *v93; // rax
  __int64 v94; // rax
  unsigned __int8 *v95; // rax
  __int64 **v96; // r12
  __int64 v97; // r13
  _QWORD *v98; // rax
  __int64 v99; // rax
  _BYTE *v100; // rdi
  unsigned __int8 v101; // al
  __int64 v102; // r11
  __int64 v103; // rax
  __int64 v104; // r13
  __int64 v105; // rax
  __int64 v106; // rdi
  __int64 v107; // r12
  __int64 v108; // rax
  __int64 *v109; // rbx
  _QWORD *v110; // rax
  _QWORD *v111; // r13
  __int64 v112; // rdx
  unsigned __int64 v113; // rcx
  __int64 v114; // rdx
  __int64 v115; // rdx
  unsigned __int64 v116; // rcx
  __int64 v117; // rdx
  __int64 v118; // rdx
  unsigned __int64 v119; // rcx
  __int64 v120; // rdx
  __int64 v121; // rax
  int v122; // eax
  __int64 v123; // r12
  const char *v124; // rax
  __int64 v125; // rsi
  __int64 v126; // rdx
  unsigned __int8 *v127; // rax
  __int64 v128; // rbx
  __int64 v129; // r12
  __int64 v130; // rdx
  __int64 v131; // rax
  __int64 *v132; // rax
  int v133; // edi
  __int64 v134; // rax
  __int64 v135; // rax
  unsigned __int8 *v136; // rax
  __int64 v137; // rdx
  __int64 v138; // rax
  __int64 v139; // rbx
  __int64 *v140; // r12
  __int64 v141; // rax
  __int64 v142; // r15
  __int64 v143; // r14
  __int64 v144; // rax
  __int64 v145; // rsi
  __int64 v146; // rax
  __int64 v147; // rax
  char v148; // al
  __int64 v149; // r12
  const char *v150; // rax
  __int64 v151; // rsi
  __int64 v152; // rdx
  unsigned __int8 *v153; // rax
  __int64 v154; // rbx
  __int64 *v155; // r12
  __int64 v156; // rdx
  __int64 v157; // rax
  unsigned __int8 *v158; // rax
  int v159; // edi
  __int64 v160; // rax
  char v161; // al
  unsigned __int8 **v162; // rdx
  __int64 v163; // r14
  const char *v164; // rax
  __int64 v165; // rsi
  __int64 v166; // rdx
  unsigned __int8 *v167; // rax
  __int64 v168; // r14
  __int64 v169; // r15
  const char *v170; // rax
  __int64 v171; // rdx
  __int64 *v172; // r13
  unsigned int v173; // esi
  int v174; // edx
  __int64 *v175; // rax
  __int64 v176; // rdx
  __int64 v177; // rdx
  char *v178; // rax
  unsigned __int64 v179; // rdx
  __int64 v180; // r14
  const char *v181; // rax
  __int64 v182; // rsi
  __int64 v183; // rdx
  unsigned __int8 *v184; // rax
  __int64 v185; // r14
  __int64 v186; // r15
  const char *v187; // rax
  __int64 v188; // rdx
  _DWORD *v189; // [rsp+0h] [rbp-A0h]
  _DWORD *v190; // [rsp+0h] [rbp-A0h]
  _DWORD *v191; // [rsp+0h] [rbp-A0h]
  unsigned int v192; // [rsp+8h] [rbp-98h]
  __int64 v193; // [rsp+8h] [rbp-98h]
  __int64 v194; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v195; // [rsp+8h] [rbp-98h]
  __int64 v196; // [rsp+8h] [rbp-98h]
  _DWORD *v197; // [rsp+10h] [rbp-90h]
  _DWORD *v198; // [rsp+10h] [rbp-90h]
  __int64 v199; // [rsp+10h] [rbp-90h]
  char v200; // [rsp+18h] [rbp-88h]
  __int64 v201; // [rsp+18h] [rbp-88h]
  __int64 v202; // [rsp+18h] [rbp-88h]
  __int64 v203; // [rsp+18h] [rbp-88h]
  __int64 v204; // [rsp+18h] [rbp-88h]
  __int64 v205; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v206; // [rsp+20h] [rbp-80h] BYREF
  char *v207; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v208; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v209; // [rsp+38h] [rbp-68h]
  const char *v210; // [rsp+40h] [rbp-60h] BYREF
  __int64 v211; // [rsp+48h] [rbp-58h]
  const char **v212; // [rsp+50h] [rbp-50h] BYREF
  char *v213; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v214; // [rsp+60h] [rbp-40h]

  v12 = a3 + 24;
  v17 = *(_BYTE *)(a4 + 16);
  v200 = v17;
  if ( a3[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 16 )
      return 0;
    v46 = sub_15A1020(a3, a2, (__int64)a3, a4);
    if ( !v46 || *(_BYTE *)(v46 + 16) != 13 )
      return 0;
    v12 = (_DWORD *)(v46 + 24);
    v17 = *(_BYTE *)(a4 + 16);
  }
  if ( v17 != 49 )
  {
    v18 = *(_QWORD **)v12;
    v19 = a4;
    if ( v12[2] > 0x40u )
      v18 = (_QWORD *)*v18;
    v197 = v12;
    v20 = sub_179D180(a2, (__int64)v18, v200 == 47, a1, a4);
    v12 = v197;
    if ( v20 )
    {
      v21 = *(_QWORD **)v197;
      if ( v197[2] > 0x40u )
        v21 = (_QWORD *)*v21;
      v22 = sub_179DF30(a2, (unsigned int)v21, v200 == 47, a1, a1[333], *(double *)a5.m128_u64, a6, a7);
      v23 = *(_QWORD *)(a4 + 8);
      v24 = v22;
      if ( v23 )
      {
        v25 = *a1;
        do
        {
          v26 = sub_1648700(v23);
          sub_170B990(v25, (__int64)v26);
          v23 = *(_QWORD *)(v23 + 8);
        }
        while ( v23 );
        if ( a4 == v24 )
          v24 = sub_1599EF0(*(__int64 ***)a4);
        sub_164D160(a4, v24, a5, a6, a7, a8, v27, v28, a11, a12);
        return v19;
      }
      return 0;
    }
  }
  v198 = v12;
  v29 = a4;
  v192 = sub_16431D0(*(_QWORD *)a2);
  v19 = sub_1713A90(a1, (_BYTE *)a4, a5, a6, a7, a8, v30, v31, a11, a12);
  if ( v19 )
    return v19;
  v33 = *(_BYTE *)(a2 + 16);
  v34 = v198;
  if ( v33 == 60 )
  {
    v76 = *(_QWORD *)(a2 - 24);
    v77 = *(unsigned __int8 *)(v76 + 16);
    if ( (unsigned __int8)v77 > 0x17u
      && (unsigned __int8)(*(_BYTE *)(a4 + 16) - 47) <= 1u
      && (unsigned int)(v77 - 47) <= 2 )
    {
      v78 = (*(_BYTE *)(v76 + 23) & 0x40) != 0 ? *(_QWORD *)(v76 - 8) : v76 - 24LL * (*(_DWORD *)(v76 + 20) & 0xFFFFFFF);
      if ( *(_BYTE *)(*(_QWORD *)(v78 + 24) + 16LL) == 13 )
      {
        v79 = sub_15A3CB0((unsigned __int64)a3, *(__int64 ***)v76, 0);
        v80 = a1[1];
        v194 = v79;
        v81 = sub_1649960(a4);
        v82 = *(unsigned __int8 *)(a4 + 16);
        v210 = v81;
        v211 = v83;
        LOWORD(v214) = 261;
        v212 = &v210;
        v195 = (unsigned __int8 *)sub_17066B0(
                                    v80,
                                    v82 - 24,
                                    v76,
                                    v194,
                                    (__int64 *)&v212,
                                    0,
                                    *(double *)a5.m128_u64,
                                    a6,
                                    a7);
        v84 = sub_16431D0(*(_QWORD *)v76);
        v85 = sub_16431D0(*(_QWORD *)a2);
        v209 = v84;
        v86 = v198;
        v87 = v85;
        if ( v84 > 0x40 )
        {
          sub_16A4EF0((__int64)&v208, 0, 0);
          v86 = v198;
        }
        else
        {
          v208 = 0;
        }
        if ( v87 )
        {
          if ( v87 > 0x40 )
          {
            v191 = v86;
            sub_16A5260(&v208, 0, v87);
            v86 = v191;
          }
          else
          {
            v88 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v87);
            if ( v209 > 0x40 )
              *(_QWORD *)v208 |= v88;
            else
              v208 |= v88;
          }
        }
        v89 = *(_QWORD **)v86;
        v90 = v86[2];
        if ( *(_BYTE *)(a4 + 16) == 47 )
        {
          if ( v90 > 0x40 )
            v89 = (_QWORD *)*v89;
          if ( v209 > 0x40 )
          {
            sub_16A7DC0((__int64 *)&v208, (unsigned int)v89);
          }
          else
          {
            if ( (_DWORD)v89 == v209 )
              v179 = 0;
            else
              v179 = v208 << (char)v89;
            v208 = v179 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v209);
          }
        }
        else
        {
          if ( v90 > 0x40 )
            v89 = (_QWORD *)*v89;
          if ( v209 > 0x40 )
          {
            sub_16A8110((__int64)&v208, (unsigned int)v89);
          }
          else if ( (_DWORD)v89 == v209 )
          {
            v208 = 0;
          }
          else
          {
            v208 >>= (char)v89;
          }
        }
        v91 = a1[1];
        v210 = sub_1649960(a2);
        LOWORD(v214) = 261;
        v211 = v92;
        v212 = &v210;
        v93 = (__int64 *)sub_16498A0(a4);
        v94 = sub_159C0E0(v93, (__int64)&v208);
        v95 = sub_1729500(v91, v195, v94, (__int64 *)&v212, *(double *)a5.m128_u64, a6, a7);
        v96 = *(__int64 ***)a4;
        v97 = (__int64)v95;
        LOWORD(v214) = 257;
        v98 = sub_1648A60(56, 1u);
        v19 = (__int64)v98;
        if ( v98 )
          sub_15FC510((__int64)v98, v97, (__int64)v96, (__int64)&v212, 0);
        if ( v209 > 0x40 && v208 )
          j_j___libc_free_0_0(v208);
        return v19;
      }
    }
  }
  v35 = *(_QWORD *)(a2 + 8);
  if ( !v35 || *(_QWORD *)(v35 + 8) || v33 <= 0x17u )
    return 0;
  v36 = v33;
  if ( (unsigned int)v33 - 35 > 0x11 )
    goto LABEL_35;
  v37 = v33 - 24;
  if ( v36 == 37 )
  {
    if ( v200 != 47 )
      goto LABEL_27;
  }
  else
  {
    if ( v37 > 0xD )
    {
      if ( (unsigned int)(v36 - 50) > 2 )
        goto LABEL_27;
    }
    else if ( v37 != 11 )
    {
      goto LABEL_27;
    }
    if ( v200 != 47 )
      goto LABEL_27;
    v145 = *(_QWORD *)(a2 - 24);
    v146 = *(_QWORD *)(v145 + 8);
    if ( v146 && !*(_QWORD *)(v146 + 8) )
    {
      v212 = (const char **)&v206;
      v213 = a3;
      v161 = sub_179D760((__int64)&v212, v145);
      v34 = v198;
      if ( v161 )
      {
        v180 = a1[1];
        v181 = sub_1649960(a2);
        v182 = *(_QWORD *)(a2 - 48);
        v210 = v181;
        v211 = v183;
        LOWORD(v214) = 261;
        v212 = &v210;
        v184 = sub_173DC60(v180, v182, (__int64)a3, (__int64 *)&v212, 0, 0, *(double *)a5.m128_u64, a6, a7);
        v185 = a1[1];
        v186 = (__int64)v184;
        v187 = sub_1649960(*(_QWORD *)(a2 - 24));
        LODWORD(v182) = *(unsigned __int8 *)(a2 + 16);
        v211 = v188;
        v210 = v187;
        LOWORD(v214) = 261;
        v212 = &v210;
        v172 = (__int64 *)sub_17066B0(
                            v185,
                            (int)v182 - 24,
                            v186,
                            (__int64)v206,
                            (__int64 *)&v212,
                            0,
                            *(double *)a5.m128_u64,
                            a6,
                            a7);
        v173 = v192;
        v174 = v192 - sub_179D670(v198, v192);
        goto LABEL_171;
      }
      v145 = *(_QWORD *)(a2 - 24);
    }
    v147 = *(_QWORD *)(v145 + 8);
    if ( v147 )
    {
      if ( !*(_QWORD *)(v147 + 8) )
      {
        v189 = v34;
        v212 = (const char **)&v206;
        v214 = &v208;
        v213 = a3;
        v148 = sub_179D860((__int64)&v212, v145);
        v34 = v189;
        if ( v148 )
        {
          v149 = a1[1];
          v150 = sub_1649960(a2);
          v151 = *(_QWORD *)(a2 - 48);
          v211 = v152;
          v212 = &v210;
          v210 = v150;
          LOWORD(v214) = 261;
          v153 = sub_173DC60(v149, v151, (__int64)a3, (__int64 *)&v212, 0, 0, *(double *)a5.m128_u64, a6, a7);
          v154 = a1[1];
          v155 = (__int64 *)v153;
          v210 = sub_1649960((__int64)v206);
          v211 = v156;
          v212 = &v210;
          LOWORD(v214) = 773;
          v213 = ".mask";
          v157 = sub_15A2D50((__int64 *)v208, (__int64)a3, 0, 0, *(double *)a5.m128_u64, a6, a7);
          v158 = sub_1729500(v154, v206, v157, (__int64 *)&v212, *(double *)a5.m128_u64, a6, a7);
          v159 = *(unsigned __int8 *)(a2 + 16);
          LOWORD(v214) = 257;
          return sub_15FB440(v159 - 24, v155, (__int64)v158, (__int64)&v212, 0);
        }
      }
    }
  }
  v29 = *(_QWORD *)(a2 - 48);
  v121 = *(_QWORD *)(v29 + 8);
  if ( !v121 || *(_QWORD *)(v121 + 8) )
  {
LABEL_27:
    v38 = *(_BYTE **)(a2 - 24);
    v39 = v38[16];
    if ( v39 == 13
      || *(_BYTE *)(*(_QWORD *)v38 + 8LL) == 16
      && v39 <= 0x10u
      && (v134 = sub_15A1020(v38, v29, *(_QWORD *)v38, (__int64)(v38 + 24))) != 0
      && *(_BYTE *)(v134 + 16) == 13 )
    {
      v29 = a4;
      if ( sub_179D6D0((__int64)a1, a4, a2) )
      {
        v40 = sub_15A2A30(
                (__int64 *)((unsigned int)*(unsigned __int8 *)(a4 + 16) - 24),
                *(__int64 **)(a2 - 24),
                (__int64)a3,
                0,
                0,
                *(double *)a5.m128_u64,
                a6,
                a7);
        v41 = a1[1];
        v42 = *(_QWORD *)(a2 - 48);
        v43 = *(unsigned __int8 *)(a4 + 16);
        v44 = v40;
        LOWORD(v214) = 257;
        v45 = (__int64 *)sub_17066B0(
                           v41,
                           v43 - 24,
                           v42,
                           (__int64)a3,
                           (__int64 *)&v212,
                           0,
                           *(double *)a5.m128_u64,
                           a6,
                           a7);
        sub_164B7C0((__int64)v45, a2);
        LODWORD(v41) = *(unsigned __int8 *)(a2 + 16);
        LOWORD(v214) = 257;
        return sub_15FB440((int)v41 - 24, v45, v44, (__int64)&v212, 0);
      }
    }
    v33 = *(_BYTE *)(a2 + 16);
    if ( v200 == 47 && v33 == 37 )
    {
      v136 = *(unsigned __int8 **)(a2 - 48);
      v137 = v136[16];
      if ( (_BYTE)v137 == 13 )
        goto LABEL_138;
      if ( *(_BYTE *)(*(_QWORD *)v136 + 8LL) != 16 || (unsigned __int8)v137 > 0x10u )
        return 0;
      v160 = sub_15A1020(v136, v29, v137, *(_QWORD *)v136);
      if ( v160 && *(_BYTE *)(v160 + 16) == 13 )
      {
        v136 = *(unsigned __int8 **)(a2 - 48);
LABEL_138:
        v138 = sub_15A2A30(
                 (__int64 *)((unsigned int)*(unsigned __int8 *)(a4 + 16) - 24),
                 (__int64 *)v136,
                 (__int64)a3,
                 0,
                 0,
                 *(double *)a5.m128_u64,
                 a6,
                 a7);
        v139 = a1[1];
        v140 = (__int64 *)v138;
        v141 = *(_QWORD *)(a2 - 24);
        LOWORD(v214) = 257;
        if ( *(_BYTE *)(v141 + 16) > 0x10u || (unsigned __int8)a3[16] > 0x10u )
        {
          v143 = (__int64)sub_179D030(v139, (__int64 *)v141, (__int64)a3, (__int64 *)&v212, 0, 0);
        }
        else
        {
          v142 = sub_15A2D50((__int64 *)v141, (__int64)a3, 0, 0, *(double *)a5.m128_u64, a6, a7);
          v143 = sub_14DBA30(v142, *(_QWORD *)(v139 + 96), 0);
          if ( !v143 )
            v143 = v142;
        }
        sub_164B7C0(v143, a2);
        LOWORD(v214) = 257;
        return sub_15FB440(13, v140, v143, (__int64)&v212, 0);
      }
      v33 = *(_BYTE *)(a2 + 16);
    }
    if ( v33 <= 0x17u )
      return 0;
LABEL_35:
    if ( v33 != 79 )
      return 0;
    v47 = *(_QWORD *)(a2 - 72);
    if ( !v47 )
      return 0;
    v48 = *(_QWORD *)(a2 - 48);
    v49 = *(_QWORD *)(v48 + 8);
    if ( v49 )
    {
      if ( !*(_QWORD *)(v49 + 8) && (unsigned __int8)(*(_BYTE *)(v48 + 16) - 35) <= 0x11u )
      {
        v50 = *(_QWORD *)(a2 - 24);
        if ( v50 )
        {
          if ( *(_BYTE *)(v50 + 16) > 0x10u )
          {
            v99 = *(_QWORD *)(v48 - 48);
            if ( v99 == v50 )
            {
              if ( v99 )
              {
                v100 = *(_BYTE **)(v48 - 24);
                v101 = v100[16];
                if ( v101 == 13 )
                  goto LABEL_97;
                if ( *(_BYTE *)(*(_QWORD *)v100 + 8LL) == 16 && v101 <= 0x10u )
                {
                  v199 = *(_QWORD *)(a2 - 72);
                  v205 = *(_QWORD *)(a2 - 48);
                  v144 = sub_15A1020(v100, v29, *(_QWORD *)v100, (__int64)(v100 + 24));
                  v48 = v205;
                  v47 = v199;
                  if ( !v144 || *(_BYTE *)(v144 + 16) != 13 )
                  {
LABEL_155:
                    if ( *(_BYTE *)(a2 + 16) != 79 )
                      return 0;
                    v47 = *(_QWORD *)(a2 - 72);
                    goto LABEL_42;
                  }
LABEL_97:
                  v29 = a4;
                  v202 = v47;
                  if ( sub_179D6D0((__int64)a1, a4, v48) )
                  {
                    v196 = v202;
                    v203 = v102;
                    v103 = sub_15A2A30(
                             (__int64 *)((unsigned int)*(unsigned __int8 *)(a4 + 16) - 24),
                             *(__int64 **)(v102 - 24),
                             (__int64)a3,
                             0,
                             0,
                             *(double *)a5.m128_u64,
                             a6,
                             a7);
                    LOWORD(v214) = 257;
                    v104 = v103;
                    v105 = sub_17066B0(
                             a1[1],
                             (unsigned int)*(unsigned __int8 *)(a4 + 16) - 24,
                             v50,
                             (__int64)a3,
                             (__int64 *)&v212,
                             0,
                             *(double *)a5.m128_u64,
                             a6,
                             a7);
                    v106 = a1[1];
                    v107 = v105;
                    LOWORD(v214) = 257;
                    v108 = sub_17066B0(
                             v106,
                             (unsigned int)*(unsigned __int8 *)(v203 + 16) - 24,
                             v105,
                             v104,
                             (__int64 *)&v212,
                             0,
                             *(double *)a5.m128_u64,
                             a6,
                             a7);
                    LOWORD(v214) = 257;
                    v109 = (__int64 *)v108;
                    v110 = sub_1648A60(56, 3u);
                    v19 = (__int64)v110;
                    if ( !v110 )
                      return v19;
                    v111 = v110 - 9;
                    sub_15F1EA0((__int64)v110, *v109, 55, (__int64)(v110 - 9), 3, 0);
                    if ( *(_QWORD *)(v19 - 72) )
                    {
                      v112 = *(_QWORD *)(v19 - 64);
                      v113 = *(_QWORD *)(v19 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v113 = v112;
                      if ( v112 )
                        *(_QWORD *)(v112 + 16) = v113 | *(_QWORD *)(v112 + 16) & 3LL;
                    }
                    *(_QWORD *)(v19 - 72) = v196;
                    v114 = *(_QWORD *)(v196 + 8);
                    *(_QWORD *)(v19 - 64) = v114;
                    if ( v114 )
                      *(_QWORD *)(v114 + 16) = (v19 - 64) | *(_QWORD *)(v114 + 16) & 3LL;
                    *(_QWORD *)(v19 - 56) = (v196 + 8) | *(_QWORD *)(v19 - 56) & 3LL;
                    *(_QWORD *)(v196 + 8) = v111;
                    if ( *(_QWORD *)(v19 - 48) )
                    {
                      v115 = *(_QWORD *)(v19 - 40);
                      v116 = *(_QWORD *)(v19 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v116 = v115;
                      if ( v115 )
                        *(_QWORD *)(v115 + 16) = v116 | *(_QWORD *)(v115 + 16) & 3LL;
                    }
                    *(_QWORD *)(v19 - 48) = v109;
                    v117 = v109[1];
                    *(_QWORD *)(v19 - 40) = v117;
                    if ( v117 )
                      *(_QWORD *)(v117 + 16) = (v19 - 40) | *(_QWORD *)(v117 + 16) & 3LL;
                    *(_QWORD *)(v19 - 32) = *(_QWORD *)(v19 - 32) & 3LL | (unsigned __int64)(v109 + 1);
                    v109[1] = v19 - 48;
                    if ( *(_QWORD *)(v19 - 24) )
                    {
                      v118 = *(_QWORD *)(v19 - 16);
                      v119 = *(_QWORD *)(v19 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v119 = v118;
                      if ( v118 )
                        *(_QWORD *)(v118 + 16) = v119 | *(_QWORD *)(v118 + 16) & 3LL;
                    }
                    *(_QWORD *)(v19 - 24) = v107;
                    if ( v107 )
                    {
                      v120 = *(_QWORD *)(v107 + 8);
                      *(_QWORD *)(v19 - 16) = v120;
                      if ( v120 )
                        *(_QWORD *)(v120 + 16) = (v19 - 16) | *(_QWORD *)(v120 + 16) & 3LL;
                      *(_QWORD *)(v19 - 8) = (v107 + 8) | *(_QWORD *)(v19 - 8) & 3LL;
                      *(_QWORD *)(v107 + 8) = v19 - 24;
                    }
                    goto LABEL_70;
                  }
                  goto LABEL_155;
                }
              }
            }
          }
        }
      }
    }
LABEL_42:
    if ( !v47 )
      return 0;
    v51 = *(_QWORD *)(a2 - 48);
    if ( !v51 )
      return 0;
    v52 = *(_QWORD *)(a2 - 24);
    v53 = *(_QWORD *)(v52 + 8);
    if ( !v53 )
      return 0;
    if ( *(_QWORD *)(v53 + 8) )
      return 0;
    if ( (unsigned __int8)(*(_BYTE *)(v52 + 16) - 35) > 0x11u )
      return 0;
    if ( *(_BYTE *)(v51 + 16) <= 0x10u )
      return 0;
    v54 = *(_QWORD *)(v52 - 48);
    if ( !v54 || v54 != v51 )
      return 0;
    v55 = *(_BYTE **)(v52 - 24);
    v56 = v55[16];
    if ( v56 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) != 16 )
        return 0;
      if ( v56 > 0x10u )
        return 0;
      v204 = v47;
      v135 = sub_15A1020(v55, v29, *(_QWORD *)v55, (__int64)(v55 + 24));
      if ( !v135 || *(_BYTE *)(v135 + 16) != 13 )
        return 0;
      v47 = v204;
    }
    v201 = v47;
    if ( !sub_179D6D0((__int64)a1, a4, v52) )
      return 0;
    v57 = sub_15A2A30(
            (__int64 *)((unsigned int)*(unsigned __int8 *)(a4 + 16) - 24),
            *(__int64 **)(v52 - 24),
            (__int64)a3,
            0,
            0,
            *(double *)a5.m128_u64,
            a6,
            a7);
    v58 = a1[1];
    v59 = *(unsigned __int8 *)(a4 + 16);
    v193 = v57;
    LOWORD(v214) = 257;
    v60 = sub_17066B0(v58, v59 - 24, v51, (__int64)a3, (__int64 *)&v212, 0, *(double *)a5.m128_u64, a6, a7);
    v61 = a1[1];
    LOWORD(v214) = 257;
    v62 = (__int64 *)v60;
    v63 = sub_17066B0(
            v61,
            (unsigned int)*(unsigned __int8 *)(v52 + 16) - 24,
            v60,
            v193,
            (__int64 *)&v212,
            0,
            *(double *)a5.m128_u64,
            a6,
            a7);
    LOWORD(v214) = 257;
    v64 = v63;
    v65 = sub_1648A60(56, 3u);
    v19 = (__int64)v65;
    if ( !v65 )
      return v19;
    v66 = v65 - 9;
    sub_15F1EA0((__int64)v65, *v62, 55, (__int64)(v65 - 9), 3, 0);
    if ( *(_QWORD *)(v19 - 72) )
    {
      v67 = *(_QWORD *)(v19 - 64);
      v68 = *(_QWORD *)(v19 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v68 = v67;
      if ( v67 )
        *(_QWORD *)(v67 + 16) = *(_QWORD *)(v67 + 16) & 3LL | v68;
    }
    *(_QWORD *)(v19 - 72) = v201;
    v69 = *(_QWORD *)(v201 + 8);
    *(_QWORD *)(v19 - 64) = v69;
    if ( v69 )
      *(_QWORD *)(v69 + 16) = (v19 - 64) | *(_QWORD *)(v69 + 16) & 3LL;
    *(_QWORD *)(v19 - 56) = *(_QWORD *)(v19 - 56) & 3LL | (v201 + 8);
    *(_QWORD *)(v201 + 8) = v66;
    if ( *(_QWORD *)(v19 - 48) )
    {
      v70 = *(_QWORD *)(v19 - 40);
      v71 = *(_QWORD *)(v19 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v71 = v70;
      if ( v70 )
        *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
    }
    *(_QWORD *)(v19 - 48) = v62;
    v72 = v62[1];
    *(_QWORD *)(v19 - 40) = v72;
    if ( v72 )
      *(_QWORD *)(v72 + 16) = (v19 - 40) | *(_QWORD *)(v72 + 16) & 3LL;
    *(_QWORD *)(v19 - 32) = (unsigned __int64)(v62 + 1) | *(_QWORD *)(v19 - 32) & 3LL;
    v62[1] = v19 - 48;
    if ( *(_QWORD *)(v19 - 24) )
    {
      v73 = *(_QWORD *)(v19 - 16);
      v74 = *(_QWORD *)(v19 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v74 = v73;
      if ( v73 )
        *(_QWORD *)(v73 + 16) = *(_QWORD *)(v73 + 16) & 3LL | v74;
    }
    *(_QWORD *)(v19 - 24) = v64;
    if ( v64 )
    {
      v75 = *(_QWORD *)(v64 + 8);
      *(_QWORD *)(v19 - 16) = v75;
      if ( v75 )
        *(_QWORD *)(v75 + 16) = (v19 - 16) | *(_QWORD *)(v75 + 16) & 3LL;
      *(_QWORD *)(v19 - 8) = *(_QWORD *)(v19 - 8) & 3LL | (v64 + 8);
      *(_QWORD *)(v64 + 8) = v19 - 24;
    }
LABEL_70:
    sub_164B780(v19, (__int64 *)&v212);
    return v19;
  }
  v122 = *(unsigned __int8 *)(v29 + 16);
  if ( (unsigned __int8)v122 <= 0x17u )
  {
    if ( (_BYTE)v122 != 5
      || (unsigned int)*(unsigned __int16 *)(v29 + 18) - 24 > 1
      || (v177 = *(_DWORD *)(v29 + 20) & 0xFFFFFFF, !*(_QWORD *)(v29 - 24 * v177))
      || (v206 = *(unsigned __int8 **)(v29 - 24 * v177), v178 = *(char **)(v29 + 24 * (1 - v177)), a3 != v178)
      || !v178 )
    {
LABEL_123:
      v212 = (const char **)&v206;
      v213 = (char *)&v207;
      v214 = &v208;
      if ( (unsigned __int8)sub_179DAF0(&v212, v29) && v207 == a3 )
      {
        v123 = a1[1];
        v124 = sub_1649960(a2);
        v125 = *(_QWORD *)(a2 - 24);
        v211 = v126;
        v212 = &v210;
        v210 = v124;
        LOWORD(v214) = 261;
        v127 = sub_173DC60(v123, v125, (__int64)a3, (__int64 *)&v212, 0, 0, *(double *)a5.m128_u64, a6, a7);
        v128 = a1[1];
        v129 = (__int64)v127;
        v210 = sub_1649960((__int64)v206);
        v211 = v130;
        v212 = &v210;
        LOWORD(v214) = 773;
        v213 = ".mask";
        v131 = sub_15A2D50((__int64 *)v208, (__int64)a3, 0, 0, *(double *)a5.m128_u64, a6, a7);
        v132 = (__int64 *)sub_1729500(v128, v206, v131, (__int64 *)&v212, *(double *)a5.m128_u64, a6, a7);
        v133 = *(unsigned __int8 *)(a2 + 16);
        LOWORD(v214) = 257;
        return sub_15FB440(v133 - 24, v132, v129, (__int64)&v212, 0);
      }
      goto LABEL_27;
    }
  }
  else
  {
    if ( (unsigned int)(v122 - 48) > 1 )
      goto LABEL_123;
    if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
    {
      if ( !**(_QWORD **)(v29 - 8) )
        goto LABEL_123;
      v206 = **(unsigned __int8 ***)(v29 - 8);
      v162 = *(unsigned __int8 ***)(v29 - 8);
    }
    else
    {
      v162 = (unsigned __int8 **)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      if ( !*v162 )
        goto LABEL_123;
      v206 = *v162;
    }
    if ( a3 != (char *)v162[3] )
      goto LABEL_123;
  }
  v163 = a1[1];
  v190 = v34;
  v164 = sub_1649960(a2);
  v165 = *(_QWORD *)(a2 - 24);
  v210 = v164;
  v211 = v166;
  LOWORD(v214) = 261;
  v212 = &v210;
  v167 = sub_173DC60(v163, v165, (__int64)a3, (__int64 *)&v212, 0, 0, *(double *)a5.m128_u64, a6, a7);
  v168 = a1[1];
  v169 = (__int64)v167;
  v170 = sub_1649960(*(_QWORD *)(a2 - 48));
  LODWORD(v165) = *(unsigned __int8 *)(a2 + 16);
  v211 = v171;
  v210 = v170;
  LOWORD(v214) = 261;
  v212 = &v210;
  v172 = (__int64 *)sub_17066B0(
                      v168,
                      (int)v165 - 24,
                      (__int64)v206,
                      v169,
                      (__int64 *)&v212,
                      0,
                      *(double *)a5.m128_u64,
                      a6,
                      a7);
  v173 = v192;
  v174 = v192 - sub_179D670(v190, v192);
LABEL_171:
  sub_171A350((__int64)&v210, v173, v174);
  v175 = (__int64 *)sub_16498A0(a4);
  v176 = sub_159C0E0(v175, (__int64)&v210);
  if ( *(_BYTE *)(*v172 + 8) == 16 )
    v176 = sub_15A0390(*(_QWORD *)(*v172 + 32), v176);
  LOWORD(v214) = 257;
  v19 = sub_15FB440(26, v172, v176, (__int64)&v212, 0);
  sub_135E100((__int64 *)&v210);
  return v19;
}
