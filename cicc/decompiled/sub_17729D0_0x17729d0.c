// Function: sub_17729D0
// Address: 0x17729d0
//
_QWORD *__fastcall sub_17729D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  double v16; // xmm4_8
  double v17; // xmm5_8
  _QWORD *v18; // r14
  _QWORD *v19; // r8
  unsigned __int8 v20; // bl
  _QWORD *v21; // rbx
  __int64 v22; // rcx
  _QWORD *v23; // r14
  _QWORD *result; // rax
  _QWORD **v25; // rsi
  unsigned int v26; // r9d
  _QWORD *v27; // rdx
  _QWORD *v28; // r8
  _QWORD *v29; // rcx
  __int64 v30; // rdi
  int v31; // eax
  bool v32; // al
  _QWORD **v33; // rcx
  unsigned int v34; // r11d
  unsigned int v35; // r10d
  __int64 v36; // r15
  unsigned __int8 v37; // r12
  _BYTE *v38; // rdi
  unsigned int v39; // ebx
  int v40; // eax
  unsigned __int8 v41; // cl
  _QWORD *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // r14
  __int16 v47; // r12
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // r12
  __int64 v51; // rbx
  __int64 *v52; // rbx
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r14
  unsigned int v58; // ebx
  int v59; // eax
  bool v60; // dl
  __int64 v61; // rax
  unsigned __int64 v62; // r8
  __int64 v63; // r15
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rdx
  _QWORD *v67; // rax
  char v68; // al
  int v69; // eax
  __int64 *v70; // rax
  char v71; // bl
  __int64 v72; // r14
  _QWORD *v73; // rax
  _QWORD *v74; // r10
  _QWORD *v75; // rcx
  _QWORD *v76; // rdx
  unsigned int v77; // eax
  __int64 v78; // rcx
  _QWORD *v79; // rax
  _QWORD *v80; // rdx
  __int64 v81; // rax
  _QWORD **v82; // rax
  _QWORD **v83; // rsi
  __int64 v84; // rax
  __int64 v85; // rbx
  __int64 *v86; // rax
  __int64 v87; // rax
  unsigned int v88; // r10d
  _QWORD *v89; // r13
  unsigned int i; // r15d
  __int64 *v91; // rcx
  __int64 *v92; // r12
  int v93; // ebx
  __int64 v94; // r12
  unsigned __int8 v95; // bl
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 *v98; // rdi
  __int64 v99; // rdx
  double v100; // xmm4_8
  double v101; // xmm5_8
  __int64 v102; // rax
  unsigned int v103; // edx
  unsigned __int64 v104; // rsi
  __int64 *v105; // rax
  __int64 v106; // rax
  __int64 v107; // rbx
  __int64 v108; // r14
  char v109; // al
  __int64 v110; // r8
  unsigned int v111; // esi
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // r14
  unsigned int v115; // ebx
  int v116; // eax
  bool v117; // dl
  __int64 v118; // rax
  unsigned __int64 v119; // r8
  __int64 v120; // r15
  __int64 v121; // rbx
  __int64 v122; // rax
  __int64 v123; // rdx
  _QWORD *v124; // rax
  char v125; // al
  int v126; // eax
  __int64 v127; // rax
  __int64 v128; // rsi
  __int64 v129; // rax
  __int64 v130; // r14
  __int64 v131; // rcx
  __int64 v132; // rsi
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 v135; // rax
  __int64 v136; // r13
  int v137; // eax
  __int64 v138; // rax
  __int64 *v139; // rdx
  unsigned __int64 v140; // rax
  unsigned int v141; // esi
  __int64 v142; // rdx
  __int64 v143; // rsi
  unsigned int v144; // eax
  __int64 v145; // rax
  int v146; // eax
  unsigned __int64 v147; // rax
  _QWORD *v148; // rax
  int v149; // eax
  __int64 v150; // rax
  __int64 v151; // rsi
  unsigned int v152; // eax
  __int64 v153; // rax
  unsigned __int64 v154; // rax
  _QWORD *v155; // rax
  int v156; // eax
  __int64 v157; // rax
  __int64 v158; // r8
  __int64 v159; // rax
  __int64 **v160; // rdx
  __int64 v161; // r14
  unsigned int v162; // r12d
  unsigned __int64 v163; // rcx
  signed __int64 v164; // rsi
  signed __int64 v165; // rcx
  __int64 v166; // rax
  __int64 v167; // rax
  __int64 v168; // rdi
  __int64 *v169; // r13
  __int64 ***v170; // rax
  __int64 **v171; // rbx
  unsigned __int8 *v172; // r14
  unsigned int v173; // r12d
  unsigned int v174; // eax
  __int64 v175; // rdi
  __int64 v176; // rbx
  __int16 v177; // ax
  __int64 v178; // rax
  __int64 v179; // rsi
  __int64 v180; // rax
  __int64 v181; // rdx
  __int64 v182; // rsi
  __int64 v183; // rsi
  __int64 v184; // rdx
  unsigned __int8 *v185; // rsi
  __int64 v186; // rax
  __int64 v187; // rax
  unsigned __int64 v188; // [rsp+0h] [rbp-F0h]
  __int64 v189; // [rsp+8h] [rbp-E8h]
  __int64 v190; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v191; // [rsp+10h] [rbp-E0h]
  __int64 v192; // [rsp+10h] [rbp-E0h]
  char v194; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v195; // [rsp+18h] [rbp-D8h]
  unsigned int v196; // [rsp+20h] [rbp-D0h]
  unsigned __int8 v197; // [rsp+20h] [rbp-D0h]
  unsigned int v198; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v199; // [rsp+20h] [rbp-D0h]
  unsigned int v200; // [rsp+28h] [rbp-C8h]
  unsigned int v201; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v202; // [rsp+28h] [rbp-C8h]
  unsigned int v203; // [rsp+28h] [rbp-C8h]
  __int64 v204; // [rsp+28h] [rbp-C8h]
  unsigned int v205; // [rsp+30h] [rbp-C0h]
  _QWORD *v206; // [rsp+30h] [rbp-C0h]
  bool v207; // [rsp+30h] [rbp-C0h]
  __int64 v208; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v209; // [rsp+30h] [rbp-C0h]
  __int64 *v210; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v211; // [rsp+30h] [rbp-C0h]
  _QWORD *v212; // [rsp+38h] [rbp-B8h]
  unsigned int v213; // [rsp+38h] [rbp-B8h]
  unsigned int v214; // [rsp+38h] [rbp-B8h]
  __int64 v215; // [rsp+38h] [rbp-B8h]
  __int64 v216; // [rsp+38h] [rbp-B8h]
  __int64 v217; // [rsp+38h] [rbp-B8h]
  _QWORD **v218; // [rsp+40h] [rbp-B0h]
  int v219; // [rsp+40h] [rbp-B0h]
  _QWORD *v220; // [rsp+40h] [rbp-B0h]
  _QWORD *v221; // [rsp+40h] [rbp-B0h]
  _QWORD *v222; // [rsp+48h] [rbp-A8h]
  __int64 v223; // [rsp+48h] [rbp-A8h]
  _QWORD **v224; // [rsp+48h] [rbp-A8h]
  _QWORD *v225; // [rsp+48h] [rbp-A8h]
  _QWORD *v226; // [rsp+50h] [rbp-A0h]
  _QWORD *v227; // [rsp+50h] [rbp-A0h]
  bool v228; // [rsp+50h] [rbp-A0h]
  __int64 *v229; // [rsp+50h] [rbp-A0h]
  unsigned int v230; // [rsp+50h] [rbp-A0h]
  __int64 v231; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v232; // [rsp+50h] [rbp-A0h]
  __int64 v233; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v234; // [rsp+50h] [rbp-A0h]
  __int64 v235; // [rsp+50h] [rbp-A0h]
  __int64 *v236; // [rsp+50h] [rbp-A0h]
  int v237; // [rsp+58h] [rbp-98h]
  _QWORD **v238; // [rsp+58h] [rbp-98h]
  int v239; // [rsp+58h] [rbp-98h]
  unsigned int v240; // [rsp+58h] [rbp-98h]
  unsigned int v241; // [rsp+58h] [rbp-98h]
  int v242; // [rsp+58h] [rbp-98h]
  __int64 v243; // [rsp+58h] [rbp-98h]
  __int64 v245; // [rsp+60h] [rbp-90h]
  _QWORD *v246; // [rsp+68h] [rbp-88h]
  _QWORD *v247; // [rsp+68h] [rbp-88h]
  __int64 v248; // [rsp+68h] [rbp-88h]
  unsigned __int64 v249; // [rsp+68h] [rbp-88h]
  int v250; // [rsp+68h] [rbp-88h]
  __int64 v251; // [rsp+68h] [rbp-88h]
  __int64 v252; // [rsp+68h] [rbp-88h]
  _QWORD *v253; // [rsp+68h] [rbp-88h]
  __int64 v254; // [rsp+68h] [rbp-88h]
  unsigned __int64 v255; // [rsp+68h] [rbp-88h]
  unsigned int v256; // [rsp+68h] [rbp-88h]
  __int64 v257; // [rsp+68h] [rbp-88h]
  __int64 **v258; // [rsp+68h] [rbp-88h]
  unsigned __int8 *v259; // [rsp+78h] [rbp-78h] BYREF
  __int64 v260[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v261; // [rsp+90h] [rbp-60h]
  __int64 v262[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v263; // [rsp+B0h] [rbp-40h]

  while ( 1 )
  {
    if ( sub_15FF7F0(a4) )
      return 0;
    if ( *(_BYTE *)(a3 + 16) != 56 )
      a3 = sub_1649C60(a3);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
      break;
    v18 = *(_QWORD **)(a2 - 8);
    v19 = (_QWORD *)*v18;
    if ( *v18 == a3 )
      goto LABEL_44;
LABEL_6:
    v20 = *(_BYTE *)(a3 + 16);
    if ( v20 <= 0x17u )
    {
      if ( v20 != 5 || *(_WORD *)(a3 + 18) != 32 )
      {
LABEL_8:
        v21 = v18 + 3;
        v22 = *(_QWORD *)(a1 + 2664);
        v23 = &v18[3 * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)];
        if ( v21 != v23 )
          goto LABEL_11;
LABEL_66:
        if ( *(_QWORD *)a2 == *(_QWORD *)a3 )
          return sub_176FEF0((__int64 *)a2, (__int64 *)a3, a4, v22, a6, a7, a8, a9, v16, v17, a12, a13);
        return 0;
      }
    }
    else if ( v20 != 56 )
    {
      goto LABEL_8;
    }
    if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
      v25 = *(_QWORD ***)(a3 - 8);
    else
      v25 = (_QWORD **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    v246 = *v25;
    v26 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v19 != *v25 )
    {
      v71 = *(_BYTE *)(a2 + 23) & 0x40;
      v215 = (__int64)v19;
      v220 = v18;
      v72 = a2;
      v224 = v25;
      v230 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v240 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
      v207 = v240 == v26;
      v73 = (_QWORD *)sub_13CF970(a2);
      v74 = (_QWORD *)*v73;
      v75 = v73;
      v76 = v220;
      if ( *(_QWORD *)*v73 == *v246 && v207 )
      {
        if ( v230 == 1 )
        {
LABEL_165:
          v245 = (__int64)v74;
          v263 = 257;
          result = sub_1648A60(56, 2u);
          if ( result )
          {
            v131 = (__int64)v246;
            v253 = result;
            sub_17582E0((__int64)result, a4, v245, v131, (__int64)v262);
            return v253;
          }
          return result;
        }
        v77 = 1;
        while ( v25[3 * v77] == (_QWORD *)v75[3 * v77] )
        {
          if ( v230 == ++v77 )
            goto LABEL_165;
        }
        v76 = v220;
      }
      v78 = 3LL * v230;
      if ( (*(_BYTE *)(a2 + 17) & 2) != 0 && (*(_BYTE *)(a3 + 17) & 2) != 0 )
      {
        v79 = v76 + 3;
        v80 = &v76[v78];
        if ( v79 == v80 )
          goto LABEL_93;
        while ( *(_BYTE *)(*v79 + 16LL) == 13 )
        {
          v79 += 3;
          if ( v80 == v79 )
            goto LABEL_93;
        }
        v81 = *(_QWORD *)(a2 + 8);
        if ( v81 )
        {
          if ( !*(_QWORD *)(v81 + 8) )
          {
LABEL_93:
            v82 = v25 + 3;
            v83 = &v25[3 * v240];
            if ( v224 + 3 == v83 )
              goto LABEL_99;
            while ( *((_BYTE *)*v82 + 16) == 13 )
            {
              v82 += 3;
              if ( v83 == v82 )
                goto LABEL_99;
            }
            v84 = *(_QWORD *)(a3 + 8);
            if ( v84 )
            {
              if ( !*(_QWORD *)(v84 + 8) )
              {
LABEL_99:
                v85 = sub_1649C60(v215);
                v86 = (__int64 *)sub_13CF970(a3);
                if ( v85 == sub_1649C60(*v86) )
                {
                  v169 = (__int64 *)sub_170B0F0(a1, a2, *(double *)a6.m128_u64, a7, a8);
                  v170 = (__int64 ***)sub_170B0F0(a1, a3, *(double *)a6.m128_u64, a7, a8);
                  v171 = *v170;
                  v172 = (unsigned __int8 *)v170;
                  if ( *v170 != (__int64 **)*v169 )
                  {
                    v258 = (__int64 **)*v169;
                    v173 = sub_1643030(*v169);
                    v174 = sub_1643030((__int64)v171);
                    v263 = 257;
                    v175 = *(_QWORD *)(a1 + 8);
                    if ( v173 >= v174 )
                      v169 = (__int64 *)sub_1708970(v175, 36, (__int64)v169, v171, v262);
                    else
                      v172 = sub_1708970(v175, 36, (__int64)v172, v258, v262);
                  }
                  v263 = 257;
                  v176 = *(_QWORD *)(a1 + 8);
                  v177 = sub_15FF420(a4);
                  v97 = a5;
                  v98 = (__int64 *)a1;
                  v99 = (__int64)sub_17203D0(v176, v177, (__int64)v169, (__int64)v172, v262);
                  return (_QWORD *)sub_170E100(v98, v97, v99, a6, a7, a8, a9, v100, v101, a12, a13);
                }
                v71 = *(_BYTE *)(a2 + 23) & 0x40;
                v78 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              }
            }
          }
        }
      }
      v87 = a2 - v78 * 8;
      if ( v71 )
      {
        v87 = *(_QWORD *)(a2 - 8);
        v72 = v87 + v78 * 8;
      }
      while ( 1 )
      {
        v87 += 24;
        if ( v87 == v72 )
          break;
        if ( *(_BYTE *)(*(_QWORD *)v87 + 16LL) != 13 )
          return 0;
      }
      if ( *(_QWORD *)a2 == *(_QWORD *)a3 )
      {
        v22 = *(_QWORD *)(a1 + 2664);
        return sub_176FEF0((__int64 *)a2, (__int64 *)a3, a4, v22, a6, a7, a8, a9, v16, v17, a12, a13);
      }
      return 0;
    }
    v27 = v18 + 3;
    v28 = &v18[3 * v26];
    if ( v18 + 3 == v28 )
    {
LABEL_69:
      a4 = sub_15FF5D0(a4);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v70 = *(__int64 **)(a2 - 8);
      else
        v70 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      a2 = a3;
      a3 = *v70;
    }
    else
    {
      v29 = v18 + 3;
      while ( 1 )
      {
        v30 = *v29;
        if ( *(_BYTE *)(*v29 + 16LL) != 13 )
          break;
        if ( *(_DWORD *)(v30 + 32) <= 0x40u )
        {
          v32 = *(_QWORD *)(v30 + 24) == 0;
        }
        else
        {
          v205 = v26;
          v212 = v28;
          v222 = v29;
          v226 = v27;
          v237 = *(_DWORD *)(v30 + 32);
          v31 = sub_16A57B0(v30 + 24);
          v27 = v226;
          v29 = v222;
          v28 = v212;
          v26 = v205;
          v32 = v237 == v31;
        }
        if ( !v32 )
          break;
        v29 += 3;
        if ( v28 == v29 )
          goto LABEL_69;
      }
      v33 = v25 + 3;
      v34 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
      if ( v25 + 3 == &v25[3 * v34] )
        goto LABEL_76;
      v238 = &v25[3 * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)];
      v35 = a4;
      v36 = a3;
      v37 = v20;
      do
      {
        v38 = *v33;
        if ( *((_BYTE *)*v33 + 16) != 13 )
        {
LABEL_30:
          v41 = v37;
          v21 = v27;
          a3 = v36;
          v42 = v18;
          a4 = v35;
          v23 = v28;
          v228 = (*(_BYTE *)(a2 + 17) & 2) != 0;
          if ( (*(_BYTE *)(a2 + 17) & 2) != 0 && (v228 = (*(_BYTE *)(a3 + 17) & 2) != 0, (*(_BYTE *)(a3 + 17) & 2) != 0) )
          {
            if ( v34 != v26 )
            {
LABEL_33:
              if ( (*(_BYTE *)(a2 + 16) == 5 || (v43 = *(_QWORD *)(a2 + 8)) != 0 && !*(_QWORD *)(v43 + 8))
                && (v41 == 5 || (v44 = *(_QWORD *)(a3 + 8)) != 0 && !*(_QWORD *)(v44 + 8)) )
              {
                v45 = sub_170B0F0(a1, a2, *(double *)a6.m128_u64, a7, a8);
                v46 = sub_170B0F0(a1, a3, *(double *)a6.m128_u64, a7, a8);
LABEL_40:
                v263 = 257;
                v47 = sub_15FF420(a4);
                result = sub_1648A60(56, 2u);
                if ( result )
                {
                  v48 = v46;
                  v49 = v45;
                  goto LABEL_42;
                }
                return result;
              }
LABEL_108:
              v22 = *(_QWORD *)(a1 + 2664);
LABEL_11:
              while ( *(_BYTE *)(*v21 + 16LL) == 13 )
              {
                v21 += 3;
                if ( v23 == v21 )
                  goto LABEL_66;
              }
              return 0;
            }
          }
          else if ( v34 != v26 )
          {
            goto LABEL_108;
          }
          if ( v34 == 1 )
          {
LABEL_118:
            v94 = *(_QWORD *)(a1 + 8);
            v95 = sub_15FF820(a4);
            v96 = sub_1643320(*(_QWORD **)(v94 + 24));
            v97 = a5;
            v98 = (__int64 *)a1;
            v99 = sub_159C470(v96, v95, 0);
            return (_QWORD *)sub_170E100(v98, v97, v99, a6, a7, a8, a9, v100, v101, a12, a13);
          }
          v197 = v41;
          v88 = 0;
          v250 = 0;
          v225 = v21;
          v241 = v34;
          v208 = a3;
          v221 = v28;
          v216 = a2;
          v89 = v42;
          v201 = a4;
          for ( i = 1; i != v241; ++i )
          {
            v91 = (__int64 *)v89[3 * i];
            v92 = v25[3 * i];
            if ( v92 != v91 )
            {
              v93 = sub_1643030(*v91);
              if ( v93 != (unsigned int)sub_1643030(*v92) || v250 )
              {
                v41 = v197;
                v21 = v225;
                v23 = v221;
                a2 = v216;
                a3 = v208;
                a4 = v201;
                if ( v228 )
                  goto LABEL_33;
                goto LABEL_108;
              }
              v250 = 1;
              v88 = i;
            }
          }
          v21 = v225;
          v23 = v221;
          a2 = v216;
          a3 = v208;
          a4 = v201;
          if ( !v250 )
            goto LABEL_118;
          v256 = v88;
          if ( v228 )
          {
            v158 = sub_13CF970(v216);
            v159 = 3LL * v256;
            v45 = *(_QWORD *)(v158 + v159 * 8);
            v46 = (__int64)v25[v159];
            goto LABEL_40;
          }
          goto LABEL_108;
        }
        v39 = *((_DWORD *)v38 + 8);
        if ( v39 <= 0x40 )
        {
          if ( *((_QWORD *)v38 + 3) )
            goto LABEL_30;
        }
        else
        {
          v196 = v35;
          v200 = v26;
          v206 = v28;
          v213 = v34;
          v218 = v33;
          v227 = v27;
          v40 = sub_16A57B0((__int64)(v38 + 24));
          v27 = v227;
          v33 = v218;
          v34 = v213;
          v28 = v206;
          v26 = v200;
          v35 = v196;
          if ( v39 != v40 )
            goto LABEL_30;
        }
        v33 += 3;
      }
      while ( v238 != v33 );
      a4 = v35;
LABEL_76:
      a3 = (__int64)v246;
    }
  }
  v18 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v19 = (_QWORD *)*v18;
  if ( *v18 != a3 )
    goto LABEL_6;
LABEL_44:
  if ( (*(_BYTE *)(a2 + 17) & 2) == 0 )
    goto LABEL_6;
  v50 = *(_QWORD *)(a1 + 2664);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v51 = *(_QWORD *)(a2 - 8);
  else
    v51 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v52 = (__int64 *)(v51 + 24);
  v53 = sub_16348C0(a2) | 4;
  v219 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v219 == 1 )
    goto LABEL_168;
  v239 = 2;
  v229 = v52;
  v223 = 0;
  v54 = 1;
  v214 = a4;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
    goto LABEL_64;
  while ( 2 )
  {
    v55 = *(_QWORD *)(a2 - 8);
LABEL_50:
    v56 = 3 * v54;
    v57 = *(_QWORD *)(v55 + 8 * v56);
    if ( *(_BYTE *)(v57 + 16) == 13 )
    {
      v58 = *(_DWORD *)(v57 + 32);
      if ( v58 <= 0x40 )
      {
        v60 = *(_QWORD *)(v57 + 24) == 0;
      }
      else
      {
        v248 = v53;
        v59 = sub_16A57B0(v57 + 24);
        v53 = v248;
        v60 = v58 == v59;
      }
      v61 = v53;
      v62 = v53 & 0xFFFFFFFFFFFFFFF8LL;
      v63 = v62;
      v64 = (v61 >> 2) & 1;
      if ( !v60 )
      {
        if ( (_BYTE)v64 )
        {
          v143 = v62;
          if ( !v62 )
            goto LABEL_205;
        }
        else
        {
          if ( v62 )
          {
            v249 = v62;
            v65 = sub_15A9930(v50, v62);
            v62 = v249;
            v66 = v65;
            v67 = *(_QWORD **)(v57 + 24);
            if ( *(_DWORD *)(v57 + 32) > 0x40u )
              v67 = (_QWORD *)*v67;
            v223 += *(_QWORD *)(v66 + 8LL * (unsigned int)v67 + 16);
LABEL_59:
            v63 = sub_1643D30(v62, *v229);
LABEL_60:
            v68 = *(_BYTE *)(v63 + 8);
            if ( ((v68 - 14) & 0xFD) != 0 )
            {
              v53 = 0;
              if ( v68 == 13 )
                v53 = v63;
            }
            else
            {
              v53 = *(_QWORD *)(v63 + 24) | 4LL;
            }
            v229 += 3;
            v69 = v239 + 1;
            if ( v219 == v239 )
            {
              a4 = v214;
              goto LABEL_168;
            }
            ++v239;
            v54 = (unsigned int)(v69 - 1);
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              continue;
LABEL_64:
            v55 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            goto LABEL_50;
          }
LABEL_205:
          v255 = v62;
          v150 = sub_1643D30(0, *v229);
          v62 = v255;
          v143 = v150;
        }
        v199 = v62;
        v144 = sub_15A9FE0(v50, v143);
        v254 = 1;
        v62 = v199;
        v211 = v144;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v143 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v145 = v254 * *(_QWORD *)(v143 + 32);
              v143 = *(_QWORD *)(v143 + 24);
              v254 = v145;
              continue;
            case 1:
              v102 = 16;
              break;
            case 2:
              v102 = 32;
              break;
            case 3:
            case 9:
              v102 = 64;
              break;
            case 4:
              v102 = 80;
              break;
            case 5:
            case 6:
              v102 = 128;
              break;
            case 7:
              v149 = sub_15A9520(v50, 0);
              v62 = v199;
              v102 = (unsigned int)(8 * v149);
              break;
            case 0xB:
              v102 = *(_DWORD *)(v143 + 8) >> 8;
              break;
            case 0xD:
              v148 = (_QWORD *)sub_15A9930(v50, v143);
              v62 = v199;
              v102 = 8LL * *v148;
              break;
            case 0xE:
              v204 = *(_QWORD *)(v143 + 32);
              v147 = sub_12BE0A0(v50, *(_QWORD *)(v143 + 24));
              v62 = v199;
              v102 = 8 * v204 * v147;
              break;
            case 0xF:
              v146 = sub_15A9520(v50, *(_DWORD *)(v143 + 8) >> 8);
              v62 = v199;
              v102 = (unsigned int)(8 * v146);
              break;
          }
          break;
        }
        v103 = *(_DWORD *)(v57 + 32);
        v104 = (v211 + ((unsigned __int64)(v254 * v102 + 7) >> 3) - 1) / v211 * v211;
        v105 = *(__int64 **)(v57 + 24);
        if ( v103 > 0x40 )
          v106 = *v105;
        else
          v106 = (__int64)((_QWORD)v105 << (64 - (unsigned __int8)v103)) >> (64 - (unsigned __int8)v103);
        v223 += v104 * v106;
      }
      if ( !(_BYTE)v64 || !v62 )
        goto LABEL_59;
      goto LABEL_60;
    }
    break;
  }
  a4 = v214;
  v107 = *(_QWORD *)(v55 + 8 * v56);
  v202 = v53 & 0xFFFFFFFFFFFFFFF8LL;
  v209 = v53 & 0xFFFFFFFFFFFFFFF8LL;
  v194 = (v53 >> 2) & 1;
  if ( ((v53 >> 2) & 1) == 0 || (v108 = v53 & 0xFFFFFFFFFFFFFFF8LL, (v53 & 0xFFFFFFFFFFFFFFF8LL) == 0) )
    v108 = sub_1643D30(v202, *v229);
  v217 = 1;
  v198 = sub_15A9FE0(v50, v108);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v108 + 8) )
    {
      case 1:
        v251 = 16;
        goto LABEL_132;
      case 2:
        v251 = 32;
        goto LABEL_132;
      case 3:
      case 9:
        v251 = 64;
        goto LABEL_132;
      case 4:
        v251 = 80;
        goto LABEL_132;
      case 5:
      case 6:
        v251 = 128;
        goto LABEL_132;
      case 7:
        v251 = 8 * (unsigned int)sub_15A9520(v50, 0);
        goto LABEL_132;
      case 0xB:
        v251 = *(_DWORD *)(v108 + 8) >> 8;
        goto LABEL_132;
      case 0xD:
        v251 = 8LL * *(_QWORD *)sub_15A9930(v50, v108);
        goto LABEL_132;
      case 0xE:
        v128 = *(_QWORD *)(v108 + 24);
        v129 = *(_QWORD *)(v108 + 32);
        v130 = 1;
        v252 = v129;
        v191 = (unsigned int)sub_15A9FE0(v50, v128);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v128 + 8) )
          {
            case 1:
              v187 = 16;
              goto LABEL_258;
            case 2:
              v187 = 32;
              goto LABEL_258;
            case 3:
            case 9:
              v187 = 64;
              goto LABEL_258;
            case 4:
              v187 = 80;
              goto LABEL_258;
            case 5:
            case 6:
              v187 = 128;
              goto LABEL_258;
            case 7:
              v187 = 8 * (unsigned int)sub_15A9520(v50, 0);
              goto LABEL_258;
            case 0xB:
              v187 = *(_DWORD *)(v128 + 8) >> 8;
              goto LABEL_258;
            case 0xD:
              v187 = 8LL * *(_QWORD *)sub_15A9930(v50, v128);
              goto LABEL_258;
            case 0xE:
              v190 = *(_QWORD *)(v128 + 32);
              v187 = 8 * v190 * sub_12BE0A0(v50, *(_QWORD *)(v128 + 24));
              goto LABEL_258;
            case 0xF:
              v187 = 8 * (unsigned int)sub_15A9520(v50, *(_DWORD *)(v128 + 8) >> 8);
LABEL_258:
              v251 = 8 * v191 * v252 * ((v191 + ((unsigned __int64)(v130 * v187 + 7) >> 3) - 1) / v191);
              goto LABEL_132;
            case 0x10:
              v186 = *(_QWORD *)(v128 + 32);
              v128 = *(_QWORD *)(v128 + 24);
              v130 *= v186;
              continue;
            default:
              goto LABEL_268;
          }
        }
      case 0xF:
        v251 = 8 * (unsigned int)sub_15A9520(v50, *(_DWORD *)(v108 + 8) >> 8);
LABEL_132:
        if ( !v194 || !v202 )
          v209 = sub_1643D30(v202, *v229);
        v109 = *(_BYTE *)(v209 + 8);
        if ( ((v109 - 14) & 0xFD) != 0 )
        {
          v110 = 0;
          if ( v109 == 13 )
            v110 = v209;
        }
        else
        {
          v110 = *(_QWORD *)(v209 + 24) | 4LL;
        }
        v111 = v239;
        if ( v219 == v239 )
          goto LABEL_225;
        v210 = v229 + 3;
        v242 = v239 + 1;
        v189 = v107;
        v203 = a4;
        v112 = v111;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          goto LABEL_138;
        break;
      case 0x10:
        v127 = v217 * *(_QWORD *)(v108 + 32);
        v108 = *(_QWORD *)(v108 + 24);
        v217 = v127;
        continue;
      default:
LABEL_268:
        BUG();
    }
    break;
  }
  while ( 2 )
  {
    v113 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
LABEL_139:
    v114 = *(_QWORD *)(v113 + 24 * v112);
    if ( *(_BYTE *)(v114 + 16) != 13 )
    {
      a4 = v203;
      goto LABEL_168;
    }
    v115 = *(_DWORD *)(v114 + 32);
    if ( v115 <= 0x40 )
    {
      v117 = *(_QWORD *)(v114 + 24) == 0;
    }
    else
    {
      v231 = v110;
      v116 = sub_16A57B0(v114 + 24);
      v110 = v231;
      v117 = v115 == v116;
    }
    v118 = v110;
    v119 = v110 & 0xFFFFFFFFFFFFFFF8LL;
    v120 = v119;
    v121 = (v118 >> 2) & 1;
    if ( v117 )
    {
LABEL_175:
      if ( !(_BYTE)v121 || !v119 )
        goto LABEL_148;
    }
    else
    {
      if ( (_BYTE)v121 )
      {
        v151 = v119;
        if ( !v119 )
        {
LABEL_219:
          v234 = v119;
          v157 = sub_1643D30(0, *v210);
          v119 = v234;
          v151 = v157;
        }
        v188 = v119;
        v233 = 1;
        v152 = sub_15A9FE0(v50, v151);
        v119 = v188;
        v195 = v152;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v151 + 8) )
          {
            case 1:
              v138 = 16;
              goto LABEL_172;
            case 2:
              v138 = 32;
              goto LABEL_172;
            case 3:
            case 9:
              v138 = 64;
              goto LABEL_172;
            case 4:
              v138 = 80;
              goto LABEL_172;
            case 5:
            case 6:
              v138 = 128;
              goto LABEL_172;
            case 7:
              v156 = sub_15A9520(v50, 0);
              v119 = v188;
              v138 = (unsigned int)(8 * v156);
              goto LABEL_172;
            case 0xB:
              v138 = *(_DWORD *)(v151 + 8) >> 8;
              goto LABEL_172;
            case 0xD:
              v155 = (_QWORD *)sub_15A9930(v50, v151);
              v119 = v188;
              v138 = 8LL * *v155;
              goto LABEL_172;
            case 0xE:
              v192 = *(_QWORD *)(v151 + 32);
              v154 = sub_12BE0A0(v50, *(_QWORD *)(v151 + 24));
              v119 = v188;
              v138 = 8 * v192 * v154;
              goto LABEL_172;
            case 0xF:
              v137 = sub_15A9520(v50, *(_DWORD *)(v151 + 8) >> 8);
              v119 = v188;
              v138 = (unsigned int)(8 * v137);
LABEL_172:
              v139 = *(__int64 **)(v114 + 24);
              v140 = v195 * ((v195 + ((unsigned __int64)(v233 * v138 + 7) >> 3) - 1) / v195);
              v141 = *(_DWORD *)(v114 + 32);
              if ( v141 > 0x40 )
                v142 = *v139;
              else
                v142 = (__int64)((_QWORD)v139 << (64 - (unsigned __int8)v141)) >> (64 - (unsigned __int8)v141);
              v223 += v142 * v140;
              goto LABEL_175;
            case 0x10:
              v153 = v233 * *(_QWORD *)(v151 + 32);
              v151 = *(_QWORD *)(v151 + 24);
              v233 = v153;
              continue;
            default:
              goto LABEL_268;
          }
        }
      }
      if ( !v119 )
        goto LABEL_219;
      v232 = v119;
      v122 = sub_15A9930(v50, v119);
      v119 = v232;
      v123 = v122;
      v124 = *(_QWORD **)(v114 + 24);
      if ( *(_DWORD *)(v114 + 32) > 0x40u )
        v124 = (_QWORD *)*v124;
      v223 += *(_QWORD *)(v123 + 8LL * (unsigned int)v124 + 16);
LABEL_148:
      v120 = sub_1643D30(v119, *v210);
    }
    v125 = *(_BYTE *)(v120 + 8);
    if ( ((v125 - 14) & 0xFD) != 0 )
    {
      v110 = 0;
      if ( v125 == 13 )
        v110 = v120;
    }
    else
    {
      v110 = *(_QWORD *)(v120 + 24) | 4LL;
    }
    v210 += 3;
    v126 = v242 + 1;
    if ( v219 != v242 )
    {
      ++v242;
      v112 = (unsigned int)(v126 - 1);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
        continue;
LABEL_138:
      v113 = *(_QWORD *)(a2 - 8);
      goto LABEL_139;
    }
    break;
  }
  v107 = v189;
  a4 = v203;
LABEL_225:
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v160 = *(__int64 ***)(a2 - 8);
  else
    v160 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v132 = **v160;
  v161 = sub_15A9650(v50, v132);
  v162 = *(_DWORD *)(v161 + 8) >> 8;
  if ( v223 )
  {
    v163 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v162);
    v164 = v163 & v223;
    v165 = (v198 * ((v198 + ((unsigned __int64)(v251 * v217 + 7) >> 3) - 1) / v198)) & v163;
    v243 = v164 / v165;
    if ( v164 != v164 / v165 * v165 )
      goto LABEL_168;
    if ( v161 != *(_QWORD *)v107 )
    {
      v261 = 257;
      v257 = *(_QWORD *)(a1 + 8);
      if ( v161 != *(_QWORD *)v107 )
      {
        if ( *(_BYTE *)(v107 + 16) > 0x10u )
        {
          v263 = 257;
          v107 = sub_15FE0A0((_QWORD *)v107, v161, 1, (__int64)v262, 0);
          v178 = *(_QWORD *)(v257 + 8);
          if ( v178 )
          {
            v236 = *(__int64 **)(v257 + 16);
            sub_157E9D0(v178 + 40, v107);
            v179 = *v236;
            v180 = *(_QWORD *)(v107 + 24) & 7LL;
            *(_QWORD *)(v107 + 32) = v236;
            v179 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v107 + 24) = v179 | v180;
            *(_QWORD *)(v179 + 8) = v107 + 24;
            *v236 = *v236 & 7 | (v107 + 24);
          }
          sub_164B780(v107, v260);
          v259 = (unsigned __int8 *)v107;
          if ( !*(_QWORD *)(v257 + 80) )
            sub_4263D6(v107, v260, v181);
          (*(void (__fastcall **)(__int64, unsigned __int8 **))(v257 + 88))(v257 + 64, &v259);
          v182 = *(_QWORD *)v257;
          if ( *(_QWORD *)v257 )
          {
            v259 = *(unsigned __int8 **)v257;
            sub_1623A60((__int64)&v259, v182, 2);
            v183 = *(_QWORD *)(v107 + 48);
            v184 = v107 + 48;
            if ( v183 )
            {
              sub_161E7C0(v107 + 48, v183);
              v184 = v107 + 48;
            }
            v185 = v259;
            *(_QWORD *)(v107 + 48) = v259;
            if ( v185 )
              sub_1623210((__int64)&v259, v185, v184);
          }
        }
        else
        {
          v235 = sub_15A4750((__int64 ***)v107, (__int64 **)v161, 1);
          v166 = sub_14DBA30(v235, *(_QWORD *)(v257 + 96), 0);
          v107 = v235;
          if ( v166 )
            v107 = v166;
        }
      }
    }
    v167 = sub_15A0680(v161, v243, 0);
    v132 = v107;
    v263 = 259;
    v168 = *(_QWORD *)(a1 + 8);
    v262[0] = (__int64)"offset";
    v107 = (__int64)sub_17094A0(v168, v107, v167, v262, 0, 0, *(double *)a6.m128_u64, a7, a8);
LABEL_235:
    if ( !v107 )
    {
LABEL_168:
      v132 = a2;
      v107 = sub_170B0F0(a1, a2, *(double *)a6.m128_u64, a7, a8);
    }
  }
  else if ( v162 < (unsigned int)sub_1643030(*(_QWORD *)v107) )
  {
    v132 = 36;
    v263 = 257;
    v107 = (__int64)sub_1708970(*(_QWORD *)(a1 + 8), 36, v107, (__int64 **)v161, v262);
    goto LABEL_235;
  }
  v47 = sub_15FF420(a4);
  v135 = sub_15A06D0(*(__int64 ***)v107, v132, v133, v134);
  v263 = 257;
  v136 = v135;
  result = sub_1648A60(56, 2u);
  if ( result )
  {
    v48 = v136;
    v49 = v107;
LABEL_42:
    v247 = result;
    sub_17582E0((__int64)result, v47, v49, v48, (__int64)v262);
    return v247;
  }
  return result;
}
