// Function: sub_20E14F0
// Address: 0x20e14f0
//
float __fastcall sub_20E14F0(
        _QWORD *a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9)
{
  _QWORD *v9; // r12
  __int64 (*v10)(void); // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  double v13; // xmm6_8
  double v14; // xmm5_8
  __int64 v15; // rbx
  unsigned int v16; // r14d
  _QWORD *v17; // r13
  __int64 v18; // r15
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // r10
  __int64 v22; // rsi
  unsigned int v23; // edi
  __int64 *v24; // rdx
  __int64 v25; // r11
  __int64 v26; // rdx
  unsigned int v27; // edx
  __int16 v28; // ax
  int v29; // eax
  __int64 v30; // rdx
  _DWORD *v31; // rax
  _DWORD *j; // rdx
  int *v33; // r8
  __int64 v34; // rbx
  _QWORD *v35; // r13
  int v36; // r14d
  int v37; // r9d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 (*v40)(); // rax
  float result; // xmm0_4
  int v42; // edx
  _DWORD *v43; // rax
  __int64 *v44; // rax
  char v45; // dl
  __int64 (*v46)(); // rax
  _DWORD *v47; // rcx
  int v48; // eax
  int v49; // r12d
  unsigned int v50; // r15d
  int v51; // edx
  _QWORD *v52; // rcx
  __int64 (*v53)(); // rax
  __int64 v54; // rcx
  __int64 v55; // rax
  int v56; // edx
  unsigned __int64 v57; // rsi
  __int64 v58; // r8
  __int64 v59; // rdx
  int *v60; // rax
  int v61; // edi
  __int64 (*v62)(); // rax
  unsigned int v63; // ecx
  int *v64; // r15
  char v65; // dl
  int *v66; // rax
  char v67; // di
  char v68; // si
  bool v69; // al
  int *v70; // r10
  unsigned int v71; // r15d
  __int64 v72; // rax
  __int64 *v73; // rdi
  __int64 *v74; // rsi
  __int64 (__fastcall *v75)(double, double, double, double, double, double, double); // rax
  __int64 v76; // rax
  __int64 v77; // r14
  int v78; // r15d
  __int64 v79; // rsi
  int v80; // eax
  __int64 *v81; // rcx
  __int64 v82; // r12
  __int64 v83; // rax
  __int64 v84; // rsi
  __int16 *v85; // rax
  __int16 v86; // dx
  __int64 v87; // rdx
  __int64 (*v88)(); // rax
  float (__fastcall *v89)(__int64, _QWORD, float); // rbx
  char v90; // al
  unsigned int v91; // eax
  unsigned int v92; // ecx
  _DWORD *v93; // rdi
  unsigned int v94; // eax
  int v95; // eax
  unsigned __int64 v96; // rax
  unsigned __int64 v97; // rax
  int v98; // ebx
  __int64 v99; // r13
  _DWORD *v100; // rax
  __int64 v101; // rdx
  _DWORD *i; // rdx
  __int64 v103; // rcx
  __int64 v104; // rax
  int v105; // esi
  __int64 v106; // r10
  int v107; // esi
  unsigned int v108; // edx
  __int64 *v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rax
  __int64 *v112; // r12
  __int64 v113; // r14
  __int64 *v114; // rbx
  __int16 v115; // ax
  char v116; // r12
  __int64 v117; // rax
  unsigned __int64 v118; // r12
  int v119; // eax
  __int64 v120; // r12
  __int64 *v121; // rax
  __int64 v122; // r8
  _DWORD *v123; // rax
  __int64 v124; // r9
  _QWORD *v125; // rcx
  __int64 v126; // r8
  __int64 v127; // rdi
  unsigned __int64 v128; // rax
  unsigned int v129; // edx
  __int64 v130; // r9
  int v131; // eax
  unsigned __int64 v132; // rsi
  __int64 v133; // r13
  __int64 v134; // rbx
  unsigned __int64 v135; // r13
  __int64 *v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // rax
  unsigned __int64 v139; // rax
  unsigned int v140; // ebx
  __int64 v141; // rcx
  char v142; // al
  __int64 v143; // rdx
  _QWORD *v144; // rdi
  _QWORD *v145; // rdx
  __int64 v146; // rsi
  __int64 v147; // r10
  unsigned __int64 v148; // r11
  __int64 v149; // rax
  __int64 v150; // r13
  __int16 v151; // ax
  char v152; // r12
  __int16 v153; // ax
  __int64 v154; // rax
  __int64 *v155; // rdi
  __int64 v156; // rbx
  __int64 *v157; // r13
  __int64 v158; // rdx
  __int64 v159; // rsi
  __int64 *v160; // rcx
  int v161; // ecx
  __int64 v162; // rdi
  __int64 (*v163)(); // rax
  __int64 v164; // rax
  char v165; // al
  __int64 v166; // rax
  int v167; // edx
  int v168; // eax
  int v169; // r15d
  int *v170; // rcx
  int v171; // edi
  _DWORD *v172; // rax
  char v173; // al
  int v174; // eax
  int v175; // ecx
  int v176; // eax
  int v177; // r8d
  __int64 v178; // r15
  unsigned int v179; // ecx
  int v180; // edi
  int v181; // r10d
  int v182; // eax
  int v183; // r8d
  __int64 v184; // r15
  unsigned int v185; // ecx
  int v186; // r10d
  int v187; // edi
  int *v188; // [rsp+0h] [rbp-170h]
  char v189; // [rsp+8h] [rbp-168h]
  _QWORD *v190; // [rsp+8h] [rbp-168h]
  unsigned int v191; // [rsp+10h] [rbp-160h]
  unsigned int v192; // [rsp+10h] [rbp-160h]
  unsigned int v193; // [rsp+10h] [rbp-160h]
  char v194; // [rsp+20h] [rbp-150h]
  __int64 v195; // [rsp+20h] [rbp-150h]
  __int64 v196; // [rsp+20h] [rbp-150h]
  __int64 v197; // [rsp+28h] [rbp-148h]
  unsigned int v198; // [rsp+28h] [rbp-148h]
  int v199; // [rsp+30h] [rbp-140h]
  __int64 *v200; // [rsp+30h] [rbp-140h]
  int *v201; // [rsp+30h] [rbp-140h]
  __int64 v202; // [rsp+38h] [rbp-138h]
  _QWORD *v203; // [rsp+38h] [rbp-138h]
  int v204; // [rsp+38h] [rbp-138h]
  float v205; // [rsp+40h] [rbp-130h]
  __int64 *v206; // [rsp+40h] [rbp-130h]
  unsigned int v207; // [rsp+48h] [rbp-128h]
  __int64 v208; // [rsp+48h] [rbp-128h]
  __int64 *v210; // [rsp+58h] [rbp-118h]
  unsigned int v213; // [rsp+70h] [rbp-100h]
  float v214; // [rsp+74h] [rbp-FCh]
  _QWORD *v215; // [rsp+78h] [rbp-F8h]
  __int64 v216; // [rsp+78h] [rbp-F8h]
  __int64 v217; // [rsp+78h] [rbp-F8h]
  int *v218; // [rsp+78h] [rbp-F8h]
  __int64 v219; // [rsp+78h] [rbp-F8h]
  __int64 v220; // [rsp+80h] [rbp-F0h]
  unsigned int v221; // [rsp+80h] [rbp-F0h]
  int v222; // [rsp+88h] [rbp-E8h]
  int v223; // [rsp+88h] [rbp-E8h]
  bool v224; // [rsp+8Ch] [rbp-E4h]
  int v225; // [rsp+A8h] [rbp-C8h] BYREF
  int *v226; // [rsp+B0h] [rbp-C0h]
  int *v227; // [rsp+B8h] [rbp-B8h]
  int *v228; // [rsp+C0h] [rbp-B0h]
  __int64 v229; // [rsp+C8h] [rbp-A8h]
  __int64 v230; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 *v231; // [rsp+D8h] [rbp-98h]
  __int64 *v232; // [rsp+E0h] [rbp-90h]
  __int64 v233; // [rsp+E8h] [rbp-88h]
  int v234; // [rsp+F0h] [rbp-80h]
  _BYTE v235[120]; // [rsp+F8h] [rbp-78h] BYREF

  v9 = a1;
  v202 = 0;
  v215 = *(_QWORD **)(*a1 + 40LL);
  v10 = *(__int64 (**)(void))(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v10 != sub_1D00B10 )
    v202 = v10();
  v233 = 8;
  v232 = (__int64 *)v235;
  v231 = (__int64 *)v235;
  v234 = 0;
  v230 = 0;
  v11 = *(unsigned int *)(a2 + 112);
  v12 = v215[26] + 40LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF);
  v222 = *(_DWORD *)(v12 + 16);
  if ( v222 )
    v222 = **(_DWORD **)(v12 + 8);
  v199 = *(_DWORD *)v12;
  *(_QWORD *)&v13 = dword_4530D80;
  *(_QWORD *)&v14 = *(unsigned int *)(a2 + 116);
  v224 = a4 != 0 && a3 != 0;
  v205 = *(float *)(a2 + 116);
  if ( !v224 )
  {
    v207 = 0;
    v214 = 0.0;
    goto LABEL_7;
  }
  v147 = *a4;
  v148 = *a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v148 || (v149 = *(_QWORD *)(v148 + 16)) == 0 )
  {
    v154 = *(_QWORD *)(a1[1] + 272LL);
    v155 = *(__int64 **)(v154 + 536);
    v156 = *(unsigned int *)(v154 + 544);
    v157 = &v155[2 * v156];
    v158 = (16 * v156) >> 4;
    if ( 16 * v156 )
    {
      do
      {
        while ( 1 )
        {
          v159 = v158 >> 1;
          v160 = &v155[2 * (v158 >> 1)];
          if ( (*(_DWORD *)((*v160 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v160 >> 1) & 3) >= (*(_DWORD *)(v148 + 24) | (unsigned int)(v147 >> 1) & 3) )
            break;
          v155 = v160 + 2;
          v158 = v158 - v159 - 1;
          if ( v158 <= 0 )
            goto LABEL_217;
        }
        v158 >>= 1;
      }
      while ( v159 > 0 );
    }
LABEL_217:
    if ( v155 == v157 )
    {
      if ( !(_DWORD)v156 )
        goto LABEL_220;
    }
    else if ( (*(_DWORD *)((*v155 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v155 >> 1) & 3)) <= (*(_DWORD *)(v148 + 24) | (unsigned int)(v147 >> 1) & 3) )
    {
LABEL_220:
      v150 = v155[1];
      goto LABEL_206;
    }
    v155 -= 2;
    goto LABEL_220;
  }
  v150 = *(_QWORD *)(v149 + 24);
LABEL_206:
  v13 = 0.0;
  *(float *)&v221 = sub_1DBCC30(1u, 0, v9[4], v150) + 0.0;
  *(float *)&a5 = sub_1DBCC30(0, 1u, v9[4], v150);
  *(_QWORD *)&v14 = v221;
  v207 = 2;
  *(float *)&v14 = *(float *)&v221 + *(float *)&a5;
  v11 = *(unsigned int *)(a2 + 112);
  v214 = *(float *)&v221 + *(float *)&a5;
LABEL_7:
  v225 = 0;
  v226 = 0;
  v227 = &v225;
  v228 = &v225;
  v229 = 0;
  if ( (int)v11 < 0 )
    v15 = *(_QWORD *)(v215[3] + 16 * (v11 & 0x7FFFFFFF) + 8);
  else
    v15 = *(_QWORD *)(v215[34] + 8 * v11);
  v213 = 0;
  v16 = v207;
  v17 = v9;
  v194 = 0;
  v197 = 0;
  if ( v15 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v15 + 16);
      do
        v15 = *(_QWORD *)(v15 + 32);
      while ( v15 && v18 == *(_QWORD *)(v15 + 16) );
      v19 = *(_QWORD *)(v17[1] + 272LL);
      v20 = v18;
      if ( (*(_BYTE *)(v18 + 46) & 4) != 0 )
      {
        do
          v20 = *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v20 + 46) & 4) != 0 );
      }
      v21 = *(_QWORD *)(v19 + 368);
      v22 = *(unsigned int *)(v19 + 384);
      if ( !(_DWORD)v22 )
        goto LABEL_48;
      v23 = (v22 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( v20 != *v24 )
        break;
LABEL_17:
      v26 = v24[1];
      if ( v224 )
      {
        v27 = *(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v26 >> 1) & 3;
        if ( v27 < (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a3 >> 1) & 3)
          || v27 > (*(_DWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a4 >> 1) & 3) )
        {
          goto LABEL_23;
        }
      }
      ++v16;
      v28 = **(_WORD **)(v18 + 16);
      if ( v28 != 15 )
      {
        if ( v28 == 9 || (unsigned __int16)(v28 - 12) <= 1u )
          goto LABEL_23;
LABEL_50:
        v44 = v231;
        if ( v232 != v231 )
          goto LABEL_51;
        goto LABEL_91;
      }
      v43 = *(_DWORD **)(v18 + 32);
      if ( v43[2] != v43[12] )
        goto LABEL_50;
      if ( ((*v43 >> 8) & 0xFFF) == ((v43[10] >> 8) & 0xFFF) )
        goto LABEL_23;
      v44 = v231;
      if ( v232 != v231 )
        goto LABEL_51;
LABEL_91:
      v73 = &v44[HIDWORD(v233)];
      if ( v44 != v73 )
      {
        v74 = 0;
        do
        {
          if ( v18 == *v44 )
            goto LABEL_23;
          if ( *v44 == -2 )
            v74 = v44;
          ++v44;
        }
        while ( v73 != v44 );
        if ( v74 )
        {
          *v74 = v18;
          --v234;
          ++v230;
          goto LABEL_52;
        }
      }
      if ( HIDWORD(v233) < (unsigned int)v233 )
      {
        ++HIDWORD(v233);
        *v73 = v18;
        ++v230;
        goto LABEL_52;
      }
LABEL_51:
      sub_16CCBA0((__int64)&v230, v18);
      if ( !v45 )
        goto LABEL_23;
LABEL_52:
      *(_QWORD *)&a8 = dword_4530D80;
      *(_QWORD *)&a9 = LODWORD(v205);
      *(_QWORD *)&a5 = 1065353216;
      if ( INFINITY == v205 )
      {
        if ( **(_WORD **)(v18 + 16) != 15 )
          goto LABEL_23;
        goto LABEL_54;
      }
      v103 = *(_QWORD *)(v18 + 24);
      v208 = v103;
      if ( v103 == v197 )
      {
        v151 = sub_1E166B0(v18, *(_DWORD *)(a2 + 112), 0);
        v152 = HIBYTE(v151);
        *(float *)&a5 = sub_1DBCCF0(HIBYTE(v151), v151, v17[4], v18);
        if ( !v152 || !v194 )
          goto LABEL_153;
      }
      else
      {
        v104 = v17[3];
        v105 = *(_DWORD *)(v104 + 256);
        if ( !v105 )
          goto LABEL_211;
        v106 = *(_QWORD *)(v104 + 240);
        v107 = v105 - 1;
        v108 = v107 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
        v109 = (__int64 *)(v106 + 16LL * v108);
        v110 = *v109;
        if ( v103 != *v109 )
        {
          v174 = 1;
          while ( v110 != -8 )
          {
            v175 = v174 + 1;
            v108 = v107 & (v174 + v108);
            v109 = (__int64 *)(v106 + 16LL * v108);
            v110 = *v109;
            if ( v208 == *v109 )
              goto LABEL_142;
            v174 = v175;
          }
LABEL_211:
          v153 = sub_1E166B0(v18, *(_DWORD *)(a2 + 112), 0);
          *(float *)&a5 = sub_1DBCCF0(HIBYTE(v153), v153, v17[4], v18);
          v194 = 0;
          v197 = v208;
          goto LABEL_153;
        }
LABEL_142:
        v111 = v109[1];
        if ( !v111 || *(_QWORD *)(v208 + 96) == *(_QWORD *)(v208 + 88) )
          goto LABEL_211;
        v198 = v16;
        v112 = *(__int64 **)(v208 + 88);
        v195 = v15;
        v113 = v111 + 56;
        v114 = *(__int64 **)(v208 + 96);
        while ( sub_1DA1810(v113, *v112) )
        {
          if ( v114 == ++v112 )
          {
            v16 = v198;
            v15 = v195;
            goto LABEL_211;
          }
        }
        v16 = v198;
        v15 = v195;
        v115 = sub_1E166B0(v18, *(_DWORD *)(a2 + 112), 0);
        v116 = HIBYTE(v115);
        *(float *)&a5 = sub_1DBCCF0(HIBYTE(v115), v115, v17[4], v18);
        v197 = v208;
        if ( !v116 )
        {
          v194 = 1;
          goto LABEL_153;
        }
      }
      v117 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v17[1] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v197 + 48) + 8);
      v118 = v117 & 0xFFFFFFFFFFFFFFF8LL;
      v119 = (v117 >> 1) & 3;
      if ( v119 )
        v120 = (2LL * (v119 - 1)) | v118;
      else
        v120 = *(_QWORD *)v118 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v121 = (__int64 *)sub_1DB3C70((__int64 *)a2, v120);
      v194 = 1;
      *(_QWORD *)&a5 = LODWORD(a5);
      if ( v121 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
        && (*(_DWORD *)((*v121 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v121 >> 1) & 3)) <= (*(_DWORD *)((v120 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v120 >> 1) & 3) )
      {
        *(float *)&a5 = *(float *)&a5 * 3.0;
      }
LABEL_153:
      *(_QWORD *)&v14 = LODWORD(v214);
      *(float *)&v14 = v214 + *(float *)&a5;
      v214 = v214 + *(float *)&a5;
      if ( **(_WORD **)(v18 + 16) != 15 )
        goto LABEL_23;
LABEL_54:
      if ( v199 )
      {
        v46 = *(__int64 (**)())(*(_QWORD *)v202 + 256LL);
        if ( v46 == sub_1F49C80 )
          goto LABEL_23;
        *(_QWORD *)&a5 = LODWORD(a5);
        if ( !((unsigned __int8 (__fastcall *)(__int64))v46)(v202) )
          goto LABEL_23;
      }
      v47 = *(_DWORD **)(v18 + 32);
      v48 = *(_DWORD *)(a2 + 112);
      v49 = v47[2];
      v50 = (v47[10] >> 8) & 0xFFF;
      v51 = (*v47 >> 8) & 0xFFF;
      if ( v48 == v49 )
      {
        v49 = v47[12];
        v50 = (*v47 >> 8) & 0xFFF;
        v51 = (v47[10] >> 8) & 0xFFF;
      }
      if ( !v49 )
        goto LABEL_23;
      if ( v49 < 0 )
      {
        if ( v50 != v51 )
          goto LABEL_23;
        goto LABEL_65;
      }
      v52 = (_QWORD *)(*(_QWORD *)(v215[3] + 16LL * (v48 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
      v53 = *(__int64 (**)())(*(_QWORD *)v202 + 256LL);
      if ( v53 != sub_1F49C80 )
      {
        v190 = v52;
        v193 = v51;
        v165 = ((__int64 (__fastcall *)(__int64))v53)(v202);
        *(_QWORD *)&a5 = LODWORD(a5);
        v52 = v190;
        if ( v165 )
        {
          if ( v193 )
          {
            v168 = sub_38D6F10(v202 + 8, (unsigned int)v49, v193);
            v52 = v190;
            *(_QWORD *)&a5 = LODWORD(a5);
            v49 = v168;
          }
          v54 = *v52;
          v166 = (unsigned int)v49 >> 3;
          if ( (unsigned int)v166 >= *(unsigned __int16 *)(v54 + 22)
            || (v167 = *(unsigned __int8 *)(*(_QWORD *)(v54 + 8) + v166), !_bittest(&v167, v49 & 7)) )
          {
            if ( !v50 )
              goto LABEL_23;
LABEL_246:
            *(_QWORD *)&a5 = LODWORD(a5);
            v49 = sub_38D6F80(v202 + 8, (unsigned int)v49, v50, v54);
          }
          if ( !v49 )
            goto LABEL_23;
          goto LABEL_65;
        }
      }
      v54 = *v52;
      if ( v50 )
        goto LABEL_246;
      v55 = (unsigned int)v49 >> 3;
      if ( (unsigned int)v55 >= *(unsigned __int16 *)(v54 + 22) )
        goto LABEL_23;
      v56 = *(unsigned __int8 *)(*(_QWORD *)(v54 + 8) + v55);
      if ( !_bittest(&v56, v49 & 7) )
        goto LABEL_23;
LABEL_65:
      v57 = *((unsigned int *)v17 + 16);
      if ( !(_DWORD)v57 )
      {
        ++v17[5];
LABEL_291:
        v57 = (unsigned int)(2 * v57);
        sub_20E1330((__int64)(v17 + 5), v57);
        v182 = *((_DWORD *)v17 + 16);
        if ( !v182 )
        {
LABEL_311:
          ++*((_DWORD *)v17 + 14);
          BUG();
        }
        v183 = v182 - 1;
        v184 = v17[6];
        *(_QWORD *)&a5 = LODWORD(a5);
        v185 = (v182 - 1) & (37 * v49);
        v59 = (unsigned int)(*((_DWORD *)v17 + 14) + 1);
        v60 = (int *)(v184 + 8LL * v185);
        v186 = *v60;
        if ( *v60 == v49 )
          goto LABEL_264;
        v187 = 1;
        v57 = 0;
        while ( v186 != -1 )
        {
          if ( v186 == -2 && !v57 )
            v57 = (unsigned __int64)v60;
          v185 = v183 & (v187 + v185);
          v60 = (int *)(v184 + 8LL * v185);
          v186 = *v60;
          if ( *v60 == v49 )
            goto LABEL_264;
          ++v187;
        }
        goto LABEL_287;
      }
      v58 = v17[6];
      v59 = ((_DWORD)v57 - 1) & (unsigned int)(37 * v49);
      v60 = (int *)(v58 + 8 * v59);
      v61 = *v60;
      if ( *v60 == v49 )
      {
LABEL_67:
        *(_QWORD *)&a6 = (unsigned int)v60[1];
        goto LABEL_68;
      }
      v169 = 1;
      v170 = 0;
      while ( v61 != -1 )
      {
        if ( !v170 && v61 == -2 )
          v170 = v60;
        v59 = ((_DWORD)v57 - 1) & (unsigned int)(v59 + v169);
        v60 = (int *)(v58 + 8 * v59);
        v61 = *v60;
        if ( *v60 == v49 )
          goto LABEL_67;
        ++v169;
      }
      v171 = *((_DWORD *)v17 + 14);
      if ( v170 )
        v60 = v170;
      ++v17[5];
      v59 = (unsigned int)(v171 + 1);
      if ( 4 * (int)v59 >= (unsigned int)(3 * v57) )
        goto LABEL_291;
      if ( (int)v57 - *((_DWORD *)v17 + 15) - (int)v59 > (unsigned int)v57 >> 3 )
        goto LABEL_264;
      sub_20E1330((__int64)(v17 + 5), v57);
      v176 = *((_DWORD *)v17 + 16);
      if ( !v176 )
        goto LABEL_311;
      v177 = v176 - 1;
      v178 = v17[6];
      v57 = 0;
      *(_QWORD *)&a5 = LODWORD(a5);
      v179 = (v176 - 1) & (37 * v49);
      v59 = (unsigned int)(*((_DWORD *)v17 + 14) + 1);
      v180 = 1;
      v60 = (int *)(v178 + 8LL * v179);
      v181 = *v60;
      if ( *v60 == v49 )
        goto LABEL_264;
      while ( v181 != -1 )
      {
        if ( v181 == -2 && !v57 )
          v57 = (unsigned __int64)v60;
        v179 = v177 & (v180 + v179);
        v60 = (int *)(v178 + 8LL * v179);
        v181 = *v60;
        if ( *v60 == v49 )
          goto LABEL_264;
        ++v180;
      }
LABEL_287:
      if ( v57 )
        v60 = (int *)v57;
LABEL_264:
      *((_DWORD *)v17 + 14) = v59;
      if ( *v60 != -1 )
        --*((_DWORD *)v17 + 15);
      *v60 = v49;
      a6 = 0.0;
      v60[1] = 0;
LABEL_68:
      *(float *)&a5 = *(float *)&a5 + *(float *)&a6;
      v60[1] = LODWORD(a5);
      if ( v49 >= 0 )
      {
        v162 = *(_QWORD *)(*v215 + 16LL);
        v163 = *(__int64 (**)())(*(_QWORD *)v162 + 112LL);
        if ( v163 == sub_1D00B10 )
          BUG();
        if ( !*(_BYTE *)(*(_QWORD *)(((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v163)(v162, v57, v59)
                                   + 232)
                       + 8LL * (unsigned int)v49
                       + 4) )
          goto LABEL_23;
        v57 = v215[38];
        v59 = (unsigned int)v49 >> 6;
        if ( (*(_QWORD *)(v57 + 8 * v59) & (1LL << v49)) != 0 )
          goto LABEL_23;
      }
      v62 = *(__int64 (**)())(*(_QWORD *)v202 + 256LL);
      if ( v62 == sub_1F49C80
        || !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))v62)(v202, v57, v59) )
      {
        v63 = v213++;
      }
      else
      {
        v63 = v49;
      }
      v64 = v226;
      *(_QWORD *)&a5 = LODWORD(a5);
      v65 = v49 > 0;
      if ( !v226 )
      {
        v64 = &v225;
        if ( v227 != &v225 )
          goto LABEL_239;
        v70 = &v225;
        v71 = 1;
LABEL_88:
        v188 = v70;
        v189 = v65;
        v191 = v63;
        v72 = sub_22077B0(48);
        *(_QWORD *)&a5 = LODWORD(a5);
        *(_DWORD *)(v72 + 32) = v49;
        *(_BYTE *)(v72 + 40) = v189;
        *(_DWORD *)(v72 + 44) = v191;
        *(_DWORD *)(v72 + 36) = LODWORD(a5);
        sub_220F040(v71, v72, v188, &v225);
        ++v229;
        goto LABEL_23;
      }
      while ( 1 )
      {
        v68 = *((_BYTE *)v64 + 40);
        if ( v65 == v68 )
          break;
        if ( v49 > 0 && !v68 )
          goto LABEL_81;
LABEL_75:
        v66 = (int *)*((_QWORD *)v64 + 3);
        v67 = 0;
        if ( !v66 )
          goto LABEL_82;
LABEL_76:
        v64 = v66;
      }
      *(_QWORD *)&a6 = (unsigned int)v64[9];
      if ( *(float *)&a5 == *(float *)&a6 )
        v69 = v63 < v64[11];
      else
        v69 = *(float *)&a5 > *(float *)&a6;
      if ( !v69 )
        goto LABEL_75;
LABEL_81:
      v66 = (int *)*((_QWORD *)v64 + 2);
      v67 = 1;
      if ( v66 )
        goto LABEL_76;
LABEL_82:
      v70 = v64;
      if ( !v67 )
        goto LABEL_83;
      if ( v64 == v227 )
      {
        v70 = v64;
        goto LABEL_87;
      }
LABEL_239:
      v192 = v63;
      v164 = sub_220EF80(v64);
      v70 = v64;
      v65 = v49 > 0;
      v63 = v192;
      v68 = *(_BYTE *)(v164 + 40);
      *(_QWORD *)&a5 = LODWORD(a5);
      v64 = (int *)v164;
LABEL_83:
      if ( v68 == v65 )
      {
        *(_QWORD *)&a6 = (unsigned int)v64[9];
        if ( *(float *)&a5 == *(float *)&a6 )
        {
          if ( v63 > v64[11] )
            goto LABEL_86;
        }
        else if ( *(float *)&a6 > *(float *)&a5 )
        {
          goto LABEL_86;
        }
      }
      else if ( v68 == 1 && !v65 )
      {
LABEL_86:
        if ( v70 )
        {
LABEL_87:
          v71 = 1;
          if ( v70 != &v225 )
          {
            v173 = *((_BYTE *)v70 + 40);
            if ( v65 == v173 )
            {
              *(_QWORD *)&a6 = (unsigned int)v70[9];
              if ( *(float *)&a5 == *(float *)&a6 )
                LOBYTE(v71) = v63 < v70[11];
              else
                LOBYTE(v71) = *(float *)&a5 > *(float *)&a6;
              v71 = (unsigned __int8)v71;
            }
            else
            {
              v71 = (unsigned __int8)(v173 | (v49 <= 0)) ^ 1;
            }
          }
          goto LABEL_88;
        }
      }
LABEL_23:
      if ( !v15 )
      {
        v207 = v16;
        v9 = v17;
        goto LABEL_25;
      }
    }
    v42 = 1;
    while ( v25 != -8 )
    {
      v161 = v42 + 1;
      v23 = (v22 - 1) & (v42 + v23);
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v20 )
        goto LABEL_17;
      v42 = v161;
    }
LABEL_48:
    v24 = (__int64 *)(v21 + 16 * v22);
    goto LABEL_17;
  }
LABEL_25:
  v29 = *((_DWORD *)v9 + 14);
  ++v9[5];
  if ( v29 )
  {
    v92 = 4 * v29;
    v30 = *((unsigned int *)v9 + 16);
    if ( (unsigned int)(4 * v29) < 0x40 )
      v92 = 64;
    if ( v92 >= (unsigned int)v30 )
      goto LABEL_28;
    v93 = (_DWORD *)v9[6];
    v94 = v29 - 1;
    if ( v94 )
    {
      _BitScanReverse(&v94, v94);
      v95 = 1 << (33 - (v94 ^ 0x1F));
      if ( v95 < 64 )
        v95 = 64;
      if ( (_DWORD)v30 == v95 )
      {
        v9[7] = 0;
        v172 = &v93[2 * v30];
        do
        {
          if ( v93 )
            *v93 = -1;
          v93 += 2;
        }
        while ( v172 != v93 );
        goto LABEL_31;
      }
      v96 = (4 * v95 / 3u + 1) | ((unsigned __int64)(4 * v95 / 3u + 1) >> 1);
      v97 = ((v96 | (v96 >> 2)) >> 4) | v96 | (v96 >> 2) | ((((v96 | (v96 >> 2)) >> 4) | v96 | (v96 >> 2)) >> 8);
      v98 = (v97 | (v97 >> 16)) + 1;
      v99 = 8 * ((v97 | (v97 >> 16)) + 1);
    }
    else
    {
      v99 = 1024;
      v98 = 128;
    }
    j___libc_free_0(v93);
    *((_DWORD *)v9 + 16) = v98;
    v100 = (_DWORD *)sub_22077B0(v99);
    v101 = *((unsigned int *)v9 + 16);
    v9[7] = 0;
    v9[6] = v100;
    for ( i = &v100[2 * v101]; i != v100; v100 += 2 )
    {
      if ( v100 )
        *v100 = -1;
    }
  }
  else if ( *((_DWORD *)v9 + 15) )
  {
    v30 = *((unsigned int *)v9 + 16);
    if ( (unsigned int)v30 <= 0x40 )
    {
LABEL_28:
      v31 = (_DWORD *)v9[6];
      for ( j = &v31[2 * v30]; j != v31; v31 += 2 )
        *v31 = -1;
      v9[7] = 0;
      goto LABEL_31;
    }
    j___libc_free_0(v9[6]);
    v9[6] = 0;
    v9[7] = 0;
    *((_DWORD *)v9 + 16) = 0;
  }
LABEL_31:
  if ( v224 )
  {
    *(_QWORD *)&v14 = dword_4530D80;
    if ( INFINITY == v205 )
    {
LABEL_45:
      result = -1.0;
      goto LABEL_118;
    }
  }
  else
  {
    if ( v229 )
    {
      if ( !v199 && v222 )
        *(_DWORD *)(v215[26] + 40LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 16) = 0;
      if ( v227 != &v225 )
      {
        v33 = v227;
        v34 = v202;
        v35 = v215;
        v36 = v199;
        do
        {
          v37 = v33[8];
          if ( !v36 || v37 != v222 )
          {
            v38 = v35[26] + 40LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF);
            v39 = *(unsigned int *)(v38 + 16);
            if ( (unsigned int)v39 >= *(_DWORD *)(v38 + 20) )
            {
              v201 = v33;
              v204 = v33[8];
              v219 = v35[26] + 40LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF);
              sub_16CD150(v38 + 8, (const void *)(v38 + 24), 0, 4, (int)v33, v37);
              v38 = v219;
              v33 = v201;
              v37 = v204;
              v39 = *(unsigned int *)(v219 + 16);
            }
            *(_DWORD *)(*(_QWORD *)(v38 + 8) + 4 * v39) = v37;
            ++*(_DWORD *)(v38 + 16);
            v40 = *(__int64 (**)())(*(_QWORD *)v34 + 256LL);
            if ( v40 == sub_1F49C80 )
              break;
            v218 = v33;
            v142 = ((__int64 (__fastcall *)(__int64))v40)(v34);
            v33 = v218;
            if ( !v142 )
              break;
          }
          v33 = (int *)sub_220EF30(v33);
        }
        while ( v33 != &v225 );
      }
      *(_QWORD *)&a7 = LODWORD(v214);
      *(float *)&a7 = v214 * 1.01;
      v214 = v214 * 1.01;
    }
    *(_QWORD *)&v13 = dword_4530D80;
    if ( INFINITY == v205 )
      goto LABEL_45;
    v124 = v9[1];
    v125 = *(_QWORD **)a2;
    v126 = *(_QWORD *)(v124 + 272);
    v127 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 == v127 )
    {
LABEL_167:
      if ( !(unsigned __int8)sub_1DB4AC0(a2, *(__int64 **)(v124 + 432), *(unsigned int *)(v124 + 440)) )
      {
        result = -1.0;
        *(_DWORD *)(a2 + 116) = dword_4530D80;
        goto LABEL_118;
      }
    }
    else
    {
      while ( 1 )
      {
        v128 = *v125 & 0xFFFFFFFFFFFFFFF8LL;
        while ( 1 )
        {
          v128 = *(_QWORD *)(v128 + 8);
          if ( v126 + 336 == v128 )
            break;
          if ( *(_QWORD *)(v128 + 16) )
            goto LABEL_165;
        }
        v128 = *(_QWORD *)(v126 + 336);
LABEL_165:
        if ( *(_DWORD *)((v128 & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((v125[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) )
          break;
        v125 += 3;
        if ( (_QWORD *)v127 == v125 )
          goto LABEL_167;
      }
    }
  }
  v200 = 0;
  v75 = *(__int64 (__fastcall **)(double, double, double, double, double, double, double))(**(_QWORD **)(*v9 + 16LL)
                                                                                         + 40LL);
  if ( (char *)v75 != (char *)sub_1D00B00 )
    v200 = (__int64 *)v75(a5, a6, a7, a8, a9, v14, v13);
  v76 = v9[2];
  v223 = 0;
  v77 = v9[1];
  v78 = *(_DWORD *)(a2 + 112);
  v79 = v76;
  if ( v76 )
  {
    v80 = *(_DWORD *)(*(_QWORD *)(v76 + 312) + 4LL * (v78 & 0x7FFFFFFF));
    if ( !v80 )
      v80 = *(_DWORD *)(a2 + 112);
    v223 = v80;
  }
  v81 = *(__int64 **)(a2 + 64);
  v210 = v81;
  v206 = &v81[*(unsigned int *)(a2 + 72)];
  if ( v81 == v206 )
    goto LABEL_124;
  v203 = v9;
  v82 = v79;
  do
  {
    v83 = *(_QWORD *)(*v210 + 8);
    if ( (v83 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_122;
    if ( (v83 & 6) == 0 )
      goto LABEL_116;
    v84 = *(_QWORD *)((*(_QWORD *)(*v210 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v85 = *(__int16 **)(v84 + 16);
    v86 = *v85;
    if ( !v82 || v86 != 15 )
    {
LABEL_112:
      if ( v86 == 9 )
        goto LABEL_122;
      goto LABEL_113;
    }
    v122 = *v210;
    while ( 1 )
    {
      v123 = *(_DWORD **)(v84 + 32);
      if ( (*v123 & 0xFFF00) != 0 || (v123[10] & 0xFFF00) != 0 )
        break;
      if ( v78 != v123[2] )
        goto LABEL_116;
      v78 = v123[12];
      if ( v78 >= 0 )
        goto LABEL_116;
      v129 = v78 & 0x7FFFFFFF;
      v130 = v78 & 0x7FFFFFFF;
      v131 = *(_DWORD *)(*(_QWORD *)(v82 + 312) + 4 * v130);
      if ( !v131 )
        v131 = v78;
      if ( v223 != v131 )
        goto LABEL_116;
      v132 = *(unsigned int *)(v77 + 408);
      v133 = 8 * v130;
      if ( v129 >= (unsigned int)v132 || (v134 = *(_QWORD *)(*(_QWORD *)(v77 + 400) + 8 * v130)) == 0 )
      {
        v140 = v129 + 1;
        if ( (unsigned int)v132 >= v129 + 1 )
          goto LABEL_185;
        v143 = v140;
        if ( v140 < v132 )
        {
          *(_DWORD *)(v77 + 408) = v140;
          goto LABEL_185;
        }
        if ( v140 <= v132 )
        {
LABEL_185:
          v141 = *(_QWORD *)(v77 + 400);
        }
        else
        {
          if ( v140 > (unsigned __int64)*(unsigned int *)(v77 + 412) )
          {
            v196 = v122;
            sub_16CD150(v77 + 400, (const void *)(v77 + 416), v140, 8, v122, v130);
            v132 = *(unsigned int *)(v77 + 408);
            v122 = v196;
            v130 = v78 & 0x7FFFFFFF;
            v143 = v140;
          }
          v141 = *(_QWORD *)(v77 + 400);
          v144 = (_QWORD *)(v141 + 8 * v143);
          v145 = (_QWORD *)(v141 + 8 * v132);
          v146 = *(_QWORD *)(v77 + 416);
          if ( v144 != v145 )
          {
            do
              *v145++ = v146;
            while ( v144 != v145 );
            v141 = *(_QWORD *)(v77 + 400);
          }
          *(_DWORD *)(v77 + 408) = v140;
        }
        v217 = v122;
        v220 = v130;
        *(_QWORD *)(v141 + v133) = sub_1DBA290(v78);
        v134 = *(_QWORD *)(*(_QWORD *)(v77 + 400) + 8 * v220);
        sub_1DBB110((_QWORD *)v77, v134);
        v122 = v217;
      }
      v135 = *(_QWORD *)(v122 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      v136 = (__int64 *)sub_1DB3C70((__int64 *)v134, v135);
      v137 = *(_QWORD *)v134 + 24LL * *(unsigned int *)(v134 + 8);
      if ( v136 == (__int64 *)v137 )
        BUG();
      if ( (*(_DWORD *)((*v136 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v136 >> 1) & 3) > *(_DWORD *)(v135 + 24)
        || ((v122 = v136[2], v138 = *(_QWORD *)(v122 + 8), v135 != (v136[1] & 0xFFFFFFFFFFFFFFF8LL))
         || (__int64 *)v137 != v136 + 3)
        && v135 == v138 )
      {
        v138 = MEMORY[8];
        v122 = 0;
      }
      if ( (v138 & 6) == 0 )
        goto LABEL_116;
      v139 = v138 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v139 )
        BUG();
      v84 = *(_QWORD *)(v139 + 16);
      v85 = *(__int16 **)(v84 + 16);
      v86 = *v85;
      if ( *v85 != 15 )
        goto LABEL_112;
    }
    v85 = *(__int16 **)(v84 + 16);
    if ( *v85 == 9 )
      goto LABEL_122;
LABEL_113:
    if ( (*((_BYTE *)v85 + 11) & 2) == 0
      || ((v87 = *(_QWORD *)(v77 + 264), v88 = *(__int64 (**)())(*v200 + 16), v88 == sub_1E1C800)
       || (v216 = *(_QWORD *)(v77 + 264), v90 = ((__int64 (__fastcall *)(__int64 *))v88)(v200), v87 = v216, !v90))
      && !(unsigned __int8)sub_1F3B9C0(v200, v84, v87) )
    {
LABEL_116:
      v89 = (float (__fastcall *)(__int64, _QWORD, float))v203[9];
      if ( v224 )
        goto LABEL_117;
LABEL_125:
      v91 = sub_1DB4D20(a2);
      result = v214;
      v89(v91, v207, v214);
      goto LABEL_118;
    }
LABEL_122:
    ++v210;
  }
  while ( v206 != v210 );
  v9 = v203;
LABEL_124:
  v89 = (float (__fastcall *)(__int64, _QWORD, float))v9[9];
  v214 = v214 * 0.5;
  if ( !v224 )
    goto LABEL_125;
LABEL_117:
  result = v89(
             (*(_DWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a4 >> 1) & 3)
           - (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a3 >> 1) & 3),
             v207,
             v214);
LABEL_118:
  sub_20E1160((__int64)v226);
  if ( v232 != v231 )
    _libc_free((unsigned __int64)v232);
  return result;
}
