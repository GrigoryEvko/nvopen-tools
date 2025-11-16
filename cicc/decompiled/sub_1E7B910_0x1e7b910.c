// Function: sub_1E7B910
// Address: 0x1e7b910
//
__int64 __fastcall sub_1E7B910(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4, int a5, __int64 a6)
{
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 (*v9)(void); // rdx
  void (*v10)(void); // rax
  __int64 v11; // r12
  unsigned __int64 v12; // rdx
  unsigned int v13; // r14d
  unsigned int v14; // eax
  __int64 v15; // r12
  unsigned __int64 v16; // rdx
  unsigned int v17; // r14d
  unsigned int v18; // eax
  _QWORD *v19; // r14
  __int64 *v20; // r12
  __int64 *v21; // rbx
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  _QWORD *v29; // r15
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // rdi
  __int64 *v35; // rax
  __int16 v37; // ax
  __int64 v38; // r12
  __int64 v39; // rbx
  int v40; // r8d
  __int64 v41; // rax
  unsigned int v42; // r9d
  __int64 v43; // rdx
  __int64 v44; // rdi
  unsigned int v45; // ecx
  _WORD *v46; // rdx
  unsigned __int16 *v47; // rsi
  int v48; // ecx
  unsigned __int16 *v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  unsigned int v52; // ecx
  _WORD *v53; // rdx
  __int16 *v54; // rsi
  unsigned __int16 v55; // cx
  __int16 *v56; // rax
  __int16 v57; // dx
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 *v60; // r14
  __int64 v61; // r12
  __int64 *v62; // rbx
  int v63; // r8d
  unsigned int v64; // edx
  __int16 v65; // cx
  _WORD *v66; // rax
  __int16 *v67; // rdx
  unsigned __int16 v68; // cx
  __int16 *v69; // rsi
  __int16 v70; // ax
  __int64 *v71; // rax
  __int64 *v72; // rbx
  __int64 *v73; // r15
  __int64 *v74; // rax
  __int64 v75; // r12
  __int64 *v76; // rdx
  unsigned int v77; // r15d
  __int64 v78; // rax
  unsigned __int64 v79; // r10
  unsigned int v80; // r8d
  char *v81; // r9
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  int v84; // ecx
  unsigned int v85; // r11d
  unsigned int v86; // eax
  unsigned int v87; // r9d
  unsigned int v88; // r15d
  int v89; // r8d
  int v90; // r10d
  unsigned int v91; // ecx
  __int16 v92; // ax
  _WORD *v93; // rcx
  __int16 *v94; // rdx
  unsigned __int16 v95; // cx
  __int16 v96; // si
  __int16 *v97; // rax
  unsigned __int64 v98; // rax
  unsigned __int64 v99; // r15
  __int64 v100; // rax
  unsigned int v101; // r10d
  char *v102; // r11
  unsigned __int64 v103; // rdx
  unsigned int v104; // r8d
  unsigned __int64 v105; // rax
  int v106; // ecx
  unsigned int v107; // eax
  unsigned __int64 v108; // rcx
  unsigned int v109; // r15d
  unsigned __int64 v110; // rcx
  int v111; // r10d
  unsigned int v112; // ecx
  int v113; // r9d
  _WORD *v114; // rdx
  __int16 *v115; // rsi
  unsigned __int16 v116; // cx
  __int16 *v117; // rax
  __int16 v118; // dx
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 *v121; // rdi
  __int64 *v122; // rcx
  unsigned int *v123; // rbx
  __int64 v124; // rdx
  __int64 v125; // r14
  unsigned int v126; // ecx
  _WORD *v127; // rdx
  __int16 *v128; // rdi
  unsigned __int16 v129; // cx
  __int16 *v130; // rax
  __int16 v131; // dx
  unsigned __int64 *v132; // rax
  __int64 v133; // rcx
  __int64 v134; // r8
  int v135; // r9d
  __int16 *v136; // r12
  __int16 *v137; // rbx
  __int16 v138; // si
  __int64 v139; // r12
  unsigned int *v140; // r15
  unsigned int *v141; // r14
  int v142; // ebx
  char *v143; // rsi
  int v144; // r12d
  __int64 v145; // r14
  unsigned __int64 v146; // rax
  unsigned int v147; // ebx
  unsigned __int64 v148; // rdx
  unsigned int v149; // ebx
  unsigned __int64 v150; // rbx
  __int64 v151; // r9
  int v152; // eax
  unsigned int v153; // ecx
  int v154; // eax
  __int64 v155; // rdx
  unsigned __int64 v156; // rax
  unsigned int v157; // ebx
  unsigned __int64 v158; // rdx
  unsigned int v159; // ebx
  unsigned __int64 v160; // rbx
  int v161; // eax
  int v162; // eax
  __int64 v163; // rsi
  __int64 v164; // rdx
  unsigned __int64 v165; // rax
  unsigned __int64 v166; // rdx
  unsigned __int64 v167; // rax
  unsigned __int64 v168; // rdx
  __int64 v169; // rax
  int v170; // [rsp+Ch] [rbp-194h]
  _QWORD *v171; // [rsp+18h] [rbp-188h]
  __int64 *v172; // [rsp+30h] [rbp-170h]
  _QWORD *v173; // [rsp+38h] [rbp-168h]
  __int16 *v174; // [rsp+40h] [rbp-160h]
  _QWORD *v175; // [rsp+48h] [rbp-158h]
  char v176; // [rsp+51h] [rbp-14Fh]
  char v177; // [rsp+52h] [rbp-14Eh]
  unsigned __int8 v178; // [rsp+53h] [rbp-14Dh]
  int v179; // [rsp+54h] [rbp-14Ch]
  unsigned int v180; // [rsp+54h] [rbp-14Ch]
  unsigned int v181; // [rsp+54h] [rbp-14Ch]
  __int64 v182; // [rsp+58h] [rbp-148h]
  _QWORD *v183; // [rsp+60h] [rbp-140h]
  _QWORD *v184; // [rsp+68h] [rbp-138h]
  unsigned __int64 v185; // [rsp+70h] [rbp-130h]
  __int16 *v186; // [rsp+78h] [rbp-128h]
  unsigned int v187; // [rsp+80h] [rbp-120h]
  unsigned int v188; // [rsp+80h] [rbp-120h]
  unsigned int v189; // [rsp+80h] [rbp-120h]
  unsigned int v190; // [rsp+80h] [rbp-120h]
  unsigned int v191; // [rsp+88h] [rbp-118h]
  void *v192; // [rsp+88h] [rbp-118h]
  char *v193; // [rsp+88h] [rbp-118h]
  unsigned int v194; // [rsp+88h] [rbp-118h]
  unsigned int v195; // [rsp+88h] [rbp-118h]
  unsigned int v196; // [rsp+90h] [rbp-110h]
  unsigned int v197; // [rsp+90h] [rbp-110h]
  unsigned int v198; // [rsp+90h] [rbp-110h]
  __int16 s; // [rsp+98h] [rbp-108h]
  char *sb; // [rsp+98h] [rbp-108h]
  unsigned int sc; // [rsp+98h] [rbp-108h]
  unsigned int sd; // [rsp+98h] [rbp-108h]
  char *sa; // [rsp+98h] [rbp-108h]
  unsigned __int64 v204; // [rsp+A0h] [rbp-100h]
  unsigned int v205; // [rsp+A8h] [rbp-F8h]
  __int64 *v206; // [rsp+A8h] [rbp-F8h]
  __int64 v207; // [rsp+A8h] [rbp-F8h]
  __int64 v208; // [rsp+B0h] [rbp-F0h]
  unsigned int *v209; // [rsp+B0h] [rbp-F0h]
  unsigned int v210; // [rsp+B8h] [rbp-E8h]
  unsigned int v211; // [rsp+B8h] [rbp-E8h]
  __int64 v212; // [rsp+C0h] [rbp-E0h]
  __int64 v214; // [rsp+C8h] [rbp-D8h]
  __int64 v215; // [rsp+C8h] [rbp-D8h]
  char **v216; // [rsp+C8h] [rbp-D8h]
  unsigned int v217; // [rsp+C8h] [rbp-D8h]
  unsigned int *v218; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v219; // [rsp+D8h] [rbp-C8h]
  _BYTE v220[16]; // [rsp+E0h] [rbp-C0h] BYREF
  __int16 *v221; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v222; // [rsp+F8h] [rbp-A8h]
  _BYTE v223[16]; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v224; // [rsp+110h] [rbp-90h] BYREF
  char *v225; // [rsp+118h] [rbp-88h]
  unsigned __int64 v226; // [rsp+120h] [rbp-80h]
  unsigned int v227; // [rsp+128h] [rbp-78h]
  __int64 v228; // [rsp+130h] [rbp-70h] BYREF
  __int64 *v229; // [rsp+138h] [rbp-68h]
  __int64 *v230; // [rsp+140h] [rbp-60h]
  __int64 v231; // [rsp+148h] [rbp-58h]
  int v232; // [rsp+150h] [rbp-50h]
  _BYTE v233[72]; // [rsp+158h] [rbp-48h] BYREF

  v7 = (__int64 *)a2[2];
  v212 = 0;
  v8 = *v7;
  v9 = *(__int64 (**)(void))(*v7 + 112);
  if ( v9 != sub_1D00B10 )
  {
    v212 = v9();
    v8 = *(_QWORD *)a2[2];
  }
  v10 = *(void (**)(void))(v8 + 40);
  if ( (char *)v10 != (char *)sub_1D00B00 )
    v10();
  v11 = *(_QWORD *)(a1 + 248);
  v173 = (_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 232) = v212;
  if ( v11 )
  {
    memset(*(void **)(a1 + 240), 0, 8 * v11);
    v11 = *(_QWORD *)(a1 + 248);
  }
  v12 = *(unsigned int *)(v212 + 44);
  v13 = *(_DWORD *)(v212 + 44);
  if ( v12 > v11 << 6 )
  {
    v160 = (unsigned int)(v12 + 63) >> 6;
    if ( v160 < 2 * v11 )
      v160 = 2 * v11;
    a6 = (__int64)realloc(*(_QWORD *)(a1 + 240), 8 * v160, v12, a4, a5, a6);
    if ( !a6 )
    {
      if ( 8 * v160 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        a6 = 0;
      }
      else
      {
        a6 = sub_13A3880(1u);
      }
    }
    v161 = *(_DWORD *)(a1 + 256);
    *(_QWORD *)(a1 + 240) = a6;
    *(_QWORD *)(a1 + 248) = v160;
    a4 = (unsigned int)(v161 + 63) >> 6;
    if ( a4 < v160 )
    {
      v210 = (unsigned int)(v161 + 63) >> 6;
      memset((void *)(a6 + 8LL * a4), 0, 8 * (v160 - a4));
      v161 = *(_DWORD *)(a1 + 256);
      a6 = *(_QWORD *)(a1 + 240);
      a4 = v210;
    }
    v162 = v161 & 0x3F;
    if ( v162 )
    {
      v163 = a4 - 1;
      a4 = v162;
      *(_QWORD *)(a6 + 8 * v163) &= ~(-1LL << v162);
      a6 = *(_QWORD *)(a1 + 240);
    }
    v164 = *(_QWORD *)(a1 + 248) - (unsigned int)v11;
    if ( v164 )
      memset((void *)(a6 + 8LL * (unsigned int)v11), 0, 8 * v164);
  }
  v14 = *(_DWORD *)(a1 + 256);
  if ( v13 > v14 )
  {
    v158 = *(_QWORD *)(a1 + 248);
    v159 = (v14 + 63) >> 6;
    if ( v158 > v159 )
    {
      v166 = v158 - v159;
      if ( v166 )
      {
        memset((void *)(*(_QWORD *)(a1 + 240) + 8LL * v159), 0, 8 * v166);
        v14 = *(_DWORD *)(a1 + 256);
      }
    }
    a4 = v14 & 0x3F;
    if ( (v14 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL * (v159 - 1)) &= ~(-1LL << a4);
      v14 = *(_DWORD *)(a1 + 256);
    }
  }
  *(_DWORD *)(a1 + 256) = v13;
  if ( v13 < v14 )
  {
    v156 = *(_QWORD *)(a1 + 248);
    v157 = (v13 + 63) >> 6;
    if ( v156 > v157 )
    {
      v167 = v156 - v157;
      if ( v167 )
      {
        memset((void *)(*(_QWORD *)(a1 + 240) + 8LL * v157), 0, 8 * v167);
        v13 = *(_DWORD *)(a1 + 256);
      }
    }
    a4 = v13 & 0x3F;
    if ( (v13 & 0x3F) != 0 )
      *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL * (v157 - 1)) &= ~(-1LL << a4);
  }
  v15 = *(_QWORD *)(a1 + 280);
  v172 = (__int64 *)(a1 + 264);
  *(_QWORD *)(a1 + 264) = v212;
  if ( v15 )
  {
    memset(*(void **)(a1 + 272), 0, 8 * v15);
    v15 = *(_QWORD *)(a1 + 280);
  }
  v16 = *(unsigned int *)(v212 + 44);
  v17 = *(_DWORD *)(v212 + 44);
  if ( v16 > v15 << 6 )
  {
    v150 = (unsigned int)(v16 + 63) >> 6;
    if ( v150 < 2 * v15 )
      v150 = 2 * v15;
    v151 = (__int64)realloc(*(_QWORD *)(a1 + 272), 8 * v150, v16, a4, a5, a6);
    if ( !v151 )
    {
      if ( 8 * v150 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v151 = 0;
      }
      else
      {
        v151 = sub_13A3880(1u);
      }
    }
    v152 = *(_DWORD *)(a1 + 288);
    *(_QWORD *)(a1 + 272) = v151;
    *(_QWORD *)(a1 + 280) = v150;
    v153 = (unsigned int)(v152 + 63) >> 6;
    if ( v153 < v150 )
    {
      v211 = (unsigned int)(v152 + 63) >> 6;
      memset((void *)(v151 + 8LL * v153), 0, 8 * (v150 - v153));
      v152 = *(_DWORD *)(a1 + 288);
      v151 = *(_QWORD *)(a1 + 272);
      v153 = v211;
    }
    v154 = v152 & 0x3F;
    if ( v154 )
    {
      *(_QWORD *)(v151 + 8LL * (v153 - 1)) &= ~(-1LL << v154);
      v151 = *(_QWORD *)(a1 + 272);
    }
    v155 = *(_QWORD *)(a1 + 280) - (unsigned int)v15;
    if ( v155 )
      memset((void *)(v151 + 8LL * (unsigned int)v15), 0, 8 * v155);
  }
  v18 = *(_DWORD *)(a1 + 288);
  if ( v17 > v18 )
  {
    v148 = *(_QWORD *)(a1 + 280);
    v149 = (v18 + 63) >> 6;
    if ( v148 > v149 )
    {
      v168 = v148 - v149;
      if ( v168 )
      {
        memset((void *)(*(_QWORD *)(a1 + 272) + 8LL * v149), 0, 8 * v168);
        v18 = *(_DWORD *)(a1 + 288);
      }
    }
    if ( (v18 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (v149 - 1)) &= ~(-1LL << (v18 & 0x3F));
      v18 = *(_DWORD *)(a1 + 288);
    }
  }
  *(_DWORD *)(a1 + 288) = v17;
  if ( v17 < v18 )
  {
    v146 = *(_QWORD *)(a1 + 280);
    v147 = (v17 + 63) >> 6;
    if ( v146 > v147 )
    {
      v165 = v146 - v147;
      if ( v165 )
      {
        memset((void *)(*(_QWORD *)(a1 + 272) + 8LL * v147), 0, 8 * v165);
        v17 = *(_DWORD *)(a1 + 288);
      }
    }
    if ( (v17 & 0x3F) != 0 )
      *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (v147 - 1)) &= ~(-1LL << (v17 & 0x3F));
  }
  v178 = 0;
  v171 = a2 + 40;
  v175 = (_QWORD *)a2[41];
  if ( v175 == a2 + 40 )
    return v178;
  v19 = (_QWORD *)a1;
  do
  {
    v228 = 0;
    v231 = 2;
    v229 = (__int64 *)v233;
    v230 = (__int64 *)v233;
    v232 = 0;
    v20 = (__int64 *)v175[12];
    v21 = (__int64 *)v175[11];
    if ( v20 == v21 )
      goto LABEL_51;
    do
    {
      while ( 1 )
      {
        v22 = *v21;
        if ( *(_QWORD *)(*v21 + 160) == *(_QWORD *)(*v21 + 152)
          || (unsigned int)((__int64)(*(_QWORD *)(v22 + 72) - *(_QWORD *)(v22 + 64)) >> 3) != 1 )
        {
          goto LABEL_19;
        }
        v23 = v229;
        if ( v230 != v229 )
          break;
        v121 = &v229[HIDWORD(v231)];
        if ( v229 != v121 )
        {
          v122 = 0;
          while ( v22 != *v23 )
          {
            if ( *v23 == -2 )
              v122 = v23;
            if ( v121 == ++v23 )
            {
              if ( !v122 )
                goto LABEL_251;
              *v122 = v22;
              --v232;
              ++v228;
              goto LABEL_19;
            }
          }
          goto LABEL_19;
        }
LABEL_251:
        if ( HIDWORD(v231) >= (unsigned int)v231 )
          break;
        ++HIDWORD(v231);
        *v121 = v22;
        ++v228;
LABEL_19:
        if ( v20 == ++v21 )
          goto LABEL_24;
      }
      ++v21;
      sub_16CCBA0((__int64)&v228, v22);
    }
    while ( v20 != v21 );
LABEL_24:
    if ( HIDWORD(v231) == v232 )
      goto LABEL_48;
    v24 = v19[31];
    if ( v24 )
      memset((void *)v19[30], 0, 8 * v24);
    v25 = v19[35];
    if ( v25 )
      memset((void *)v19[34], 0, 8 * v25);
    v214 = v175[3];
    v26 = v214 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v214 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    v27 = *(_QWORD *)v26;
    v204 = v214 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v26 & 4) == 0 && (*(_BYTE *)(v26 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v28 = v27 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v28 + 46) & 4) == 0 )
          break;
        v27 = *(_QWORD *)v28;
      }
      v204 = v28;
    }
    v177 = 0;
    v29 = v19;
    v184 = v175 + 3;
    if ( v175 + 3 == (_QWORD *)v204 )
      goto LABEL_48;
    while ( 2 )
    {
      v30 = *(_QWORD *)v204 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v30 )
        BUG();
      v31 = *(_QWORD *)v30;
      v185 = *(_QWORD *)v204 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v30 & 4) == 0 && (*(_BYTE *)(v30 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v32 + 46) & 4) == 0 )
            break;
          v31 = *(_QWORD *)v32;
        }
        v185 = v32;
      }
      v33 = *(_QWORD *)(v204 + 16);
      if ( (unsigned __int16)(*(_WORD *)v33 - 12) <= 1u )
        goto LABEL_45;
      v37 = *(_WORD *)(v204 + 46);
      if ( (v37 & 4) != 0 || (v37 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v33 + 8) & 0x10LL) == 0 )
          goto LABEL_56;
LABEL_190:
        v34 = (unsigned __int64)v230;
        v35 = v229;
        v19 = v29;
        goto LABEL_49;
      }
      if ( sub_1E15D00(v204, 0x10u, 1) )
        goto LABEL_190;
LABEL_56:
      if ( **(_WORD **)(v204 + 16) != 15 || (v176 = sub_1E31310(*(_QWORD *)(v204 + 32))) == 0 )
      {
        sub_1E7A2D0(v204, v173, v172, v212);
        goto LABEL_45;
      }
      v38 = *(unsigned int *)(v204 + 40);
      v218 = (unsigned int *)v220;
      v219 = 0x200000000LL;
      v221 = (__int16 *)v223;
      v222 = 0x200000000LL;
      if ( !(_DWORD)v38 )
        goto LABEL_87;
      v39 = 0;
      while ( 2 )
      {
        v40 = v39;
        v41 = *(_QWORD *)(v204 + 32) + 40 * v39;
        if ( *(_BYTE *)v41 || (v42 = *(_DWORD *)(v41 + 8)) == 0 )
        {
LABEL_78:
          if ( v38 != ++v39 )
            continue;
          v174 = &v221[2 * (unsigned int)v222];
          if ( v174 == v221 )
            goto LABEL_87;
          v186 = v221;
          v182 = 0;
          v183 = v29;
          while ( 2 )
          {
            v205 = *(_DWORD *)v186;
            v59 = v230;
            if ( v230 == v229 )
              v60 = &v230[HIDWORD(v231)];
            else
              v60 = &v230[(unsigned int)v231];
            if ( v230 == v60 )
              goto LABEL_86;
            while ( 1 )
            {
              v61 = *v59;
              v62 = v59;
              if ( (unsigned __int64)*v59 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v60 == ++v59 )
                goto LABEL_86;
            }
            if ( v60 == v59 )
            {
LABEL_86:
              v29 = v183;
              goto LABEL_87;
            }
            v215 = 0;
            v208 = 24LL * v205;
LABEL_93:
            v226 = 0;
            v225 = 0;
            v227 = 0;
            v224 = v212;
            v63 = *(_DWORD *)(v212 + 44);
            if ( !v63 )
            {
LABEL_94:
              sub_2104AC0(&v224, v61);
              if ( !v224 )
                BUG();
              v64 = *(_DWORD *)(*(_QWORD *)(v224 + 8) + v208 + 16);
              v65 = v205 * (v64 & 0xF);
              v66 = (_WORD *)(*(_QWORD *)(v224 + 56) + 2LL * (v64 >> 4));
              v67 = v66 + 1;
              v68 = *v66 + v65;
LABEL_96:
              v69 = v67;
              if ( v67 )
              {
                while ( (*(_QWORD *)&v225[8 * ((unsigned __int64)v68 >> 6)] & (1LL << v68)) == 0 )
                {
                  v70 = *v69;
                  v67 = 0;
                  ++v69;
                  if ( !v70 )
                    goto LABEL_96;
                  v68 += v70;
                  if ( !v69 )
                    goto LABEL_100;
                }
                _libc_free((unsigned __int64)v225);
                if ( v215 )
                  goto LABEL_86;
                v215 = v61;
              }
              else
              {
LABEL_100:
                _libc_free((unsigned __int64)v225);
              }
              v71 = v62 + 1;
              if ( v62 + 1 == v60 )
                goto LABEL_104;
              while ( 1 )
              {
                v61 = *v71;
                v62 = v71;
                if ( (unsigned __int64)*v71 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v60 == ++v71 )
                  goto LABEL_104;
              }
              if ( v60 == v71 )
              {
LABEL_104:
                s = v205;
                if ( !v215 )
                  goto LABEL_86;
                v72 = (__int64 *)v175[11];
                v206 = (__int64 *)v175[12];
                if ( v206 != v72 )
                {
                  while ( 2 )
                  {
                    while ( 2 )
                    {
                      v74 = v229;
                      v75 = *v72;
                      if ( v230 == v229 )
                      {
                        v76 = &v229[HIDWORD(v231)];
                        if ( v229 == v76 )
                        {
                          v73 = v229;
                        }
                        else
                        {
                          do
                          {
                            if ( v75 == *v74 )
                              break;
                            ++v74;
                          }
                          while ( v76 != v74 );
                          v73 = &v229[HIDWORD(v231)];
                        }
LABEL_118:
                        while ( v76 != v74 && (unsigned __int64)*v74 >= 0xFFFFFFFFFFFFFFFELL )
                          ++v74;
                      }
                      else
                      {
                        v73 = &v230[(unsigned int)v231];
                        v74 = sub_16CC9F0((__int64)&v228, *v72);
                        if ( v75 == *v74 )
                        {
                          if ( v230 == v229 )
                            v76 = &v230[HIDWORD(v231)];
                          else
                            v76 = &v230[(unsigned int)v231];
                          goto LABEL_118;
                        }
                        if ( v230 == v229 )
                        {
                          v76 = &v230[HIDWORD(v231)];
                          v74 = v76;
                          goto LABEL_118;
                        }
                        v74 = &v230[(unsigned int)v231];
                      }
                      if ( v74 != v73 )
                      {
                        if ( v206 == ++v72 )
                          goto LABEL_149;
                        continue;
                      }
                      break;
                    }
                    v226 = 0;
                    v225 = 0;
                    v227 = 0;
                    v224 = v212;
                    v90 = *(_DWORD *)(v212 + 44);
                    if ( !v90 )
                    {
LABEL_142:
                      sub_2104AC0(&v224, v75);
                      if ( !v224 )
                        BUG();
                      v91 = *(_DWORD *)(*(_QWORD *)(v224 + 8) + v208 + 16);
                      v92 = s * (v91 & 0xF);
                      v93 = (_WORD *)(*(_QWORD *)(v224 + 56) + 2LL * (v91 >> 4));
                      v94 = v93 + 1;
                      v95 = *v93 + v92;
LABEL_146:
                      v97 = v94;
                      while ( v97 )
                      {
                        if ( (*(_QWORD *)&v225[8 * ((unsigned __int64)v95 >> 6)] & (1LL << v95)) != 0 )
                        {
                          v29 = v183;
                          _libc_free((unsigned __int64)v225);
                          goto LABEL_87;
                        }
                        v96 = *v97;
                        v94 = 0;
                        ++v97;
                        v95 += v96;
                        if ( !v96 )
                          goto LABEL_146;
                      }
                      _libc_free((unsigned __int64)v225);
                      if ( v206 == ++v72 )
                        goto LABEL_149;
                      continue;
                    }
                    break;
                  }
                  v180 = *(_DWORD *)(v212 + 44);
                  v198 = v90 + 63;
                  v99 = (unsigned int)(v90 + 63) >> 6;
                  v189 = (unsigned int)(v90 + 63) >> 6;
                  v100 = malloc(8 * v99);
                  v101 = v180;
                  v102 = (char *)v100;
                  if ( v100 )
                  {
                    v225 = (char *)v100;
                    v226 = v99;
                    if ( v189 )
                    {
                      v103 = v99;
                      v104 = 0;
                      v105 = 0;
                      v106 = 0;
                      goto LABEL_169;
                    }
LABEL_172:
                    v107 = v227;
                    if ( v101 > v227 )
                    {
                      v108 = (v227 + 63) >> 6;
                      v109 = (v227 + 63) >> 6;
                      if ( v226 > v108 && v226 != v108 )
                      {
                        v195 = v101;
                        memset(&v225[8 * v108], 0, 8 * (v226 - v108));
                        v107 = v227;
                        v101 = v195;
                      }
                      if ( (v107 & 0x3F) != 0 )
                      {
                        *(_QWORD *)&v225[8 * v109 - 8] &= ~(-1LL << (v107 & 0x3F));
                        v107 = v227;
                      }
                    }
                    v227 = v101;
                    if ( v101 < v107 )
                    {
                      v110 = v198 >> 6;
                      if ( v226 > v110 && v226 != v110 )
                      {
                        memset(&v225[8 * v110], 0, 8 * (v226 - v110));
                        LOBYTE(v101) = v227;
                      }
                      v111 = v101 & 0x3F;
                      if ( v111 )
                        *(_QWORD *)&v225[8 * (v198 >> 6) - 8] &= ~(-1LL << v111);
                    }
                    goto LABEL_142;
                  }
                  if ( 8 * v99 )
                  {
                    sub_16BD1C0("Allocation failed", 1u);
                    v102 = 0;
                    v101 = v180;
                  }
                  else
                  {
                    v120 = sub_13A3880(1u);
                    v101 = v180;
                    v102 = (char *)v120;
                  }
                  v105 = (v227 + 63) >> 6;
                  v106 = v227 & 0x3F;
                  v104 = (v227 + 63) >> 6;
                  v225 = v102;
                  v226 = v99;
                  if ( v99 > v105 )
                  {
                    v103 = v99 - v105;
                    if ( v99 != v105 )
                    {
LABEL_169:
                      v170 = v106;
                      v181 = v104;
                      v190 = v101;
                      v193 = v102;
                      memset(&v102[8 * v105], 0, 8 * v103);
                      v106 = v170;
                      v104 = v181;
                      v101 = v190;
                      v102 = v193;
                    }
                    if ( v106 )
                    {
LABEL_196:
                      *(_QWORD *)&v102[8 * v104 - 8] &= ~(-1LL << v106);
                      v99 = v226;
                      v102 = v225;
                      goto LABEL_197;
                    }
                  }
                  else
                  {
                    if ( (v227 & 0x3F) != 0 )
                      goto LABEL_196;
LABEL_197:
                    if ( !v99 )
                      goto LABEL_172;
                  }
                  v194 = v101;
                  memset(v102, 0, 8 * v99);
                  v101 = v194;
                  goto LABEL_172;
                }
LABEL_149:
                if ( v182 && v215 != v182 )
                  goto LABEL_86;
                v186 += 2;
                if ( v174 != v186 )
                {
                  v182 = v215;
                  continue;
                }
                v29 = v183;
                v123 = v218;
                v209 = &v218[(unsigned int)v219];
                if ( v218 != v209 )
                {
                  while ( 1 )
                  {
LABEL_213:
                    v124 = v183[33];
                    if ( !v124 )
                      BUG();
                    v207 = *(_QWORD *)(v204 + 32) + 40LL * *v123;
                    v125 = *(unsigned int *)(v207 + 8);
                    v126 = *(_DWORD *)(*(_QWORD *)(v124 + 8) + 24 * v125 + 16);
                    v127 = (_WORD *)(*(_QWORD *)(v124 + 56) + 2LL * (v126 >> 4));
                    v128 = v127 + 1;
                    v129 = *v127 + v125 * (v126 & 0xF);
LABEL_215:
                    v130 = v128;
                    if ( v128 )
                    {
                      while ( (*(_QWORD *)(v183[34] + 8 * ((unsigned __int64)v129 >> 6)) & (1LL << v129)) == 0 )
                      {
                        v131 = *v130;
                        v128 = 0;
                        ++v130;
                        if ( !v131 )
                          goto LABEL_215;
                        v129 += v131;
                        if ( !v130 )
                          goto LABEL_219;
                      }
                      if ( v184 != *(_QWORD **)(v204 + 8) )
                        break;
                    }
LABEL_219:
                    if ( v209 == ++v123 )
                      goto LABEL_220;
                  }
                  sa = (char *)v123;
                  v144 = *(_DWORD *)(v207 + 8);
                  v145 = *(_QWORD *)(v204 + 8);
                  while ( (unsigned int)sub_1E165A0(v145, v144, 1, v212) == -1 )
                  {
                    if ( !v145 )
                      BUG();
                    if ( (*(_BYTE *)v145 & 4) != 0 )
                    {
                      v145 = *(_QWORD *)(v145 + 8);
                      if ( v184 == (_QWORD *)v145 )
                        goto LABEL_243;
                    }
                    else
                    {
                      while ( (*(_BYTE *)(v145 + 46) & 8) != 0 )
                        v145 = *(_QWORD *)(v145 + 8);
                      v145 = *(_QWORD *)(v145 + 8);
                      if ( v184 == (_QWORD *)v145 )
                      {
LABEL_243:
                        ++v123;
                        if ( v209 != (unsigned int *)(sa + 4) )
                          goto LABEL_213;
                        goto LABEL_220;
                      }
                    }
                  }
                  sub_1E1A450(v145, v144, v212);
                  ++v123;
                  *(_BYTE *)(v207 + 3) |= 0x40u;
                  if ( v209 != (unsigned int *)(sa + 4) )
                    goto LABEL_213;
                }
LABEL_220:
                v132 = (unsigned __int64 *)sub_1DD5D10(v215);
                sub_1E79560(v204, v215, v132, v133, v134, v135);
                v136 = v221;
                v137 = &v221[2 * (unsigned int)v222];
                if ( v221 != v137 )
                {
                  do
                  {
                    v138 = *v136;
                    v136 += 2;
                    sub_1DD6540(v215, v138, -1);
                  }
                  while ( v137 != v136 );
                }
                if ( v218 == &v218[(unsigned int)v219] )
                {
LABEL_231:
                  if ( v221 != (__int16 *)v223 )
                    _libc_free((unsigned __int64)v221);
                  if ( v218 != (unsigned int *)v220 )
                    _libc_free((unsigned __int64)v218);
                  v177 = v176;
                  goto LABEL_45;
                }
                v139 = v215;
                v140 = v218;
                v141 = &v218[(unsigned int)v219];
                v216 = (char **)(v215 + 152);
                while ( 1 )
                {
LABEL_225:
                  v142 = *(_DWORD *)(*(_QWORD *)(v204 + 32) + 40LL * *v140 + 8);
                  if ( sub_1DD6670(v139, v142, -1) )
                    goto LABEL_224;
                  LOWORD(v224) = v142;
                  HIDWORD(v224) = -1;
                  v143 = *(char **)(v139 + 160);
                  if ( v143 == *(char **)(v139 + 168) )
                    break;
                  if ( v143 )
                  {
                    *(_QWORD *)v143 = v224;
                    v143 = *(char **)(v139 + 160);
                  }
                  ++v140;
                  *(_QWORD *)(v139 + 160) = v143 + 8;
                  if ( v141 == v140 )
                  {
LABEL_230:
                    v29 = v183;
                    goto LABEL_231;
                  }
                }
                sub_1D4B220(v216, v143, &v224);
LABEL_224:
                if ( v141 == ++v140 )
                  goto LABEL_230;
                goto LABEL_225;
              }
              goto LABEL_93;
            }
            break;
          }
          v77 = v63 + 63;
          v187 = *(_DWORD *)(v212 + 44);
          v191 = (unsigned int)(v63 + 63) >> 6;
          v78 = malloc(8LL * v191);
          v79 = v191;
          v80 = v187;
          v81 = (char *)v78;
          if ( v78 )
          {
            v225 = (char *)v78;
            v226 = v191;
            if ( v191 )
            {
              v82 = v191;
              v83 = 0;
              v84 = 0;
              v85 = 0;
              goto LABEL_128;
            }
LABEL_131:
            v86 = v227;
            if ( v80 > v227 )
            {
              v87 = (v227 + 63) >> 6;
              if ( v226 > v87 && v226 != v87 )
              {
                v197 = (v227 + 63) >> 6;
                sd = v80;
                memset(&v225[8 * v87], 0, 8 * (v226 - v87));
                v86 = v227;
                v87 = v197;
                v80 = sd;
              }
              if ( (v86 & 0x3F) != 0 )
              {
                *(_QWORD *)&v225[8 * v87 - 8] &= ~(-1LL << (v86 & 0x3F));
                v86 = v227;
              }
            }
            v227 = v80;
            if ( v86 > v80 )
            {
              v88 = v77 >> 6;
              if ( v226 > v88 )
              {
                v98 = v226 - v88;
                if ( v98 )
                {
                  memset(&v225[8 * v88], 0, 8 * v98);
                  LOBYTE(v80) = v227;
                }
              }
              v89 = v80 & 0x3F;
              if ( v89 )
                *(_QWORD *)&v225[8 * v88 - 8] &= ~(-1LL << v89);
            }
            goto LABEL_94;
          }
          if ( 8LL * v191 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v81 = 0;
            v79 = v191;
            v80 = v187;
          }
          else
          {
            v169 = sub_13A3880(1u);
            v80 = v187;
            v81 = (char *)v169;
            v79 = v191;
          }
          v84 = v227 & 0x3F;
          v85 = (v227 + 63) >> 6;
          v83 = v85;
          v225 = v81;
          v226 = v79;
          if ( v79 > v85 )
          {
            v82 = v79 - v85;
            if ( v79 != v85 )
            {
LABEL_128:
              v179 = v84;
              v188 = v85;
              v192 = (void *)v79;
              v196 = v80;
              sb = v81;
              memset(&v81[8 * v83], 0, 8 * v82);
              v84 = v179;
              v85 = v188;
              v79 = (unsigned __int64)v192;
              v80 = v196;
              v81 = sb;
            }
            if ( v84 )
            {
LABEL_157:
              *(_QWORD *)&v81[8 * v85 - 8] &= ~(-1LL << v84);
              v79 = v226;
              v81 = v225;
              goto LABEL_158;
            }
          }
          else
          {
            if ( (v227 & 0x3F) != 0 )
              goto LABEL_157;
LABEL_158:
            if ( !v79 )
              goto LABEL_131;
          }
          sc = v80;
          memset(v81, 0, 8 * v79);
          v80 = sc;
          goto LABEL_131;
        }
        break;
      }
      v43 = v29[29];
      if ( (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
      {
        if ( !v43 )
          BUG();
        v44 = 24LL * v42;
        v45 = *(_DWORD *)(*(_QWORD *)(v43 + 8) + v44 + 16);
        v46 = (_WORD *)(*(_QWORD *)(v43 + 56) + 2LL * (v45 >> 4));
        v48 = v42 * (v45 & 0xF);
        v47 = v46 + 1;
        LOWORD(v48) = *v46 + v48;
        while ( 1 )
        {
          v49 = v47;
          if ( !v47 )
            break;
          while ( 1 )
          {
            if ( (*(_QWORD *)(v29[30] + 8 * ((unsigned __int64)(unsigned __int16)v48 >> 6)) & (1LL << v48)) != 0 )
              goto LABEL_87;
            v50 = *v49;
            v47 = 0;
            ++v49;
            v40 = v50 + v48;
            if ( !(_WORD)v50 )
              break;
            v48 += v50;
            if ( !v49 )
              goto LABEL_69;
          }
        }
LABEL_69:
        v51 = v29[33];
        if ( !v51 )
          BUG();
        v52 = *(_DWORD *)(*(_QWORD *)(v51 + 8) + v44 + 16);
        v53 = (_WORD *)(*(_QWORD *)(v51 + 56) + 2LL * (v52 >> 4));
        v54 = v53 + 1;
        v55 = *v53 + v42 * (v52 & 0xF);
        while ( 1 )
        {
          v56 = v54;
          if ( !v54 )
            break;
          while ( 1 )
          {
            if ( (*(_QWORD *)(v29[34] + 8 * ((unsigned __int64)v55 >> 6)) & (1LL << v55)) != 0 )
              goto LABEL_87;
            v57 = *v56;
            v54 = 0;
            ++v56;
            if ( !v57 )
              break;
            v55 += v57;
            if ( !v56 )
              goto LABEL_75;
          }
        }
LABEL_75:
        v58 = (unsigned int)v222;
        if ( (unsigned int)v222 >= HIDWORD(v222) )
        {
          v217 = v42;
          sub_16CD150((__int64)&v221, v223, 0, 4, v40, v42);
          v58 = (unsigned int)v222;
          v42 = v217;
        }
        *(_DWORD *)&v221[2 * v58] = v42;
        LODWORD(v222) = v222 + 1;
        goto LABEL_78;
      }
      if ( !v43 )
        BUG();
      v112 = *(_DWORD *)(*(_QWORD *)(v43 + 8) + 24LL * v42 + 16);
      v113 = (v112 & 0xF) * v42;
      v114 = (_WORD *)(*(_QWORD *)(v43 + 56) + 2LL * (v112 >> 4));
      v115 = v114 + 1;
      v116 = v113 + *v114;
LABEL_182:
      v117 = v115;
      if ( !v115 )
      {
LABEL_186:
        v119 = (unsigned int)v219;
        if ( (unsigned int)v219 >= HIDWORD(v219) )
        {
          sub_16CD150((__int64)&v218, v220, 0, 4, v39, v113);
          v119 = (unsigned int)v219;
          v40 = v39;
        }
        v218[v119] = v40;
        LODWORD(v219) = v219 + 1;
        goto LABEL_78;
      }
      while ( (*(_QWORD *)(v29[30] + 8 * ((unsigned __int64)v116 >> 6)) & (1LL << v116)) == 0 )
      {
        v118 = *v117;
        v115 = 0;
        ++v117;
        if ( !v118 )
          goto LABEL_182;
        v116 += v118;
        if ( !v117 )
          goto LABEL_186;
      }
LABEL_87:
      sub_1E7A2D0(v204, v173, v172, v212);
      if ( v221 != (__int16 *)v223 )
        _libc_free((unsigned __int64)v221);
      if ( v218 != (unsigned int *)v220 )
        _libc_free((unsigned __int64)v218);
LABEL_45:
      if ( v184 != (_QWORD *)v185 )
      {
        v204 = v185;
        continue;
      }
      break;
    }
    v178 |= v177;
    v19 = v29;
LABEL_48:
    v34 = (unsigned __int64)v230;
    v35 = v229;
LABEL_49:
    if ( v35 != (__int64 *)v34 )
      _libc_free(v34);
LABEL_51:
    v175 = (_QWORD *)v175[1];
  }
  while ( v171 != v175 );
  return v178;
}
