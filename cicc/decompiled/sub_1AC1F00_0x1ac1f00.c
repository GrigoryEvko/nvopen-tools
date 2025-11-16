// Function: sub_1AC1F00
// Address: 0x1ac1f00
//
__int64 __fastcall sub_1AC1F00(
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
  __int64 *v9; // rax
  __int64 v10; // r15
  __int64 v11; // rdi
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r9
  int v16; // r8d
  __int64 v17; // rbx
  int v18; // r10d
  unsigned int v19; // edx
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  _QWORD *v26; // rax
  int v27; // ecx
  __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // r13
  int v32; // eax
  int v33; // r14d
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // r12
  __int64 v38; // r12
  _QWORD *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 *v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // rbx
  __int64 v46; // rbx
  __int64 v47; // rax
  _QWORD **v48; // rbx
  _QWORD **i; // r12
  _QWORD *v50; // r14
  __int64 *v51; // rax
  __int64 v52; // rax
  unsigned __int64 v53; // rax
  _QWORD **v54; // r12
  _QWORD **v55; // rbx
  __int64 v56; // r13
  _QWORD *v57; // rdi
  __int64 *v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // r14
  unsigned __int64 v61; // rax
  __int64 v62; // r13
  int v63; // r12d
  unsigned int v64; // ebx
  int v65; // ecx
  __int64 v66; // rdi
  unsigned int v67; // edx
  __int64 v68; // rsi
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rax
  unsigned int v72; // edi
  __int64 *v73; // rcx
  __int64 v74; // r8
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 *v78; // rax
  int v79; // eax
  __int64 v80; // rax
  __int64 *v81; // rsi
  __int64 j; // rdi
  __int64 v83; // rdx
  __int64 v84; // r8
  __int64 v85; // rsi
  int v86; // ecx
  _QWORD *v87; // rax
  int v88; // esi
  __int64 v89; // rbx
  __int64 v90; // rcx
  __int64 v91; // r12
  __int64 *v92; // rdi
  __int64 *v93; // rcx
  int v94; // r13d
  int v95; // r8d
  __int64 *v96; // r11
  int v97; // edi
  unsigned __int64 v98; // rdi
  int v99; // ebx
  unsigned __int64 v100; // r12
  _QWORD *v101; // r14
  unsigned int k; // r13d
  __int64 v103; // rax
  __int64 m; // r13
  int *v105; // r12
  int v106; // r14d
  __int64 v107; // r9
  unsigned int v108; // r15d
  __int64 v109; // rcx
  int v110; // eax
  int v111; // eax
  __int64 v112; // rsi
  __int64 v113; // r8
  int v114; // edi
  unsigned __int64 v115; // rbx
  unsigned int v116; // edx
  __int64 v117; // rcx
  unsigned __int64 v118; // rdx
  int *v119; // rax
  _BOOL4 v120; // r11d
  __int64 v121; // rax
  __int64 v122; // rdx
  int *v123; // rdi
  unsigned int v125; // edx
  __int64 v126; // r9
  int v127; // r8d
  __int64 *v128; // rsi
  int v129; // r8d
  unsigned int v130; // edx
  __int64 v131; // r9
  __int64 v132; // rax
  int v133; // esi
  int v134; // ecx
  __int64 v135; // rax
  __int64 v136; // [rsp+8h] [rbp-228h]
  __int64 v137; // [rsp+10h] [rbp-220h]
  __int64 v138; // [rsp+18h] [rbp-218h]
  _QWORD *v139; // [rsp+20h] [rbp-210h]
  __int64 v140; // [rsp+20h] [rbp-210h]
  __int64 v141; // [rsp+28h] [rbp-208h]
  __int64 v142; // [rsp+28h] [rbp-208h]
  _QWORD *v143; // [rsp+30h] [rbp-200h]
  __int64 *v144; // [rsp+38h] [rbp-1F8h]
  __int64 v145; // [rsp+38h] [rbp-1F8h]
  _BOOL4 v146; // [rsp+40h] [rbp-1F0h]
  __int64 v147; // [rsp+40h] [rbp-1F0h]
  __int64 *v148; // [rsp+48h] [rbp-1E8h]
  unsigned int v149; // [rsp+48h] [rbp-1E8h]
  __int64 v150; // [rsp+48h] [rbp-1E8h]
  __int64 v151; // [rsp+48h] [rbp-1E8h]
  __int64 v152; // [rsp+50h] [rbp-1E0h]
  int v153; // [rsp+50h] [rbp-1E0h]
  __int64 v154; // [rsp+50h] [rbp-1E0h]
  __int64 v155; // [rsp+50h] [rbp-1E0h]
  __int64 *v156; // [rsp+58h] [rbp-1D8h]
  __int64 v157; // [rsp+58h] [rbp-1D8h]
  __int64 v158; // [rsp+60h] [rbp-1D0h] BYREF
  _QWORD *v159; // [rsp+68h] [rbp-1C8h] BYREF
  __int64 v160; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v161; // [rsp+78h] [rbp-1B8h] BYREF
  __int64 v162; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v163; // [rsp+88h] [rbp-1A8h]
  __int64 v164; // [rsp+90h] [rbp-1A0h]
  unsigned int v165; // [rsp+98h] [rbp-198h]
  __int64 v166; // [rsp+A0h] [rbp-190h] BYREF
  __int64 *v167; // [rsp+A8h] [rbp-188h]
  __int64 *v168; // [rsp+B0h] [rbp-180h]
  __int64 v169; // [rsp+B8h] [rbp-178h]
  int v170; // [rsp+C0h] [rbp-170h]
  char v171; // [rsp+C8h] [rbp-168h] BYREF
  const char *v172; // [rsp+D0h] [rbp-160h] BYREF
  int v173; // [rsp+D8h] [rbp-158h] BYREF
  int *v174; // [rsp+E0h] [rbp-150h]
  int *v175; // [rsp+E8h] [rbp-148h]
  int *v176; // [rsp+F0h] [rbp-140h]
  __int64 v177; // [rsp+F8h] [rbp-138h]
  __int64 v178; // [rsp+100h] [rbp-130h] BYREF
  __int64 v179; // [rsp+108h] [rbp-128h]
  __int64 v180; // [rsp+110h] [rbp-120h]
  __int64 v181; // [rsp+118h] [rbp-118h]
  __int64 v182; // [rsp+120h] [rbp-110h]
  __int64 v183; // [rsp+128h] [rbp-108h]
  __int64 v184; // [rsp+130h] [rbp-100h]
  __int64 v185; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v186; // [rsp+148h] [rbp-E8h]
  __int64 v187; // [rsp+150h] [rbp-E0h]
  __int64 v188; // [rsp+158h] [rbp-D8h]
  __int64 v189; // [rsp+160h] [rbp-D0h]
  __int64 v190; // [rsp+168h] [rbp-C8h]
  __int64 v191; // [rsp+170h] [rbp-C0h]
  __int64 v192; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v193; // [rsp+188h] [rbp-A8h]
  __int64 v194; // [rsp+190h] [rbp-A0h]
  __int64 v195; // [rsp+198h] [rbp-98h]
  _QWORD **v196; // [rsp+1A0h] [rbp-90h]
  _QWORD **v197; // [rsp+1A8h] [rbp-88h]
  __int64 v198; // [rsp+1B0h] [rbp-80h]
  __int64 v199; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v200; // [rsp+1C8h] [rbp-68h]
  __int64 v201; // [rsp+1D0h] [rbp-60h]
  __int64 v202; // [rsp+1D8h] [rbp-58h]
  _QWORD **v203; // [rsp+1E0h] [rbp-50h]
  _QWORD **v204; // [rsp+1E8h] [rbp-48h]
  __int64 v205; // [rsp+1F0h] [rbp-40h]

  v9 = *(__int64 **)(a1 + 72);
  if ( v9 == *(__int64 **)(a1 + 80) )
    return 0;
  v10 = a1;
  v11 = *v9;
  v12 = *(_BYTE *)(v10 + 32) == 0;
  v13 = *(_QWORD *)(*v9 + 56);
  v158 = v11;
  v141 = v13;
  if ( !v12 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(v13 + 24) + 8LL) >> 8 )
    {
      v14 = *(_QWORD *)(v13 + 80);
      v15 = v13 + 72;
      if ( v14 != v13 + 72 )
      {
        v16 = *(_DWORD *)(v10 + 64);
        v17 = *(_QWORD *)(v10 + 48);
        v18 = v16 - 1;
        do
        {
          v21 = v14 - 24;
          if ( !v14 )
            v21 = 0;
          if ( v16 )
          {
            v19 = v18 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v20 = *(_QWORD *)(v17 + 8LL * v19);
            if ( v21 == v20 )
              goto LABEL_7;
            v134 = 1;
            while ( v20 != -8 )
            {
              v19 = v18 & (v134 + v19);
              v20 = *(_QWORD *)(v17 + 8LL * v19);
              if ( v21 == v20 )
                goto LABEL_7;
              ++v134;
            }
          }
          v22 = v21 + 40;
          v23 = *(_QWORD *)(v21 + 48);
          if ( v22 != v23 )
          {
            while ( 1 )
            {
              if ( !v23 )
LABEL_219:
                BUG();
              if ( *(_BYTE *)(v23 - 8) == 78 )
              {
                v24 = *(_QWORD *)(v23 - 48);
                if ( !*(_BYTE *)(v24 + 16) && (unsigned int)(*(_DWORD *)(v24 + 36) - 213) <= 1 )
                  break;
              }
              v23 = *(_QWORD *)(v23 + 8);
              if ( v22 == v23 )
                goto LABEL_7;
            }
            if ( v22 != v23 )
              return 0;
          }
LABEL_7:
          v14 = *(_QWORD *)(v14 + 8);
        }
        while ( v14 != v15 );
      }
    }
  }
  v12 = *(_QWORD *)(v10 + 16) == 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v204 = 0;
  v205 = 0;
  v159 = 0;
  v160 = 0;
  if ( !v12 )
  {
    v25 = *(_QWORD *)(v11 + 8);
    if ( v25 )
    {
      while ( 1 )
      {
        v26 = sub_1648700(v25);
        if ( (unsigned __int8)(*((_BYTE *)v26 + 16) - 25) <= 9u )
          break;
        v25 = *(_QWORD *)(v25 + 8);
        if ( !v25 )
          goto LABEL_27;
      }
LABEL_25:
      v31 = v26[5];
      v32 = *(_DWORD *)(v10 + 64);
      if ( v32 )
      {
        v27 = v32 - 1;
        v28 = *(_QWORD *)(v10 + 48);
        v29 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v30 = *(_QWORD *)(v28 + 8LL * v29);
        if ( v31 == v30 )
          goto LABEL_23;
        v133 = 1;
        while ( v30 != -8 )
        {
          v29 = v27 & (v133 + v29);
          v30 = *(_QWORD *)(v28 + 8LL * v29);
          if ( v31 == v30 )
          {
LABEL_23:
            while ( 1 )
            {
              v25 = *(_QWORD *)(v25 + 8);
              if ( !v25 )
                goto LABEL_27;
LABEL_24:
              v26 = sub_1648700(v25);
              if ( (unsigned __int8)(*((_BYTE *)v26 + 16) - 25) <= 9u )
                goto LABEL_25;
            }
          }
          ++v133;
        }
      }
      v33 = sub_13774B0(*(_QWORD *)(v10 + 24), v31, v158);
      v172 = (const char *)sub_1368AA0(*(__int64 **)(v10 + 16), v31);
      v34 = sub_16AF500((__int64 *)&v172, v33);
      sub_16AF570(&v160, v34);
      v25 = *(_QWORD *)(v25 + 8);
      if ( v25 )
        goto LABEL_24;
    }
  }
LABEL_27:
  sub_1AC0CC0(v10, &v158, a2, a3, a4, a5, a6, a7, a8, a9);
  sub_1AC15E0((_QWORD *)v10);
  v35 = v158;
  v172 = "codeRepl";
  LOWORD(v174) = 259;
  v36 = sub_157E9C0(v158);
  v143 = (_QWORD *)sub_22077B0(64);
  if ( v143 )
    sub_157FB60(v143, v36, (__int64)&v172, v141, v35);
  v172 = "newFuncRoot";
  LOWORD(v174) = 259;
  v37 = sub_157E9C0(v158);
  v139 = (_QWORD *)sub_22077B0(64);
  if ( v139 )
    sub_157FB60(v139, v37, (__int64)&v172, 0, 0);
  v38 = v158;
  v39 = sub_1648A60(56, 1u);
  v40 = (__int64)v39;
  if ( v39 )
    sub_15F8320((__int64)v39, v38, 0);
  v161 = v40;
  if ( sub_1626D20(v141) )
  {
    v41 = *(_QWORD *)(v10 + 80);
    v42 = *(__int64 **)(v10 + 72);
    v43 = (v41 - (__int64)v42) >> 5;
    v44 = (v41 - (__int64)v42) >> 3;
    if ( v43 > 0 )
    {
      v45 = &v42[4 * v43];
      while ( !sub_1ABB100(*v42, &v161)
           && !sub_1ABB100(v42[1], &v161)
           && !sub_1ABB100(v42[2], &v161)
           && !sub_1ABB100(v42[3], &v161) )
      {
        v42 += 4;
        if ( v42 == v45 )
        {
          v44 = (v41 - (__int64)v42) >> 3;
          goto LABEL_205;
        }
      }
      goto LABEL_41;
    }
LABEL_205:
    if ( v44 != 2 )
    {
      if ( v44 != 3 )
      {
        if ( v44 != 1 )
          goto LABEL_41;
        goto LABEL_208;
      }
      if ( sub_1ABB100(*v42, &v161) )
        goto LABEL_41;
      ++v42;
    }
    if ( sub_1ABB100(*v42, &v161) )
      goto LABEL_41;
    ++v42;
LABEL_208:
    sub_1ABB100(*v42, &v161);
  }
LABEL_41:
  v46 = v161;
  sub_157E9D0((__int64)(v139 + 5), v161);
  v47 = v139[5];
  *(_QWORD *)(v46 + 32) = v139 + 5;
  *(_QWORD *)(v46 + 24) = v47 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(v46 + 24) & 7LL;
  *(_QWORD *)((v47 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v46 + 24;
  v139[5] = v139[5] & 7LL | (v46 + 24);
  sub_1ABE760(v10, (__int64)&v192, (__int64)&v199, (__int64 *)&v159);
  sub_1ABF1D0(v10, (__int64)&v178, (__int64)&v185, (__int64)&v192);
  v48 = v197;
  for ( i = v196; v48 != i; ++i )
  {
    v50 = *i;
    v51 = (__int64 *)sub_157EE30((__int64)v139);
    sub_15F2240(v50, (__int64)v139, v51);
  }
  if ( v204 != v203 )
  {
    v52 = sub_1AC0A70(v10, v159);
    v53 = sub_157EBA0(v52);
    v54 = v203;
    v55 = v204;
    v56 = v53;
    while ( v55 != v54 )
    {
      v57 = *v54++;
      sub_15F22F0(v57, v56);
    }
  }
  v58 = *(__int64 **)(v10 + 72);
  v162 = 0;
  v167 = (__int64 *)&v171;
  v168 = (__int64 *)&v171;
  v59 = *(__int64 **)(v10 + 80);
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v169 = 1;
  v170 = 0;
  v144 = v59;
  v156 = v58;
  if ( v59 != v58 )
  {
    while ( 1 )
    {
      v60 = *v156;
      v61 = sub_157EBA0(*v156);
      v62 = v61;
      if ( v61 )
      {
        v63 = sub_15F4D60(v61);
        if ( v63 )
          break;
      }
LABEL_59:
      if ( v144 == ++v156 )
      {
        v79 = HIDWORD(v169) - v170;
        goto LABEL_61;
      }
    }
    v64 = 0;
    while ( 1 )
    {
      v69 = sub_15F4DF0(v62, v64);
      v70 = *(_DWORD *)(v10 + 64);
      if ( v70 )
      {
        v65 = v70 - 1;
        v66 = *(_QWORD *)(v10 + 48);
        v67 = (v70 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v68 = *(_QWORD *)(v66 + 8LL * v67);
        if ( v69 == v68 )
          goto LABEL_51;
        v95 = 1;
        while ( v68 != -8 )
        {
          v67 = v65 & (v95 + v67);
          v68 = *(_QWORD *)(v66 + 8LL * v67);
          if ( v69 == v68 )
            goto LABEL_51;
          ++v95;
        }
      }
      if ( *(_QWORD *)(v10 + 16) )
        break;
LABEL_57:
      v77 = sub_15F4DF0(v62, v64);
      v78 = v167;
      if ( v168 == v167 )
      {
        v92 = &v167[HIDWORD(v169)];
        if ( v167 == v92 )
          goto LABEL_97;
        v93 = 0;
        do
        {
          if ( v77 == *v78 )
            goto LABEL_51;
          if ( *v78 == -2 )
            v93 = v78;
          ++v78;
        }
        while ( v92 != v78 );
        if ( !v93 )
        {
LABEL_97:
          if ( HIDWORD(v169) >= (unsigned int)v169 )
            goto LABEL_58;
          ++HIDWORD(v169);
          *v92 = v77;
          ++v166;
        }
        else
        {
          *v93 = v77;
          --v170;
          ++v166;
        }
LABEL_51:
        if ( v63 == ++v64 )
          goto LABEL_59;
      }
      else
      {
LABEL_58:
        ++v64;
        sub_16CCBA0((__int64)&v166, v77);
        if ( v63 == v64 )
          goto LABEL_59;
      }
    }
    v71 = sub_15F4DF0(v62, v64);
    if ( v165 )
    {
      v72 = (v165 - 1) & (((unsigned int)v71 >> 4) ^ ((unsigned int)v71 >> 9));
      v73 = (__int64 *)(v163 + 16LL * v72);
      v74 = *v73;
      if ( v71 == *v73 )
      {
LABEL_56:
        v148 = v73;
        v152 = *(_QWORD *)(v10 + 24);
        v75 = sub_15F4DF0(v62, v64);
        LODWORD(v152) = sub_13774B0(v152, v60, v75);
        v172 = (const char *)sub_1368AA0(*(__int64 **)(v10 + 16), v60);
        v76 = sub_16AF500((__int64 *)&v172, v152);
        sub_16AF570(v148 + 1, v76);
        goto LABEL_57;
      }
      v153 = 1;
      v96 = 0;
      v149 = ((unsigned int)v71 >> 4) ^ ((unsigned int)v71 >> 9);
      while ( v74 != -8 )
      {
        if ( v74 == -16 && !v96 )
          v96 = v73;
        v72 = (v165 - 1) & (v153 + v72);
        v73 = (__int64 *)(v163 + 16LL * v72);
        v74 = *v73;
        if ( v71 == *v73 )
          goto LABEL_56;
        ++v153;
      }
      if ( v96 )
        v73 = v96;
      ++v162;
      v97 = v164 + 1;
      if ( 4 * ((int)v164 + 1) < 3 * v165 )
      {
        if ( v165 - HIDWORD(v164) - v97 > v165 >> 3 )
          goto LABEL_105;
        v155 = v71;
        sub_1956860((__int64)&v162, v165);
        if ( !v165 )
        {
LABEL_220:
          LODWORD(v164) = v164 + 1;
          BUG();
        }
        v128 = 0;
        v129 = 1;
        v130 = (v165 - 1) & v149;
        v97 = v164 + 1;
        v71 = v155;
        v73 = (__int64 *)(v163 + 16LL * v130);
        v131 = *v73;
        if ( v155 == *v73 )
          goto LABEL_105;
        while ( v131 != -8 )
        {
          if ( v131 == -16 && !v128 )
            v128 = v73;
          v130 = (v165 - 1) & (v129 + v130);
          v73 = (__int64 *)(v163 + 16LL * v130);
          v131 = *v73;
          if ( v155 == *v73 )
            goto LABEL_105;
          ++v129;
        }
        goto LABEL_170;
      }
    }
    else
    {
      ++v162;
    }
    v154 = v71;
    sub_1956860((__int64)&v162, 2 * v165);
    if ( !v165 )
      goto LABEL_220;
    v71 = v154;
    v97 = v164 + 1;
    v125 = (v165 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
    v73 = (__int64 *)(v163 + 16LL * v125);
    v126 = *v73;
    if ( v154 == *v73 )
      goto LABEL_105;
    v127 = 1;
    v128 = 0;
    while ( v126 != -8 )
    {
      if ( !v128 && v126 == -16 )
        v128 = v73;
      v125 = (v165 - 1) & (v127 + v125);
      v73 = (__int64 *)(v163 + 16LL * v125);
      v126 = *v73;
      if ( v154 == *v73 )
        goto LABEL_105;
      ++v127;
    }
LABEL_170:
    if ( v128 )
      v73 = v128;
LABEL_105:
    LODWORD(v164) = v97;
    if ( *v73 != -8 )
      --HIDWORD(v164);
    *v73 = v71;
    v73[1] = 0;
    goto LABEL_56;
  }
  v79 = 0;
LABEL_61:
  *(_DWORD *)(v10 + 96) = v79;
  v80 = sub_1ABBA40(
          v10,
          (__int64)&v178,
          (__int64)&v185,
          v158,
          (__int64)v139,
          (__int64)v143,
          v141,
          *(_QWORD ***)(v141 + 40));
  v81 = *(__int64 **)(v10 + 16);
  v138 = v80;
  if ( v81 )
  {
    sub_1368BE0((__int64)&v172, v81, v160);
    if ( (_BYTE)v173 )
      sub_15E4450(v138, (__int64)v172, 1, 0);
    sub_136C010(*(__int64 **)(v10 + 16), (__int64)v143, v160);
  }
  sub_1ABC9A0(v10, v138, (__int64)v143, (__int64)&v178, (__int64)&v185);
  sub_1ABB940(v10, v138);
  if ( (*(_BYTE *)(v141 + 18) & 8) != 0 )
  {
    v135 = sub_15E38F0(v141);
    sub_15E3D80(v138, v135);
  }
  if ( *(_QWORD *)(v10 + 16) && *(_DWORD *)(v10 + 96) > 1u )
    sub_1ABF6E0(v10, (__int64)v143, (__int64)&v162, *(_QWORD *)(v10 + 24));
  for ( j = *(_QWORD *)(v158 + 48); ; j = *(_QWORD *)(j + 8) )
  {
    if ( !j )
      goto LABEL_219;
    if ( *(_BYTE *)(j - 8) != 77 )
      break;
    if ( (*(_DWORD *)(j - 4) & 0xFFFFFFF) != 0 )
    {
      v83 = 0;
      v84 = 8LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
      do
      {
        while ( 1 )
        {
          v85 = (*(_BYTE *)(j - 1) & 0x40) != 0 ? *(_QWORD *)(j - 32) : j - 24 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
          v86 = *(_DWORD *)(v10 + 64);
          v87 = (_QWORD *)(v85 + v83 + 24LL * *(unsigned int *)(j + 32) + 8);
          if ( v86 )
            break;
LABEL_91:
          v83 += 8;
          *v87 = v139;
          if ( v84 == v83 )
            goto LABEL_92;
        }
        v88 = v86 - 1;
        v89 = *(_QWORD *)(v10 + 48);
        LODWORD(v90) = (v86 - 1) & (((unsigned int)*v87 >> 9) ^ ((unsigned int)*v87 >> 4));
        v91 = *(_QWORD *)(v89 + 8LL * (v88 & (((unsigned int)*v87 >> 9) ^ ((unsigned int)*v87 >> 4))));
        if ( *v87 != v91 )
        {
          v94 = 1;
          while ( v91 != -8 )
          {
            v90 = v88 & (unsigned int)(v90 + v94);
            v91 = *(_QWORD *)(v89 + 8 * v90);
            if ( *v87 == v91 )
              goto LABEL_78;
            ++v94;
          }
          goto LABEL_91;
        }
LABEL_78:
        v83 += 8;
      }
      while ( v84 != v83 );
    }
LABEL_92:
    ;
  }
  v98 = sub_157EBA0((__int64)v143);
  if ( v98 )
  {
    v99 = sub_15F4D60(v98);
    v100 = sub_157EBA0((__int64)v143);
    if ( (unsigned __int64)v99 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v136 = 8LL * v99;
    if ( v99 )
    {
      v137 = sub_22077B0(8LL * v99);
      v101 = (_QWORD *)v137;
      for ( k = 0; k != v99; ++k )
      {
        v103 = sub_15F4DF0(v100, k);
        if ( v101 )
          *v101 = v103;
        ++v101;
      }
      v142 = v137;
      v140 = v137 + 8LL * (unsigned int)(v99 - 1) + 8;
      while ( 1 )
      {
        for ( m = *(_QWORD *)(*(_QWORD *)v142 + 48LL); ; m = *(_QWORD *)(m + 8) )
        {
          if ( !m )
            goto LABEL_219;
          v157 = m - 24;
          if ( *(_BYTE *)(m - 8) != 77 )
            break;
          v105 = 0;
          v173 = 0;
          v174 = 0;
          v175 = &v173;
          v176 = &v173;
          v177 = 0;
          v106 = *(_DWORD *)(m - 4) & 0xFFFFFFF;
          if ( !v106 )
            continue;
          v107 = v10;
          v108 = 0;
          do
          {
            if ( (*(_BYTE *)(m - 1) & 0x40) != 0 )
              v109 = *(_QWORD *)(m - 32);
            else
              v109 = v157 - 24LL * (*(_DWORD *)(m - 4) & 0xFFFFFFF);
            v110 = *(_DWORD *)(v107 + 64);
            if ( !v110 )
              goto LABEL_176;
            v111 = v110 - 1;
            v112 = *(_QWORD *)(v107 + 48);
            v113 = 8LL * v108;
            v114 = 1;
            v115 = *(_QWORD *)(v113 + v109 + 24LL * *(unsigned int *)(m + 32) + 8);
            v116 = v111 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
            v117 = *(_QWORD *)(v112 + 8LL * v116);
            if ( v115 != v117 )
            {
              while ( v117 != -8 )
              {
                v116 = v111 & (v114 + v116);
                v117 = *(_QWORD *)(v112 + 8LL * v116);
                if ( v115 == v117 )
                  goto LABEL_125;
                ++v114;
              }
LABEL_176:
              ++v108;
              continue;
            }
LABEL_125:
            if ( v105 )
            {
              while ( 1 )
              {
                v118 = *((_QWORD *)v105 + 4);
                v119 = (int *)*((_QWORD *)v105 + 3);
                if ( v115 < v118 )
                  v119 = (int *)*((_QWORD *)v105 + 2);
                if ( !v119 )
                  break;
                v105 = v119;
              }
              if ( v115 < v118 )
              {
                if ( v105 != v175 )
                {
LABEL_181:
                  v147 = v107;
                  v132 = sub_220EF80(v105);
                  v113 = 8LL * v108;
                  v107 = v147;
                  if ( *(_QWORD *)(v132 + 32) >= v115 )
                  {
LABEL_182:
                    --v106;
                    v151 = v107;
                    sub_15F5350(v157, v108, 0);
                    v105 = v174;
                    v107 = v151;
                    continue;
                  }
                }
              }
              else if ( v115 <= v118 )
              {
                goto LABEL_182;
              }
              v120 = 1;
              if ( v105 != &v173 )
                v120 = v115 < *((_QWORD *)v105 + 4);
              goto LABEL_135;
            }
            v105 = &v173;
            if ( v175 != &v173 )
              goto LABEL_181;
            v105 = &v173;
            v120 = 1;
LABEL_135:
            v145 = v107;
            v150 = v113;
            v146 = v120;
            v121 = sub_22077B0(40);
            *(_QWORD *)(v121 + 32) = v115;
            sub_220F040(v146, v121, v105, &v173);
            ++v177;
            v107 = v145;
            if ( (*(_BYTE *)(m - 1) & 0x40) != 0 )
              v122 = *(_QWORD *)(m - 32);
            else
              v122 = v157 - 24LL * (*(_DWORD *)(m - 4) & 0xFFFFFFF);
            ++v108;
            *(_QWORD *)(v150 + v122 + 24LL * *(unsigned int *)(m + 32) + 8) = v143;
            v105 = v174;
          }
          while ( v106 != v108 );
          v10 = v107;
          while ( v105 )
          {
            sub_1ABB3B0(*((_QWORD *)v105 + 3));
            v123 = v105;
            v105 = (int *)*((_QWORD *)v105 + 2);
            j_j___libc_free_0(v123, 40);
          }
        }
        v142 += 8;
        if ( v140 == v142 )
        {
          if ( v137 )
            j_j___libc_free_0(v137, v136);
          break;
        }
      }
    }
  }
  if ( v168 != v167 )
    _libc_free((unsigned __int64)v168);
  j___libc_free_0(v163);
  if ( v203 )
    j_j___libc_free_0(v203, v205 - (_QWORD)v203);
  j___libc_free_0(v200);
  if ( v196 )
    j_j___libc_free_0(v196, v198 - (_QWORD)v196);
  j___libc_free_0(v193);
  if ( v189 )
    j_j___libc_free_0(v189, v191 - v189);
  j___libc_free_0(v186);
  if ( v182 )
    j_j___libc_free_0(v182, v184 - v182);
  j___libc_free_0(v179);
  return v138;
}
