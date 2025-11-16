// Function: sub_D8E7E0
// Address: 0xd8e7e0
//
__int64 __fastcall sub_D8E7E0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r14
  int *v11; // rdi
  __int64 v12; // rax
  unsigned int *v13; // rdi
  unsigned int *v14; // rsi
  unsigned int *v15; // rax
  unsigned int *v16; // rsi
  int *v17; // rax
  int *v18; // rdi
  __int64 v19; // rax
  int *v20; // rdi
  int *v21; // rsi
  int *v22; // rax
  int *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rcx
  unsigned int *v28; // rax
  unsigned int v29; // esi
  int *v30; // rax
  __int64 v31; // rdx
  int v32; // ecx
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _BOOL8 v35; // rdi
  __int64 v36; // rax
  bool v37; // zf
  __int64 v38; // rcx
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 *v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r12
  __int64 kk; // r15
  _BOOL4 v48; // r13d
  __int64 v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rdx
  __int64 *v52; // rax
  int *v53; // rdi
  __int64 v54; // rax
  __int64 *v55; // rdx
  __int64 v56; // r13
  __int64 *v57; // rax
  __int64 *v58; // rcx
  __int64 *v59; // r14
  __int64 *i; // rbx
  __int64 v61; // rax
  unsigned __int64 v62; // rsi
  __int64 v63; // r8
  __int64 v64; // r9
  unsigned int *v65; // rax
  __int64 v66; // r14
  __int64 v67; // r12
  __int64 v68; // r14
  __int64 j; // r13
  __int64 v70; // r15
  unsigned __int64 v71; // rax
  unsigned __int64 *v72; // rbx
  __int64 v73; // r12
  unsigned __int64 *v74; // r8
  unsigned __int64 *v75; // rax
  unsigned __int64 *v76; // rdx
  unsigned __int64 *k; // r13
  unsigned __int64 *v78; // r12
  unsigned int v79; // eax
  __int64 v80; // r9
  unsigned __int64 *v81; // r15
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  unsigned __int64 *v84; // r12
  __int64 v85; // r8
  unsigned int v86; // eax
  unsigned __int64 v87; // rdi
  int v88; // edx
  unsigned __int64 *v89; // rax
  __int64 m; // r12
  __int64 n; // rdx
  unsigned int v92; // eax
  __int64 *v93; // rdx
  __int64 v94; // rdi
  unsigned int *v95; // rax
  unsigned int *v96; // rdi
  __int64 v97; // rcx
  __int64 v98; // rdx
  int *v99; // rdi
  __int64 v100; // rax
  __int64 *v101; // rcx
  __int64 *v102; // rdx
  __int64 *v103; // rax
  __int64 *v104; // rdx
  int *v105; // rax
  int *v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rcx
  int *v109; // rdx
  __int64 v110; // rax
  int *v111; // rdx
  __int64 v112; // rax
  int *v113; // rax
  unsigned __int64 v114; // rsi
  int *v115; // rdi
  __int64 v116; // rcx
  __int64 v117; // rdx
  __int64 *ii; // r13
  __int64 v119; // r15
  __int64 *v120; // r12
  __int64 v121; // r13
  __int64 v122; // rdi
  unsigned int v123; // eax
  __int64 v124; // rdi
  int *v125; // rax
  unsigned __int64 v126; // rdi
  int *v127; // rsi
  __int64 v128; // rcx
  __int64 v129; // rdx
  __int64 jj; // r12
  __int64 v131; // rax
  unsigned int v132; // edi
  int *v133; // rsi
  __int64 v134; // rcx
  __int64 v135; // rdx
  _QWORD *v136; // rax
  __int64 v137; // r12
  __int64 v138; // rcx
  __int64 v139; // rdx
  __int64 *v140; // rdx
  int v141; // ecx
  __int64 v142; // rdx
  int v143; // ecx
  __int64 v144; // rsi
  __int64 v145; // rax
  _QWORD *v146; // rbx
  _QWORD *v147; // r12
  _QWORD *v148; // rdi
  unsigned __int64 *v149; // rcx
  _BYTE *v150; // rdi
  __int64 v151; // rdx
  unsigned __int64 *v152; // rax
  size_t v153; // rdx
  unsigned __int64 *v154; // r10
  unsigned int v155; // eax
  int v156; // ecx
  void *v157; // rax
  int v158; // edx
  int v159; // r9d
  int v160; // r11d
  unsigned __int64 *v161; // rcx
  unsigned int v162; // r11d
  __int64 v163; // [rsp+18h] [rbp-298h]
  __int64 v164; // [rsp+20h] [rbp-290h]
  __int64 v165; // [rsp+40h] [rbp-270h]
  __int64 v166; // [rsp+40h] [rbp-270h]
  __int64 v167; // [rsp+48h] [rbp-268h]
  __int64 v168; // [rsp+48h] [rbp-268h]
  __int64 v169; // [rsp+58h] [rbp-258h]
  int *v170; // [rsp+58h] [rbp-258h]
  __int64 v172; // [rsp+68h] [rbp-248h]
  __int64 v173; // [rsp+70h] [rbp-240h]
  __int64 v174; // [rsp+78h] [rbp-238h]
  __int64 v175; // [rsp+78h] [rbp-238h]
  _QWORD *v176; // [rsp+78h] [rbp-238h]
  __int64 v177; // [rsp+78h] [rbp-238h]
  __int64 v178; // [rsp+80h] [rbp-230h] BYREF
  unsigned int v179; // [rsp+88h] [rbp-228h]
  __int64 v180; // [rsp+90h] [rbp-220h]
  unsigned int v181; // [rsp+98h] [rbp-218h]
  unsigned __int64 *v182; // [rsp+A0h] [rbp-210h] BYREF
  unsigned int v183; // [rsp+A8h] [rbp-208h]
  __int64 v184; // [rsp+B0h] [rbp-200h]
  int v185; // [rsp+B8h] [rbp-1F8h]
  char v186; // [rsp+C0h] [rbp-1F0h] BYREF
  int v187; // [rsp+C8h] [rbp-1E8h] BYREF
  int *v188; // [rsp+D0h] [rbp-1E0h]
  int *v189; // [rsp+D8h] [rbp-1D8h]
  int *v190; // [rsp+E0h] [rbp-1D0h]
  int *v191; // [rsp+E8h] [rbp-1C8h]
  __int64 v192; // [rsp+F0h] [rbp-1C0h] BYREF
  int v193; // [rsp+F8h] [rbp-1B8h] BYREF
  int *v194; // [rsp+100h] [rbp-1B0h]
  int *v195; // [rsp+108h] [rbp-1A8h]
  int *v196; // [rsp+110h] [rbp-1A0h]
  int *v197; // [rsp+118h] [rbp-198h]
  int v198; // [rsp+128h] [rbp-188h] BYREF
  __int64 *v199; // [rsp+130h] [rbp-180h]
  int *v200; // [rsp+138h] [rbp-178h]
  __int64 *v201; // [rsp+140h] [rbp-170h]
  int *v202; // [rsp+148h] [rbp-168h]
  void *base; // [rsp+150h] [rbp-160h] BYREF
  __int64 v204; // [rsp+158h] [rbp-158h] BYREF
  __int64 *v205; // [rsp+160h] [rbp-150h] BYREF
  __int64 *v206; // [rsp+168h] [rbp-148h]
  __int64 *v207; // [rsp+170h] [rbp-140h]
  int *v208; // [rsp+178h] [rbp-138h]
  int v209; // [rsp+188h] [rbp-128h] BYREF
  __int64 v210; // [rsp+190h] [rbp-120h]
  int *v211; // [rsp+198h] [rbp-118h]
  int *v212; // [rsp+1A0h] [rbp-110h]
  __int64 v213; // [rsp+1A8h] [rbp-108h]
  int v214; // [rsp+1B0h] [rbp-100h]
  __int64 v215; // [rsp+1E0h] [rbp-D0h] BYREF
  unsigned int v216; // [rsp+1E8h] [rbp-C8h] BYREF
  unsigned int *v217; // [rsp+1F0h] [rbp-C0h]
  unsigned int *v218; // [rsp+1F8h] [rbp-B8h]
  unsigned int *v219; // [rsp+200h] [rbp-B0h]
  int *v220; // [rsp+208h] [rbp-A8h]
  __int64 v221; // [rsp+210h] [rbp-A0h] BYREF
  int v222; // [rsp+218h] [rbp-98h] BYREF
  int *v223; // [rsp+220h] [rbp-90h] BYREF
  int *v224; // [rsp+228h] [rbp-88h]
  int *v225; // [rsp+230h] [rbp-80h] BYREF
  _QWORD *v226; // [rsp+238h] [rbp-78h]
  __int64 v227; // [rsp+240h] [rbp-70h]
  unsigned int v228; // [rsp+248h] [rbp-68h]
  __int64 v229; // [rsp+250h] [rbp-60h]
  __int64 v230; // [rsp+258h] [rbp-58h]
  __int64 v231; // [rsp+260h] [rbp-50h]
  __int64 v232; // [rsp+268h] [rbp-48h]
  _BYTE *v233; // [rsp+270h] [rbp-40h]
  __int64 v234; // [rsp+278h] [rbp-38h]
  _BYTE v235[48]; // [rsp+280h] [rbp-30h] BYREF

  result = a1[6];
  if ( result )
    return result;
  v191 = 0;
  v190 = &v187;
  v189 = &v187;
  v3 = *a1;
  v187 = 0;
  v4 = *(_QWORD *)(v3 + 32);
  v188 = 0;
  v174 = v3 + 24;
  if ( v4 == v3 + 24 )
    goto LABEL_242;
  do
  {
    while ( 1 )
    {
      v5 = v4 - 56;
      if ( !v4 )
        v5 = 0;
      if ( !sub_B2FC80(v5) )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v174 == v4 )
        goto LABEL_30;
    }
    if ( !a1[3] )
      sub_4263D6(v5, a2, v6);
    v7 = ((__int64 (__fastcall *)(__int64 *, __int64))a1[4])(a1 + 1, v5);
    v9 = sub_D8C9C0(v7, v5, v8);
    v216 = 0;
    v217 = 0;
    v10 = v9;
    v218 = &v216;
    v219 = &v216;
    v220 = 0;
    v11 = *(int **)(v9 + 16);
    if ( v11 )
    {
      v12 = sub_D864A0(v11, (__int64)&v216);
      v13 = (unsigned int *)v12;
      do
      {
        v14 = (unsigned int *)v12;
        v12 = *(_QWORD *)(v12 + 16);
      }
      while ( v12 );
      v218 = v14;
      v15 = v13;
      do
      {
        v16 = v15;
        v15 = (unsigned int *)*((_QWORD *)v15 + 3);
      }
      while ( v15 );
      v219 = v16;
      v17 = *(int **)(v10 + 40);
      v217 = v13;
      v220 = v17;
    }
    v223 = 0;
    v222 = 0;
    v224 = &v222;
    v225 = &v222;
    v226 = 0;
    v18 = *(int **)(v10 + 64);
    if ( v18 )
    {
      v19 = sub_D86860(v18, (__int64)&v222);
      v20 = (int *)v19;
      do
      {
        v21 = (int *)v19;
        v19 = *(_QWORD *)(v19 + 16);
      }
      while ( v19 );
      v224 = v21;
      v22 = v20;
      do
      {
        v23 = v22;
        v22 = (int *)*((_QWORD *)v22 + 3);
      }
      while ( v22 );
      v225 = v23;
      v24 = *(_QWORD *)(v10 + 88);
      v223 = v20;
      v226 = (_QWORD *)v24;
    }
    LODWORD(v227) = *(_DWORD *)(v10 + 96);
    v25 = sub_22077B0(144);
    *(_QWORD *)(v25 + 32) = v5;
    v26 = v25;
    v27 = v25 + 48;
    v28 = v217;
    if ( v217 )
    {
      v29 = v216;
      *(_QWORD *)(v26 + 56) = v217;
      *(_DWORD *)(v26 + 48) = v29;
      *(_QWORD *)(v26 + 64) = v218;
      *(_QWORD *)(v26 + 72) = v219;
      *((_QWORD *)v28 + 1) = v27;
      v218 = &v216;
      *(_QWORD *)(v26 + 80) = v220;
      v30 = v223;
      v219 = &v216;
      v31 = v26 + 96;
      v217 = 0;
      v220 = 0;
      if ( v223 )
        goto LABEL_23;
    }
    else
    {
      v30 = v223;
      *(_DWORD *)(v26 + 48) = 0;
      v31 = v26 + 96;
      *(_QWORD *)(v26 + 56) = 0;
      *(_QWORD *)(v26 + 64) = v27;
      *(_QWORD *)(v26 + 72) = v27;
      *(_QWORD *)(v26 + 80) = 0;
      if ( v30 )
      {
LABEL_23:
        v32 = v222;
        *(_QWORD *)(v26 + 104) = v30;
        *(_DWORD *)(v26 + 96) = v32;
        *(_QWORD *)(v26 + 112) = v224;
        *(_QWORD *)(v26 + 120) = v225;
        v30[1] = v31;
        v223 = 0;
        *(_QWORD *)(v26 + 128) = v226;
        v224 = &v222;
        v225 = &v222;
        v226 = 0;
        goto LABEL_24;
      }
    }
    *(_DWORD *)(v26 + 96) = 0;
    *(_QWORD *)(v26 + 104) = 0;
    *(_QWORD *)(v26 + 112) = v31;
    *(_QWORD *)(v26 + 120) = v31;
    *(_QWORD *)(v26 + 128) = 0;
LABEL_24:
    *(_DWORD *)(v26 + 136) = v227;
    v33 = sub_D85400((__int64)&v186, (unsigned __int64 *)(v26 + 32));
    if ( v34 )
    {
      v35 = v33 || (int *)v34 == &v187 || *(_QWORD *)(v26 + 32) < *(_QWORD *)(v34 + 32);
      a2 = v26;
      sub_220F040(v35, v26, v34, &v187);
      v191 = (int *)((char *)v191 + 1);
    }
    else
    {
      sub_D85F30(*(_QWORD **)(v26 + 104));
      sub_D85E30(*(_QWORD **)(v26 + 56));
      a2 = 144;
      j_j___libc_free_0(v26, 144);
    }
    sub_D85F30(v223);
    sub_D85E30(v217);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v174 != v4 );
LABEL_30:
  if ( v188 )
  {
    v194 = v188;
    v193 = v187;
    v195 = v189;
    v196 = v190;
    *((_QWORD *)v188 + 1) = &v193;
    v188 = 0;
    v197 = v191;
    v191 = 0;
    v189 = &v187;
    v190 = &v187;
    goto LABEL_32;
  }
LABEL_242:
  v193 = 0;
  v194 = 0;
  v195 = &v193;
  v196 = &v193;
  v197 = 0;
LABEL_32:
  v36 = sub_22077B0(192);
  v163 = v36;
  if ( !v36 )
    goto LABEL_35;
  v37 = v197 == 0;
  *(_DWORD *)(v36 + 8) = 0;
  *(_QWORD *)(v36 + 16) = 0;
  v38 = a1[5];
  *(_QWORD *)(v36 + 40) = 0;
  v167 = v38;
  v164 = v36 + 8;
  *(_QWORD *)(v36 + 24) = v36 + 8;
  *(_QWORD *)(v36 + 32) = v36 + 8;
  if ( v37 )
    goto LABEL_34;
  v53 = v194;
  v198 = 0;
  v199 = 0;
  v200 = &v198;
  v201 = (__int64 *)&v198;
  v202 = 0;
  if ( v194 )
  {
    v54 = sub_D86C20(v194, (__int64)&v198);
    v55 = (__int64 *)v54;
    do
    {
      v56 = v54;
      v54 = *(_QWORD *)(v54 + 16);
    }
    while ( v54 );
    v200 = (int *)v56;
    v57 = v55;
    do
    {
      v58 = v57;
      v57 = (__int64 *)v57[3];
    }
    while ( v57 );
    v53 = v197;
    v201 = v58;
    v199 = v55;
    v202 = v197;
    if ( (int *)v56 != &v198 )
    {
      do
      {
        v59 = *(__int64 **)(v56 + 112);
        for ( i = (__int64 *)(v56 + 96); i != v59; v59 = (__int64 *)sub_220EEE0(v59) )
        {
          while ( 1 )
          {
            sub_D87E70((__int64)(v59 + 5), v167);
            if ( sub_AAF760((__int64)(v59 + 5)) )
              break;
            v59 = (__int64 *)sub_220EEE0(v59);
            if ( i == v59 )
              goto LABEL_84;
          }
          sub_D85C20(v59[17]);
          v59[17] = 0;
          v59[18] = (__int64)(v59 + 16);
          v59[19] = (__int64)(v59 + 16);
          v59[20] = 0;
        }
LABEL_84:
        v56 = sub_220EEE0(v56);
      }
      while ( (int *)v56 != &v198 );
      v53 = (int *)*((_QWORD *)v200 + 4);
    }
  }
  v61 = sub_B2F730((__int64)v53);
  v62 = (unsigned int)sub_AE2980(v61, 0)[1];
  if ( v199 )
  {
    v205 = v199;
    LODWORD(v204) = v198;
    v206 = (__int64 *)v200;
    v207 = v201;
    v199[1] = (__int64)&v204;
    v199 = 0;
    v208 = v202;
    v200 = &v198;
    v201 = (__int64 *)&v198;
    v202 = 0;
    if ( v205 )
    {
      v217 = (unsigned int *)v205;
      v216 = v204;
      v218 = (unsigned int *)v206;
      v219 = (unsigned int *)v207;
      v205[1] = (__int64)&v216;
      v205 = 0;
      v220 = v208;
      v208 = 0;
      v206 = &v204;
      v207 = &v204;
      goto LABEL_89;
    }
  }
  else
  {
    LODWORD(v204) = 0;
    v205 = 0;
    v206 = &v204;
    v207 = &v204;
    v208 = 0;
  }
  v216 = 0;
  v217 = 0;
  v218 = &v216;
  v219 = &v216;
  v220 = 0;
LABEL_89:
  sub_AADB10((__int64)&v221, v62, 1);
  v225 = 0;
  v233 = v235;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v234 = 0;
  sub_D86030(v205);
  base = &v205;
  v204 = 0x1000000000LL;
  v65 = v218;
  v66 = (__int64)v218;
  if ( v218 == &v216 )
  {
    n = (unsigned int)v234;
    if ( (_DWORD)v234 )
      goto LABEL_120;
    goto LABEL_135;
  }
  while ( 2 )
  {
    LODWORD(v204) = 0;
    v67 = *(_QWORD *)(v66 + 112);
    v175 = v66 + 96;
    if ( v67 != v66 + 96 )
    {
      v169 = v66;
      v68 = 0;
      do
      {
        for ( j = *(_QWORD *)(v67 + 144); v67 + 128 != j; j = sub_220EEE0(j) )
        {
          v70 = *(_QWORD *)(j + 32);
          if ( v68 + 1 > (unsigned __int64)HIDWORD(v204) )
          {
            v62 = (unsigned __int64)&v205;
            sub_C8D5F0((__int64)&base, &v205, v68 + 1, 8u, v63, v64);
            v68 = (unsigned int)v204;
          }
          *((_QWORD *)base + v68) = v70;
          v68 = (unsigned int)(v204 + 1);
          LODWORD(v204) = v204 + 1;
        }
        v67 = sub_220EEE0(v67);
      }
      while ( v175 != v67 );
      v71 = (unsigned int)v68;
      v72 = (unsigned __int64 *)base;
      v66 = v169;
      v73 = 8 * v71;
      if ( v71 > 1 )
      {
        v62 = v73 >> 3;
        qsort(base, v73 >> 3, 8u, (__compar_fn_t)sub_D853E0);
        v72 = (unsigned __int64 *)base;
        v73 = 8LL * (unsigned int)v204;
      }
      v74 = (unsigned __int64 *)((char *)v72 + v73);
      if ( v72 != (unsigned __int64 *)((char *)v72 + v73) )
      {
        v75 = v72;
        while ( 1 )
        {
          v76 = v75++;
          if ( v74 == v75 )
            break;
          v62 = *(v75 - 1);
          if ( v62 == *v75 )
          {
            if ( v74 != v76 )
            {
              v149 = v76 + 2;
              if ( v74 == v76 + 2 )
              {
                v73 = (char *)v75 - (char *)v72;
              }
              else
              {
                while ( 1 )
                {
                  if ( v62 != *v149 )
                  {
                    v76[1] = *v149;
                    ++v76;
                  }
                  if ( v74 == ++v149 )
                    break;
                  v62 = *v76;
                }
                v150 = v76 + 1;
                v151 = (unsigned int)v204;
                v152 = &v72[v151];
                v153 = v151 * 8 - v73;
                v73 = &v150[v153] - (_BYTE *)v72;
                if ( v74 != v152 )
                {
                  v62 = (unsigned __int64)v74;
                  memmove(v150, v74, v153);
                }
              }
            }
            break;
          }
        }
      }
      LODWORD(v204) = v73 >> 3;
      for ( k = &v72[(unsigned int)v204]; k != v72; ++*((_DWORD *)v84 + 2) )
      {
        v62 = v228;
        if ( v228 )
        {
          v78 = 0;
          v79 = (v228 - 1) & (((unsigned int)*v72 >> 9) ^ ((unsigned int)*v72 >> 4));
          v80 = 1;
          v81 = &v226[7 * v79];
          v82 = *v81;
          if ( *v72 == *v81 )
          {
LABEL_106:
            v83 = *((unsigned int *)v81 + 4);
            v84 = v81 + 1;
            v85 = *(_QWORD *)(v169 + 32);
            if ( v83 + 1 > (unsigned __int64)*((unsigned int *)v81 + 5) )
            {
              v62 = (unsigned __int64)(v81 + 3);
              v177 = *(_QWORD *)(v169 + 32);
              sub_C8D5F0((__int64)(v81 + 1), v81 + 3, v83 + 1, 8u, v85, v80);
              v83 = *((unsigned int *)v81 + 4);
              v85 = v177;
            }
            goto LABEL_108;
          }
          while ( v82 != -4096 )
          {
            if ( v78 || v82 != -8192 )
              v81 = v78;
            v160 = v80 + 1;
            v79 = (v228 - 1) & (v79 + v80);
            v80 = v79;
            v82 = v226[7 * v79];
            if ( *v72 == v82 )
            {
              v81 = &v226[7 * v79];
              goto LABEL_106;
            }
            v78 = v81;
            LODWORD(v80) = v160;
            v81 = &v226[7 * v79];
          }
          if ( !v78 )
            v78 = v81;
          v225 = (int *)((char *)v225 + 1);
          v88 = v227 + 1;
          if ( 4 * ((int)v227 + 1) < 3 * v228 )
          {
            if ( v228 - HIDWORD(v227) - v88 <= v228 >> 3 )
            {
              sub_D8CE70((__int64)&v225, v228);
              if ( !v228 )
              {
LABEL_274:
                LODWORD(v227) = v227 + 1;
                BUG();
              }
              v154 = 0;
              v155 = (v228 - 1) & (((unsigned int)*v72 >> 9) ^ ((unsigned int)*v72 >> 4));
              v78 = &v226[7 * v155];
              v62 = *v78;
              v88 = v227 + 1;
              v156 = 1;
              if ( *v78 != *v72 )
              {
                while ( v62 != -4096 )
                {
                  if ( v62 == -8192 && !v154 )
                    v154 = v78;
                  v155 = (v228 - 1) & (v155 + v156);
                  v78 = &v226[7 * v155];
                  v62 = *v78;
                  if ( *v72 == *v78 )
                    goto LABEL_113;
                  ++v156;
                }
                if ( v154 )
                  v78 = v154;
              }
            }
            goto LABEL_113;
          }
        }
        else
        {
          v225 = (int *)((char *)v225 + 1);
        }
        v62 = 2 * v228;
        sub_D8CE70((__int64)&v225, v62);
        if ( !v228 )
          goto LABEL_274;
        v86 = (v228 - 1) & (((unsigned int)*v72 >> 9) ^ ((unsigned int)*v72 >> 4));
        v78 = &v226[7 * v86];
        v87 = *v78;
        v88 = v227 + 1;
        if ( *v72 != *v78 )
        {
          v161 = 0;
          v62 = 1;
          while ( v87 != -4096 )
          {
            if ( v87 == -8192 && !v161 )
              v161 = v78;
            v162 = v62 + 1;
            v86 = (v228 - 1) & (v86 + v62);
            v62 = v86;
            v78 = &v226[7 * v86];
            v87 = *v78;
            if ( *v72 == *v78 )
              goto LABEL_113;
            v62 = v162;
          }
          if ( v161 )
            v78 = v161;
        }
LABEL_113:
        LODWORD(v227) = v88;
        if ( *v78 != -4096 )
          --HIDWORD(v227);
        *v78 = *v72;
        v89 = v78 + 3;
        v84 = v78 + 1;
        *v84 = (unsigned __int64)v89;
        v84[1] = 0x400000000LL;
        v83 = 0;
        v85 = *(_QWORD *)(v169 + 32);
LABEL_108:
        ++v72;
        *(_QWORD *)(*v84 + 8 * v83) = v85;
      }
    }
    v66 = sub_220EEE0(v66);
    if ( (unsigned int *)v66 != &v216 )
      continue;
    break;
  }
  for ( m = (__int64)v218; (unsigned int *)m != &v216; m = sub_220EEE0(m) )
  {
    v62 = *(_QWORD *)(m + 32);
    sub_D8E2A0((__int64)&v215, v62, m + 40);
  }
  for ( n = (unsigned int)v234; (_DWORD)v234; n = (unsigned int)v234 )
  {
LABEL_120:
    v62 = *(_QWORD *)&v233[8 * n - 8];
    if ( (_DWORD)v232 )
    {
      v92 = (v232 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v93 = (__int64 *)(v230 + 8LL * v92);
      v94 = *v93;
      if ( v62 == *v93 )
      {
LABEL_122:
        *v93 = -8192;
        LODWORD(v231) = v231 - 1;
        ++HIDWORD(v231);
      }
      else
      {
        v158 = 1;
        while ( v94 != -4096 )
        {
          v159 = v158 + 1;
          v92 = (v232 - 1) & (v158 + v92);
          v93 = (__int64 *)(v230 + 8LL * v92);
          v94 = *v93;
          if ( v62 == *v93 )
            goto LABEL_122;
          v158 = v159;
        }
      }
    }
    v95 = v217;
    LODWORD(v234) = v234 - 1;
    v96 = &v216;
    if ( v217 )
    {
      do
      {
        while ( 1 )
        {
          v97 = *((_QWORD *)v95 + 2);
          v98 = *((_QWORD *)v95 + 3);
          if ( *((_QWORD *)v95 + 4) >= v62 )
            break;
          v95 = (unsigned int *)*((_QWORD *)v95 + 3);
          if ( !v98 )
            goto LABEL_128;
        }
        v96 = v95;
        v95 = (unsigned int *)*((_QWORD *)v95 + 2);
      }
      while ( v97 );
LABEL_128:
      if ( v96 != &v216 && *((_QWORD *)v96 + 4) > v62 )
        v96 = &v216;
    }
    sub_D8E2A0((__int64)&v215, v62, (__int64)(v96 + 10));
  }
  if ( base != &v205 )
    _libc_free(base, v62);
  v65 = v218;
LABEL_135:
  v166 = (__int64)v65;
  if ( v65 != &v216 )
  {
    while ( 2 )
    {
      LODWORD(v204) = 0;
      v205 = 0;
      v206 = &v204;
      v207 = &v204;
      v208 = 0;
      v99 = *(int **)(v166 + 56);
      if ( v99 )
      {
        v100 = sub_D864A0(v99, (__int64)&v204);
        v101 = (__int64 *)v100;
        do
        {
          v102 = (__int64 *)v100;
          v100 = *(_QWORD *)(v100 + 16);
        }
        while ( v100 );
        v206 = v102;
        v103 = v101;
        do
        {
          v104 = v103;
          v103 = (__int64 *)v103[3];
        }
        while ( v103 );
        v207 = v104;
        v105 = *(int **)(v166 + 80);
        v205 = v101;
        v208 = v105;
      }
      v209 = 0;
      v210 = 0;
      v211 = &v209;
      v212 = &v209;
      v213 = 0;
      v106 = *(int **)(v166 + 104);
      if ( v106 )
      {
        v107 = sub_D86860(v106, (__int64)&v209);
        v108 = v107;
        do
        {
          v109 = (int *)v107;
          v107 = *(_QWORD *)(v107 + 16);
        }
        while ( v107 );
        v211 = v109;
        v110 = v108;
        do
        {
          v111 = (int *)v110;
          v110 = *(_QWORD *)(v110 + 24);
        }
        while ( v110 );
        v212 = v111;
        v112 = *(_QWORD *)(v166 + 128);
        v210 = v108;
        v213 = v112;
      }
      v214 = *(_DWORD *)(v166 + 136);
      v113 = v194;
      if ( !v194 )
      {
        v170 = &v193;
        goto LABEL_155;
      }
      v114 = *(_QWORD *)(v166 + 32);
      v115 = &v193;
      do
      {
        while ( 1 )
        {
          v116 = *((_QWORD *)v113 + 2);
          v117 = *((_QWORD *)v113 + 3);
          if ( *((_QWORD *)v113 + 4) >= v114 )
            break;
          v113 = (int *)*((_QWORD *)v113 + 3);
          if ( !v117 )
            goto LABEL_153;
        }
        v115 = v113;
        v113 = (int *)*((_QWORD *)v113 + 2);
      }
      while ( v116 );
LABEL_153:
      v170 = v115;
      if ( v115 == &v193 || *((_QWORD *)v115 + 4) > v114 )
      {
LABEL_155:
        v182 = (unsigned __int64 *)(v166 + 32);
        v170 = (int *)sub_D86090(&v192, v170, &v182);
      }
      for ( ii = v206; ii != &v204; ii = (__int64 *)sub_220EEE0(ii) )
      {
        v119 = (__int64)(ii + 5);
        sub_D87E70((__int64)(ii + 5), v167);
        v176 = ii + 16;
        if ( (__int64 *)ii[18] != ii + 16 )
        {
          v120 = ii;
          v121 = ii[18];
          do
          {
            sub_D88EE0((__int64)&v178, (__int64)&v215, *(_QWORD *)(v121 + 32), *(_DWORD *)(v121 + 40), v121 + 48);
            sub_D87290((__int64)&v182, v119, (__int64)&v178);
            if ( *((_DWORD *)v120 + 12) > 0x40u )
            {
              v124 = v120[5];
              if ( v124 )
                j_j___libc_free_0_0(v124);
            }
            v120[5] = (__int64)v182;
            *((_DWORD *)v120 + 12) = v183;
            v183 = 0;
            if ( *((_DWORD *)v120 + 16) > 0x40u && (v122 = v120[7]) != 0 )
            {
              j_j___libc_free_0_0(v122);
              v123 = v183;
              v120[7] = v184;
              *((_DWORD *)v120 + 16) = v185;
              if ( v123 > 0x40 && v182 )
                j_j___libc_free_0_0(v182);
            }
            else
            {
              v120[7] = v184;
              *((_DWORD *)v120 + 16) = v185;
            }
            if ( v181 > 0x40 && v180 )
              j_j___libc_free_0_0(v180);
            if ( v179 > 0x40 && v178 )
              j_j___libc_free_0_0(v178);
            v121 = sub_220EEE0(v121);
          }
          while ( v176 != (_QWORD *)v121 );
          ii = v120;
        }
        v125 = (int *)*((_QWORD *)v170 + 7);
        if ( v125 )
        {
          v126 = ii[4];
          v127 = v170 + 12;
          do
          {
            while ( 1 )
            {
              v128 = *((_QWORD *)v125 + 2);
              v129 = *((_QWORD *)v125 + 3);
              if ( *((_QWORD *)v125 + 4) >= v126 )
                break;
              v125 = (int *)*((_QWORD *)v125 + 3);
              if ( !v129 )
                goto LABEL_181;
            }
            v127 = v125;
            v125 = (int *)*((_QWORD *)v125 + 2);
          }
          while ( v128 );
LABEL_181:
          if ( v127 != v170 + 12 && *((_QWORD *)v127 + 4) > v126 )
            v127 = v170 + 12;
        }
        else
        {
          v127 = v170 + 12;
        }
        sub_D87B80(ii + 15, (_QWORD *)v127 + 15);
      }
      for ( jj = (__int64)v211; (int *)jj != &v209; jj = sub_220EEE0(jj) )
      {
        v131 = *((_QWORD *)v170 + 13);
        if ( v131 )
        {
          v132 = *(_DWORD *)(jj + 32);
          v133 = v170 + 24;
          do
          {
            while ( 1 )
            {
              v134 = *(_QWORD *)(v131 + 16);
              v135 = *(_QWORD *)(v131 + 24);
              if ( *(_DWORD *)(v131 + 32) >= v132 )
                break;
              v131 = *(_QWORD *)(v131 + 24);
              if ( !v135 )
                goto LABEL_191;
            }
            v133 = (int *)v131;
            v131 = *(_QWORD *)(v131 + 16);
          }
          while ( v134 );
LABEL_191:
          if ( v170 + 24 != v133 && v132 < v133[8] )
            v133 = v170 + 24;
        }
        else
        {
          v133 = v170 + 24;
        }
        sub_D87B80((_QWORD *)(jj + 120), (_QWORD *)v133 + 15);
      }
      v136 = *(_QWORD **)(v163 + 16);
      if ( v136 )
      {
        v137 = v164;
        v62 = *(_QWORD *)(v166 + 32);
        do
        {
          while ( 1 )
          {
            v138 = v136[2];
            v139 = v136[3];
            if ( v136[4] >= v62 )
              break;
            v136 = (_QWORD *)v136[3];
            if ( !v139 )
              goto LABEL_200;
          }
          v137 = (__int64)v136;
          v136 = (_QWORD *)v136[2];
        }
        while ( v138 );
LABEL_200:
        if ( v164 == v137 || *(_QWORD *)(v137 + 32) > v62 )
        {
LABEL_202:
          v62 = v137;
          v182 = (unsigned __int64 *)(v166 + 32);
          v137 = sub_D86090((_QWORD *)v163, (_QWORD *)v137, &v182);
        }
        sub_D85E30(*(_QWORD **)(v137 + 56));
        *(_QWORD *)(v137 + 56) = 0;
        *(_QWORD *)(v137 + 64) = v137 + 48;
        *(_QWORD *)(v137 + 72) = v137 + 48;
        *(_QWORD *)(v137 + 80) = 0;
        v140 = v205;
        if ( v205 )
        {
          v141 = v204;
          *(_QWORD *)(v137 + 56) = v205;
          *(_DWORD *)(v137 + 48) = v141;
          *(_QWORD *)(v137 + 64) = v206;
          *(_QWORD *)(v137 + 72) = v207;
          v140[1] = v137 + 48;
          *(_QWORD *)(v137 + 80) = v208;
          v205 = 0;
          v206 = &v204;
          v207 = &v204;
          v208 = 0;
        }
        sub_D85F30(*(_QWORD **)(v137 + 104));
        *(_QWORD *)(v137 + 104) = 0;
        *(_QWORD *)(v137 + 112) = v137 + 96;
        *(_QWORD *)(v137 + 120) = v137 + 96;
        *(_QWORD *)(v137 + 128) = 0;
        v142 = v210;
        if ( v210 )
        {
          v143 = v209;
          *(_QWORD *)(v137 + 104) = v210;
          *(_DWORD *)(v137 + 96) = v143;
          *(_QWORD *)(v137 + 112) = v211;
          *(_QWORD *)(v137 + 120) = v212;
          *(_QWORD *)(v142 + 8) = v137 + 96;
          *(_QWORD *)(v137 + 128) = v213;
          v210 = 0;
          v211 = &v209;
          v212 = &v209;
          v213 = 0;
        }
        *(_DWORD *)(v137 + 136) = v214;
        sub_D85F30(0);
        sub_D85E30(v205);
        v166 = sub_220EF30(v166);
        if ( (unsigned int *)v166 == &v216 )
          goto LABEL_208;
        continue;
      }
      break;
    }
    v137 = v164;
    goto LABEL_202;
  }
LABEL_208:
  if ( v233 != v235 )
    _libc_free(v233, v62);
  v144 = 8LL * (unsigned int)v232;
  sub_C7D6A0(v230, v144, 8);
  v145 = v228;
  if ( v228 )
  {
    v146 = v226;
    v147 = &v226[7 * v228];
    do
    {
      if ( *v146 != -8192 && *v146 != -4096 )
      {
        v148 = (_QWORD *)v146[1];
        if ( v148 != v146 + 3 )
          _libc_free(v148, v144);
      }
      v146 += 7;
    }
    while ( v147 != v146 );
    v145 = v228;
  }
  a2 = 56 * v145;
  sub_C7D6A0((__int64)v226, 56 * v145, 8);
  sub_969240((__int64 *)&v223);
  sub_969240(&v221);
  sub_D86030(v217);
  sub_D86030(v199);
LABEL_34:
  *(_QWORD *)(v163 + 48) = 0;
  *(_QWORD *)(v163 + 56) = v163 + 80;
  *(_QWORD *)(v163 + 64) = 8;
  *(_DWORD *)(v163 + 72) = 0;
  *(_BYTE *)(v163 + 76) = 1;
  *(_DWORD *)(v163 + 152) = 0;
  *(_QWORD *)(v163 + 160) = 0;
  *(_QWORD *)(v163 + 168) = v163 + 152;
  *(_QWORD *)(v163 + 176) = v163 + 152;
  *(_QWORD *)(v163 + 184) = 0;
LABEL_35:
  v39 = a1[6];
  a1[6] = v163;
  if ( v39 )
  {
    sub_D85A50(*(_QWORD *)(v39 + 160));
    if ( !*(_BYTE *)(v39 + 76) )
      _libc_free(*(_QWORD *)(v39 + 56), a2);
    sub_D86030(*(_QWORD **)(v39 + 16));
    j_j___libc_free_0(v39, 192);
  }
  sub_D86030(v194);
  v40 = a1[6];
  v165 = v40 + 8;
  v168 = *(_QWORD *)(v40 + 24);
  if ( v40 + 8 != v168 )
  {
    while ( 1 )
    {
      v172 = *(_QWORD *)(v168 + 64);
      if ( v168 + 48 != v172 )
        break;
LABEL_58:
      v168 = sub_220EEE0(v168);
      if ( v165 == v168 )
        goto LABEL_59;
    }
    while ( 2 )
    {
      v41 = *(_QWORD *)(v172 + 32);
      sub_D882F0(&v215, v41);
      v37 = (unsigned __int8)sub_AB1BB0((__int64)&v215, v172 + 40) == 0;
      v46 = a1[6];
      if ( v37 )
      {
LABEL_42:
        for ( kk = *(_QWORD *)(v172 + 96); v172 + 80 != kk; kk = sub_220EF30(kk) )
        {
          v50 = sub_D8CD70((_QWORD *)(v46 + 144), (_QWORD *)(v46 + 152), (unsigned __int64 *)(kk + 32));
          if ( v51 )
          {
            v48 = v50 || v46 + 152 == v51 || *(_QWORD *)(kk + 32) < *(_QWORD *)(v51 + 32);
            v173 = v51;
            v49 = sub_22077B0(40);
            *(_QWORD *)(v49 + 32) = *(_QWORD *)(kk + 32);
            sub_220F040(v48, v49, v173, v46 + 152);
            ++*(_QWORD *)(v46 + 184);
          }
        }
        if ( (unsigned int)v218 > 0x40 && v217 )
          j_j___libc_free_0_0(v217);
        if ( v216 > 0x40 && v215 )
          j_j___libc_free_0_0(v215);
        v172 = sub_220EEE0(v172);
        if ( v168 + 48 == v172 )
          goto LABEL_58;
        continue;
      }
      break;
    }
    if ( *(_BYTE *)(v46 + 76) )
    {
      v52 = *(__int64 **)(v46 + 56);
      v43 = *(unsigned int *)(v46 + 68);
      v42 = &v52[v43];
      if ( v52 != v42 )
      {
        while ( v41 != *v52 )
        {
          if ( v42 == ++v52 )
            goto LABEL_66;
        }
        goto LABEL_42;
      }
LABEL_66:
      if ( (unsigned int)v43 < *(_DWORD *)(v46 + 64) )
      {
        *(_DWORD *)(v46 + 68) = v43 + 1;
        *v42 = v41;
        ++*(_QWORD *)(v46 + 48);
        v46 = a1[6];
        goto LABEL_42;
      }
    }
    sub_C8CC70(v46 + 48, v41, (__int64)v42, v43, v44, v45);
    v46 = a1[6];
    goto LABEL_42;
  }
LABEL_59:
  if ( (_BYTE)qword_4F880A8 )
  {
    v157 = sub_CB72A0();
    sub_D90530(a1, v157);
  }
  sub_D86030(v188);
  return a1[6];
}
