// Function: sub_3860710
// Address: 0x3860710
//
__int64 __fastcall sub_3860710(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, double a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  unsigned int v17; // ecx
  unsigned __int64 *v18; // rax
  __int64 *v19; // rdx
  bool v20; // zf
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // esi
  __int64 v31; // rax
  int v32; // esi
  int v33; // r12d
  __int64 **v34; // r15
  __int64 **v35; // rbx
  __int64 v36; // r13
  __int64 v37; // r12
  __int64 v38; // rsi
  __int64 v39; // r9
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int8 *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // rax
  unsigned __int8 v46; // al
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 *v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  __int64 v57; // rax
  __int64 v58; // r13
  unsigned __int64 *v59; // rbx
  unsigned __int64 *v60; // r14
  __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rax
  _QWORD *v65; // rbx
  _QWORD *v66; // r14
  __int64 v67; // rax
  _QWORD *v69; // rax
  __int64 v70; // r13
  _QWORD **v71; // rax
  __int64 *v72; // rax
  __int64 v73; // rsi
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // r15
  _QWORD **v80; // rax
  __int64 *v81; // rax
  __int64 v82; // rax
  __int64 v83; // r9
  __int64 *v84; // r13
  __int64 v85; // rax
  __int64 v86; // rcx
  __int64 v87; // rsi
  unsigned __int8 *v88; // rsi
  __int64 v89; // rax
  __int64 *v90; // r12
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // rsi
  unsigned int v94; // r12d
  bool v95; // al
  __int64 v96; // rsi
  __int64 v97; // rax
  __int64 v98; // r9
  __int64 *v99; // rbx
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // r15
  unsigned __int8 *v104; // rsi
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 *v107; // r12
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // rsi
  unsigned __int8 *v111; // rsi
  __int64 v112; // rsi
  __int64 v113; // rax
  __int64 v114; // rcx
  __int64 v115; // rax
  __int64 v116; // rsi
  __int64 v117; // rdx
  unsigned __int8 *v118; // rsi
  __int64 v119; // rax
  __int64 *v120; // rbx
  __int64 v121; // rcx
  __int64 v122; // rax
  __int64 v123; // rbx
  __int64 v124; // rsi
  unsigned __int8 *v125; // rsi
  __int64 v126; // rax
  __int64 v127; // r15
  __int64 *v128; // rbx
  __int64 v129; // rax
  __int64 v130; // rcx
  __int64 v131; // rsi
  unsigned __int8 *v132; // rsi
  unsigned __int64 v133; // rax
  unsigned __int64 v134; // rax
  unsigned __int64 v135; // rdi
  __int64 v136; // rax
  __int64 v137; // rax
  unsigned int v138; // ecx
  __int64 v139; // rax
  unsigned __int64 *v140; // rdi
  unsigned __int64 *v141; // rbx
  unsigned __int64 *v142; // r13
  unsigned __int64 v143; // rcx
  unsigned __int64 v144; // rcx
  unsigned __int64 v145; // rcx
  unsigned __int64 v146; // rcx
  unsigned __int64 *v147; // rbx
  unsigned __int64 *v148; // r12
  __int64 v149; // rdx
  unsigned __int64 v150; // rdx
  unsigned __int64 v151; // rdx
  unsigned __int64 v152; // rdx
  __int64 v153; // [rsp+0h] [rbp-490h]
  __int64 *v154; // [rsp+8h] [rbp-488h]
  __int64 v155; // [rsp+20h] [rbp-470h]
  unsigned int v156; // [rsp+20h] [rbp-470h]
  __int64 v157; // [rsp+20h] [rbp-470h]
  _QWORD *v158; // [rsp+28h] [rbp-468h]
  __int64 v159; // [rsp+28h] [rbp-468h]
  _QWORD *v160; // [rsp+28h] [rbp-468h]
  __int64 v161; // [rsp+28h] [rbp-468h]
  __int64 v162; // [rsp+28h] [rbp-468h]
  __int64 v163; // [rsp+28h] [rbp-468h]
  int v164; // [rsp+28h] [rbp-468h]
  __int64 *v165; // [rsp+30h] [rbp-460h]
  __int64 *v166; // [rsp+30h] [rbp-460h]
  __int64 *v168; // [rsp+40h] [rbp-450h]
  __int64 v169; // [rsp+48h] [rbp-448h]
  __int64 v170; // [rsp+50h] [rbp-440h]
  __int64 v171; // [rsp+50h] [rbp-440h]
  __int64 *v172; // [rsp+50h] [rbp-440h]
  unsigned __int64 *v173; // [rsp+58h] [rbp-438h]
  unsigned __int64 *v174; // [rsp+58h] [rbp-438h]
  unsigned __int64 *v175; // [rsp+58h] [rbp-438h]
  unsigned __int64 *v176; // [rsp+58h] [rbp-438h]
  __int64 v177; // [rsp+58h] [rbp-438h]
  __int64 v178; // [rsp+68h] [rbp-428h] BYREF
  __int64 v179[2]; // [rsp+70h] [rbp-420h] BYREF
  __int64 *v180; // [rsp+80h] [rbp-410h]
  __int64 v181; // [rsp+88h] [rbp-408h] BYREF
  unsigned __int64 v182; // [rsp+98h] [rbp-3F8h]
  unsigned __int8 *v183[2]; // [rsp+A0h] [rbp-3F0h] BYREF
  unsigned __int64 v184; // [rsp+B0h] [rbp-3E0h]
  __int64 v185; // [rsp+B8h] [rbp-3D8h] BYREF
  unsigned __int64 v186; // [rsp+C8h] [rbp-3C8h]
  __int64 v187; // [rsp+D0h] [rbp-3C0h] BYREF
  __int64 v188; // [rsp+D8h] [rbp-3B8h]
  __int64 *v189; // [rsp+E0h] [rbp-3B0h]
  __int64 v190; // [rsp+E8h] [rbp-3A8h] BYREF
  __int64 v191; // [rsp+F0h] [rbp-3A0h]
  unsigned __int64 v192; // [rsp+F8h] [rbp-398h]
  unsigned __int64 v193; // [rsp+100h] [rbp-390h] BYREF
  __int64 v194; // [rsp+108h] [rbp-388h]
  unsigned __int64 v195; // [rsp+110h] [rbp-380h]
  unsigned __int64 v196[2]; // [rsp+118h] [rbp-378h] BYREF
  unsigned __int64 v197; // [rsp+128h] [rbp-368h]
  unsigned __int64 *v198; // [rsp+130h] [rbp-360h]
  __int64 v199; // [rsp+138h] [rbp-358h]
  _BYTE v200[384]; // [rsp+140h] [rbp-350h] BYREF
  _QWORD v201[4]; // [rsp+2C0h] [rbp-1D0h] BYREF
  _QWORD *v202; // [rsp+2E0h] [rbp-1B0h]
  __int64 v203; // [rsp+2E8h] [rbp-1A8h]
  unsigned int v204; // [rsp+2F0h] [rbp-1A0h]
  __int64 v205; // [rsp+2F8h] [rbp-198h]
  unsigned __int64 v206; // [rsp+300h] [rbp-190h]
  __int64 v207; // [rsp+308h] [rbp-188h]
  __int64 v208; // [rsp+310h] [rbp-180h]
  __int64 v209; // [rsp+318h] [rbp-178h]
  unsigned __int64 v210; // [rsp+320h] [rbp-170h]
  __int64 v211; // [rsp+328h] [rbp-168h]
  __int64 v212; // [rsp+330h] [rbp-160h]
  __int64 v213; // [rsp+338h] [rbp-158h]
  unsigned __int64 v214; // [rsp+340h] [rbp-150h]
  __int64 v215; // [rsp+348h] [rbp-148h]
  int v216; // [rsp+350h] [rbp-140h]
  __int64 v217; // [rsp+358h] [rbp-138h]
  _BYTE *v218; // [rsp+360h] [rbp-130h]
  _BYTE *v219; // [rsp+368h] [rbp-128h]
  __int64 v220; // [rsp+370h] [rbp-120h]
  int v221; // [rsp+378h] [rbp-118h]
  _BYTE v222[16]; // [rsp+380h] [rbp-110h] BYREF
  __int64 v223; // [rsp+390h] [rbp-100h]
  __int64 v224; // [rsp+398h] [rbp-F8h]
  __int64 v225; // [rsp+3A0h] [rbp-F0h]
  unsigned __int64 v226; // [rsp+3A8h] [rbp-E8h]
  __int64 v227; // [rsp+3B0h] [rbp-E0h]
  __int64 v228; // [rsp+3B8h] [rbp-D8h]
  __int16 v229; // [rsp+3C0h] [rbp-D0h]
  __int64 v230[5]; // [rsp+3C8h] [rbp-C8h] BYREF
  int v231; // [rsp+3F0h] [rbp-A0h]
  __int64 v232; // [rsp+3F8h] [rbp-98h]
  __int64 v233; // [rsp+400h] [rbp-90h]
  __int64 v234; // [rsp+408h] [rbp-88h]
  _BYTE *v235; // [rsp+410h] [rbp-80h]
  __int64 v236; // [rsp+418h] [rbp-78h]
  _BYTE v237[112]; // [rsp+420h] [rbp-70h] BYREF

  v7 = sub_157EB90(**(_QWORD **)(a1[3] + 32));
  v201[3] = 0;
  v202 = 0;
  v8 = sub_1632FA0(v7);
  v9 = *a1;
  v201[1] = v8;
  v203 = 0;
  v10 = *(_QWORD *)(v9 + 112);
  v201[2] = "induction";
  v218 = v222;
  v219 = v222;
  v201[0] = v10;
  v204 = 0;
  v205 = 0;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v216 = 0;
  v217 = 0;
  v220 = 2;
  v221 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 1;
  v11 = sub_15E0530(*(_QWORD *)(v10 + 24));
  memset(v230, 0, 24);
  v230[3] = v11;
  v235 = v237;
  v236 = 0x800000000LL;
  v170 = a1[1];
  v12 = *(__int64 **)a3;
  v13 = a1[3];
  v230[4] = 0;
  v198 = (unsigned __int64 *)v200;
  v199 = 0x400000000LL;
  v14 = *(unsigned int *)(a3 + 8);
  v231 = 0;
  v232 = 0;
  v234 = v8;
  v233 = 0;
  v165 = &v12[2 * v14];
  if ( v12 != v165 )
  {
    v15 = a2;
    v16 = v13;
    do
    {
      sub_385C5C0(v179, *v12, v16, v15, (__int64)v201, v10, a4, a5, v170);
      sub_385C5C0(v183, v12[1], v16, v15, (__int64)v201, v10, a4, a5, v170);
      v187 = 6;
      v188 = 0;
      v189 = v180;
      if ( v180 + 1 != 0 && v180 != 0 && v180 != (__int64 *)-16LL )
        sub_1649AC0((unsigned __int64 *)&v187, v179[0] & 0xFFFFFFFFFFFFFFF8LL);
      v190 = 6;
      v191 = 0;
      v192 = v182;
      if ( v182 != 0 && v182 != -8 && v182 != -16 )
        sub_1649AC0((unsigned __int64 *)&v190, v181 & 0xFFFFFFFFFFFFFFF8LL);
      v193 = 6;
      v194 = 0;
      v195 = v184;
      if ( v184 != 0 && v184 != -8 && v184 != -16 )
        sub_1649AC0(&v193, (unsigned __int64)v183[0] & 0xFFFFFFFFFFFFFFF8LL);
      v196[0] = 6;
      v196[1] = 0;
      v197 = v186;
      if ( v186 != 0 && v186 != -8 && v186 != -16 )
      {
        sub_1649AC0(v196, v185 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v186 != 0 && v186 != -8 && v186 != -16 )
          sub_1649B30(&v185);
      }
      if ( v184 != -8 && v184 != 0 && v184 != -16 )
        sub_1649B30(v183);
      if ( v182 != 0 && v182 != -8 && v182 != -16 )
        sub_1649B30(&v181);
      if ( v180 != 0 && v180 + 1 != 0 && v180 != (__int64 *)-16LL )
        sub_1649B30(v179);
      v17 = v199;
      if ( (unsigned int)v199 >= HIDWORD(v199) )
      {
        v156 = v199;
        v133 = (((((unsigned __int64)HIDWORD(v199) + 2) >> 1) | (HIDWORD(v199) + 2LL)) >> 2)
             | (((unsigned __int64)HIDWORD(v199) + 2) >> 1)
             | (HIDWORD(v199) + 2LL);
        v134 = (((v133 >> 4) | v133) >> 8) | (v133 >> 4) | v133;
        v135 = (v134 | (v134 >> 16) | HIDWORD(v134)) + 1;
        v136 = 0xFFFFFFFFLL;
        if ( v135 <= 0xFFFFFFFF )
          v136 = v135;
        v164 = v136;
        v137 = malloc(96 * v136);
        v138 = v156;
        v173 = (unsigned __int64 *)v137;
        if ( !v137 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v138 = v199;
        }
        v139 = 12LL * v138;
        v140 = &v198[v139];
        if ( v198 != &v198[v139] )
        {
          v157 = v16;
          v154 = v12;
          v153 = v15;
          v141 = v173;
          v142 = v198;
          do
          {
            if ( v141 )
            {
              *v141 = 6;
              v141[1] = 0;
              v143 = v142[2];
              v141[2] = v143;
              if ( v143 != -8 && v143 != 0 && v143 != -16 )
                sub_1649AC0(v141, *v142 & 0xFFFFFFFFFFFFFFF8LL);
              v141[3] = 6;
              v141[4] = 0;
              v144 = v142[5];
              v141[5] = v144;
              if ( v144 != 0 && v144 != -8 && v144 != -16 )
                sub_1649AC0(v141 + 3, v142[3] & 0xFFFFFFFFFFFFFFF8LL);
              v141[6] = 6;
              v141[7] = 0;
              v145 = v142[8];
              v141[8] = v145;
              if ( v145 != 0 && v145 != -8 && v145 != -16 )
                sub_1649AC0(v141 + 6, v142[6] & 0xFFFFFFFFFFFFFFF8LL);
              v141[9] = 6;
              v141[10] = 0;
              v146 = v142[11];
              v141[11] = v146;
              if ( v146 != 0 && v146 != -8 && v146 != -16 )
                sub_1649AC0(v141 + 9, v142[9] & 0xFFFFFFFFFFFFFFF8LL);
            }
            v142 += 12;
            v141 += 12;
          }
          while ( v140 != v142 );
          v140 = v198;
          v12 = v154;
          v15 = v153;
          if ( &v198[12 * (unsigned int)v199] != v198 )
          {
            v147 = &v198[12 * (unsigned int)v199];
            v148 = v198;
            do
            {
              v149 = *(v147 - 1);
              v147 -= 12;
              if ( v149 != 0 && v149 != -8 && v149 != -16 )
                sub_1649B30(v147 + 9);
              v150 = v147[8];
              if ( v150 != -8 && v150 != 0 && v150 != -16 )
                sub_1649B30(v147 + 6);
              v151 = v147[5];
              if ( v151 != -8 && v151 != 0 && v151 != -16 )
                sub_1649B30(v147 + 3);
              v152 = v147[2];
              if ( v152 != -8 && v152 != 0 && v152 != -16 )
                sub_1649B30(v147);
            }
            while ( v147 != v148 );
            v16 = v157;
            v12 = v154;
            v140 = v198;
          }
        }
        if ( v140 != (unsigned __int64 *)v200 )
          _libc_free((unsigned __int64)v140);
        v17 = v199;
        v198 = v173;
        HIDWORD(v199) = v164;
      }
      else
      {
        v173 = v198;
      }
      v18 = &v173[12 * v17];
      if ( v18 )
      {
        *v18 = 6;
        v18[1] = 0;
        v19 = v189;
        v20 = v189 == 0;
        v18[2] = (unsigned __int64)v189;
        if ( v19 + 1 != 0 && !v20 && v19 != (__int64 *)-16LL )
        {
          v174 = &v173[12 * v17];
          sub_1649AC0(v18, v187 & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v174;
        }
        v18[3] = 6;
        v18[4] = 0;
        v21 = v192;
        v20 = v192 == -8;
        v18[5] = v192;
        if ( v21 != 0 && !v20 && v21 != -16 )
        {
          v175 = v18;
          sub_1649AC0(v18 + 3, v190 & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v175;
        }
        v18[6] = 6;
        v18[7] = 0;
        v22 = v195;
        v20 = v195 == 0;
        v18[8] = v195;
        if ( v22 != -8 && !v20 && v22 != -16 )
        {
          v176 = v18;
          sub_1649AC0(v18 + 6, v193 & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v176;
        }
        v18[9] = 6;
        v18[10] = 0;
        v23 = v197;
        v20 = v197 == -8;
        v18[11] = v197;
        if ( v23 != 0 && !v20 && v23 != -16 )
          sub_1649AC0(v18 + 9, v196[0] & 0xFFFFFFFFFFFFFFF8LL);
        v17 = v199;
      }
      LODWORD(v199) = v17 + 1;
      if ( v197 != 0 && v197 != -8 && v197 != -16 )
        sub_1649B30(v196);
      if ( v195 != -8 && v195 != 0 && v195 != -16 )
        sub_1649B30(&v193);
      if ( v192 != -8 && v192 != 0 && v192 != -16 )
        sub_1649B30(&v190);
      if ( v189 + 1 != 0 && v189 != 0 && v189 != (__int64 *)-16LL )
        sub_1649B30(&v187);
      v12 += 2;
    }
    while ( v165 != v12 );
  }
  v166 = (__int64 *)sub_16498A0(a2);
  v24 = sub_16498A0(a2);
  v25 = *(unsigned __int8 **)(a2 + 48);
  v187 = 0;
  v190 = v24;
  v26 = *(_QWORD *)(a2 + 40);
  v191 = 0;
  v188 = v26;
  LODWORD(v192) = 0;
  v193 = 0;
  v194 = 0;
  v189 = (__int64 *)(a2 + 24);
  v183[0] = v25;
  if ( v25 )
  {
    sub_1623A60((__int64)v183, (__int64)v25, 2);
    if ( v187 )
      sub_161E7C0((__int64)&v187, v187);
    v187 = (__int64)v183[0];
    if ( v183[0] )
      sub_1623210((__int64)v183, v183[0], (__int64)&v187);
  }
  v27 = (__int64 *)v198;
  v168 = (__int64 *)&v198[12 * (unsigned int)v199];
  if ( v198 == (unsigned __int64 *)v168 )
    goto LABEL_278;
  v177 = 0;
  v169 = 0;
  do
  {
    while ( 1 )
    {
      v29 = *(_QWORD *)v27[2];
      if ( *(_BYTE *)(v29 + 8) == 16 )
        v29 = **(_QWORD **)(v29 + 16);
      v30 = *(_DWORD *)(v29 + 8);
      v31 = *(_QWORD *)v27[8];
      v32 = v30 >> 8;
      if ( *(_BYTE *)(v31 + 8) == 16 )
        v31 = **(_QWORD **)(v31 + 16);
      v33 = *(_DWORD *)(v31 + 8) >> 8;
      v34 = (__int64 **)sub_16471D0(v166, v32);
      v35 = (__int64 **)sub_16471D0(v166, v33);
      LOWORD(v180) = 259;
      v179[0] = (__int64)"bc";
      v36 = v27[2];
      if ( v34 != *(__int64 ***)v36 )
      {
        if ( *(_BYTE *)(v36 + 16) > 0x10u )
        {
          v105 = v27[2];
          LOWORD(v184) = 257;
          v106 = sub_15FDBD0(47, v105, (__int64)v34, (__int64)v183, 0);
          v36 = v106;
          if ( v188 )
          {
            v107 = v189;
            sub_157E9D0(v188 + 40, v106);
            v108 = *(_QWORD *)(v36 + 24);
            v109 = *v107;
            *(_QWORD *)(v36 + 32) = v107;
            v109 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v36 + 24) = v109 | v108 & 7;
            *(_QWORD *)(v109 + 8) = v36 + 24;
            *v107 = *v107 & 7 | (v36 + 24);
          }
          sub_164B780(v36, v179);
          if ( v187 )
          {
            v178 = v187;
            sub_1623A60((__int64)&v178, v187, 2);
            v110 = *(_QWORD *)(v36 + 48);
            if ( v110 )
              sub_161E7C0(v36 + 48, v110);
            v111 = (unsigned __int8 *)v178;
            *(_QWORD *)(v36 + 48) = v178;
            if ( v111 )
              sub_1623210((__int64)&v178, v111, v36 + 48);
          }
        }
        else
        {
          v36 = sub_15A46C0(47, (__int64 ***)v27[2], v34, 0);
        }
      }
      v179[0] = (__int64)"bc";
      LOWORD(v180) = 259;
      v37 = v27[8];
      if ( v35 != *(__int64 ***)v37 )
      {
        if ( *(_BYTE *)(v37 + 16) > 0x10u )
        {
          v112 = v27[8];
          LOWORD(v184) = 257;
          v113 = sub_15FDBD0(47, v112, (__int64)v35, (__int64)v183, 0);
          v37 = v113;
          if ( v188 )
          {
            v172 = v189;
            sub_157E9D0(v188 + 40, v113);
            v114 = *v172;
            v115 = *(_QWORD *)(v37 + 24) & 7LL;
            *(_QWORD *)(v37 + 32) = v172;
            v114 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v37 + 24) = v114 | v115;
            *(_QWORD *)(v114 + 8) = v37 + 24;
            *v172 = *v172 & 7 | (v37 + 24);
          }
          sub_164B780(v37, v179);
          if ( v187 )
          {
            v178 = v187;
            sub_1623A60((__int64)&v178, v187, 2);
            v116 = *(_QWORD *)(v37 + 48);
            v117 = v37 + 48;
            if ( v116 )
            {
              sub_161E7C0(v37 + 48, v116);
              v117 = v37 + 48;
            }
            v118 = (unsigned __int8 *)v178;
            *(_QWORD *)(v37 + 48) = v178;
            if ( v118 )
              sub_1623210((__int64)&v178, v118, v117);
          }
        }
        else
        {
          v37 = sub_15A46C0(47, (__int64 ***)v27[8], v35, 0);
        }
      }
      v179[0] = (__int64)"bc";
      LOWORD(v180) = 259;
      v38 = v27[5];
      v171 = v38;
      if ( v35 != *(__int64 ***)v38 )
      {
        if ( *(_BYTE *)(v38 + 16) > 0x10u )
        {
          LOWORD(v184) = 257;
          v119 = sub_15FDBD0(47, v38, (__int64)v35, (__int64)v183, 0);
          v171 = v119;
          if ( v188 )
          {
            v120 = v189;
            sub_157E9D0(v188 + 40, v119);
            v121 = *v120;
            v122 = *(_QWORD *)(v171 + 24);
            *(_QWORD *)(v171 + 32) = v120;
            v121 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v171 + 24) = v121 | v122 & 7;
            *(_QWORD *)(v121 + 8) = v171 + 24;
            *v120 = *v120 & 7 | (v171 + 24);
          }
          sub_164B780(v171, v179);
          if ( v187 )
          {
            v178 = v187;
            sub_1623A60((__int64)&v178, v187, 2);
            v123 = v171 + 48;
            v124 = *(_QWORD *)(v171 + 48);
            if ( v124 )
              sub_161E7C0(v123, v124);
            v125 = (unsigned __int8 *)v178;
            *(_QWORD *)(v171 + 48) = v178;
            if ( v125 )
              sub_1623210((__int64)&v178, v125, v123);
          }
        }
        else
        {
          v171 = sub_15A46C0(47, (__int64 ***)v38, v35, 0);
        }
      }
      v179[0] = (__int64)"bc";
      LOWORD(v180) = 259;
      v39 = v27[11];
      if ( v34 != *(__int64 ***)v39 )
      {
        if ( *(_BYTE *)(v39 + 16) > 0x10u )
        {
          v96 = v27[11];
          LOWORD(v184) = 257;
          v97 = sub_15FDBD0(47, v96, (__int64)v34, (__int64)v183, 0);
          v98 = v97;
          if ( v188 )
          {
            v99 = v189;
            v161 = v97;
            sub_157E9D0(v188 + 40, v97);
            v98 = v161;
            v100 = *v99;
            v101 = *(_QWORD *)(v161 + 24);
            *(_QWORD *)(v161 + 32) = v99;
            v100 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v161 + 24) = v100 | v101 & 7;
            *(_QWORD *)(v100 + 8) = v161 + 24;
            *v99 = *v99 & 7 | (v161 + 24);
          }
          v162 = v98;
          sub_164B780(v98, v179);
          v39 = v162;
          if ( v187 )
          {
            v178 = v187;
            sub_1623A60((__int64)&v178, v187, 2);
            v39 = v162;
            v102 = *(_QWORD *)(v162 + 48);
            v103 = v162 + 48;
            if ( v102 )
            {
              sub_161E7C0(v162 + 48, v102);
              v39 = v162;
            }
            v104 = (unsigned __int8 *)v178;
            *(_QWORD *)(v39 + 48) = v178;
            if ( v104 )
            {
              v163 = v39;
              sub_1623210((__int64)&v178, v104, v103);
              v39 = v163;
            }
          }
        }
        else
        {
          v39 = sub_15A46C0(47, (__int64 ***)v27[11], v34, 0);
        }
      }
      v179[0] = (__int64)"bound0";
      LOWORD(v180) = 259;
      if ( *(_BYTE *)(v36 + 16) <= 0x10u && *(_BYTE *)(v39 + 16) <= 0x10u )
      {
        v40 = sub_15A37B0(0x24u, (_QWORD *)v36, (_QWORD *)v39, 0);
LABEL_87:
        v41 = v177;
        if ( v177 )
          goto LABEL_88;
        goto LABEL_172;
      }
      v159 = v39;
      LOWORD(v184) = 257;
      v78 = sub_1648A60(56, 2u);
      v40 = (__int64)v78;
      if ( v78 )
      {
        v79 = (__int64)v78;
        v80 = *(_QWORD ***)v36;
        if ( *(_BYTE *)(*(_QWORD *)v36 + 8LL) == 16 )
        {
          v155 = v159;
          v160 = v80[4];
          v81 = (__int64 *)sub_1643320(*v80);
          v82 = (__int64)sub_16463B0(v81, (unsigned int)v160);
          v83 = v155;
        }
        else
        {
          v82 = sub_1643320(*v80);
          v83 = v159;
        }
        sub_15FEC10(v40, v82, 51, 36, v36, v83, (__int64)v183, 0);
      }
      else
      {
        v79 = 0;
      }
      if ( v188 )
      {
        v84 = v189;
        sub_157E9D0(v188 + 40, v40);
        v85 = *(_QWORD *)(v40 + 24);
        v86 = *v84;
        *(_QWORD *)(v40 + 32) = v84;
        v86 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v40 + 24) = v86 | v85 & 7;
        *(_QWORD *)(v86 + 8) = v40 + 24;
        *v84 = *v84 & 7 | (v40 + 24);
      }
      sub_164B780(v79, v179);
      if ( !v187 )
        goto LABEL_87;
      v178 = v187;
      sub_1623A60((__int64)&v178, v187, 2);
      v87 = *(_QWORD *)(v40 + 48);
      if ( v87 )
        sub_161E7C0(v40 + 48, v87);
      v88 = (unsigned __int8 *)v178;
      *(_QWORD *)(v40 + 48) = v178;
      if ( !v88 )
        goto LABEL_87;
      sub_1623210((__int64)&v178, v88, v40 + 48);
      v41 = v177;
      if ( v177 )
        goto LABEL_88;
LABEL_172:
      if ( *(_BYTE *)(v40 + 16) > 0x17u )
      {
        if ( *(_QWORD *)(v40 + 40) == *(_QWORD *)(a2 + 40) )
          v41 = v40;
        v177 = v41;
      }
LABEL_88:
      v179[0] = (__int64)"bound1";
      LOWORD(v180) = 259;
      if ( *(_BYTE *)(v37 + 16) > 0x10u || *(_BYTE *)(v171 + 16) > 0x10u )
      {
        LOWORD(v184) = 257;
        v69 = sub_1648A60(56, 2u);
        v44 = (__int64)v69;
        if ( v69 )
        {
          v70 = (__int64)v69;
          v71 = *(_QWORD ***)v37;
          if ( *(_BYTE *)(*(_QWORD *)v37 + 8LL) == 16 )
          {
            v158 = v71[4];
            v72 = (__int64 *)sub_1643320(*v71);
            v73 = (__int64)sub_16463B0(v72, (unsigned int)v158);
          }
          else
          {
            v73 = sub_1643320(*v71);
          }
          sub_15FEC10(v44, v73, 51, 36, v37, v171, (__int64)v183, 0);
        }
        else
        {
          v70 = 0;
        }
        if ( v188 )
        {
          v74 = v189;
          sub_157E9D0(v188 + 40, v44);
          v75 = *(_QWORD *)(v44 + 24);
          v76 = *v74;
          *(_QWORD *)(v44 + 32) = v74;
          v76 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v44 + 24) = v76 | v75 & 7;
          *(_QWORD *)(v76 + 8) = v44 + 24;
          *v74 = *v74 & 7 | (v44 + 24);
        }
        sub_164B780(v70, v179);
        v42 = (unsigned __int8 *)v187;
        if ( v187 )
        {
          v178 = v187;
          sub_1623A60((__int64)&v178, v187, 2);
          v77 = *(_QWORD *)(v44 + 48);
          if ( v77 )
            sub_161E7C0(v44 + 48, v77);
          v42 = (unsigned __int8 *)v178;
          *(_QWORD *)(v44 + 48) = v178;
          if ( v42 )
          {
            sub_1623210((__int64)&v178, v42, v44 + 48);
            v45 = v177;
            if ( v177 )
              goto LABEL_92;
            goto LABEL_157;
          }
        }
      }
      else
      {
        v42 = (unsigned __int8 *)v37;
        v44 = sub_15A37B0(0x24u, (_QWORD *)v37, (_QWORD *)v171, 0);
      }
      v45 = v177;
      if ( v177 )
        goto LABEL_92;
LABEL_157:
      if ( *(_BYTE *)(v44 + 16) > 0x17u )
      {
        if ( *(_QWORD *)(v44 + 40) == *(_QWORD *)(a2 + 40) )
          v45 = v44;
        v177 = v45;
      }
LABEL_92:
      v179[0] = (__int64)"found.conflict";
      LOWORD(v180) = 259;
      v46 = *(_BYTE *)(v44 + 16);
      if ( v46 <= 0x10u )
      {
        if ( v46 == 13 )
        {
          v94 = *(_DWORD *)(v44 + 32);
          if ( v94 <= 0x40 )
          {
            v47 = 64 - v94;
            v95 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v94) == *(_QWORD *)(v44 + 24);
          }
          else
          {
            v95 = v94 == (unsigned int)sub_16A58F0(v44 + 24);
          }
          if ( v95 )
            goto LABEL_96;
        }
        if ( *(_BYTE *)(v40 + 16) <= 0x10u )
        {
          v42 = (unsigned __int8 *)v44;
          v40 = sub_15A2CF0((__int64 *)v40, v44, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
          goto LABEL_96;
        }
      }
      LOWORD(v184) = 257;
      v89 = sub_15FB440(26, (__int64 *)v40, v44, (__int64)v183, 0);
      v40 = v89;
      if ( v188 )
      {
        v90 = v189;
        sub_157E9D0(v188 + 40, v89);
        v91 = *(_QWORD *)(v40 + 24);
        v92 = *v90;
        *(_QWORD *)(v40 + 32) = v90;
        v92 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v40 + 24) = v92 | v91 & 7;
        *(_QWORD *)(v92 + 8) = v40 + 24;
        *v90 = *v90 & 7 | (v40 + 24);
      }
      sub_164B780(v40, v179);
      v42 = (unsigned __int8 *)v187;
      if ( v187 )
      {
        v178 = v187;
        sub_1623A60((__int64)&v178, v187, 2);
        v93 = *(_QWORD *)(v40 + 48);
        if ( v93 )
          sub_161E7C0(v40 + 48, v93);
        v42 = (unsigned __int8 *)v178;
        *(_QWORD *)(v40 + 48) = v178;
        if ( v42 )
        {
          sub_1623210((__int64)&v178, v42, v40 + 48);
          v48 = v177;
          if ( v177 )
            goto LABEL_97;
          goto LABEL_183;
        }
      }
LABEL_96:
      v48 = v177;
      if ( v177 )
        goto LABEL_97;
LABEL_183:
      if ( *(_BYTE *)(v40 + 16) > 0x17u )
      {
        if ( *(_QWORD *)(v40 + 40) == *(_QWORD *)(a2 + 40) )
          v48 = v40;
        v177 = v48;
      }
LABEL_97:
      if ( v169 )
        break;
      v169 = v40;
      v27 += 12;
      if ( v168 == v27 )
        goto LABEL_99;
    }
    v179[0] = (__int64)"conflict.rdx";
    LOWORD(v180) = 259;
    if ( *(_BYTE *)(v40 + 16) <= 0x10u )
    {
      if ( sub_1593BB0(v40, (__int64)v42, v43, v47) )
        goto LABEL_66;
      if ( *(_BYTE *)(v169 + 16) <= 0x10u )
      {
        v169 = sub_15A2D10((__int64 *)v169, v40, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        goto LABEL_66;
      }
    }
    LOWORD(v184) = 257;
    v126 = sub_15FB440(27, (__int64 *)v169, v40, (__int64)v183, 0);
    v169 = v126;
    v127 = v126;
    if ( v188 )
    {
      v128 = v189;
      sub_157E9D0(v188 + 40, v126);
      v129 = *(_QWORD *)(v127 + 24);
      v130 = *v128;
      *(_QWORD *)(v127 + 32) = v128;
      v130 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v127 + 24) = v130 | v129 & 7;
      *(_QWORD *)(v130 + 8) = v127 + 24;
      *v128 = *v128 & 7 | (v127 + 24);
    }
    sub_164B780(v169, v179);
    if ( v187 )
    {
      v178 = v187;
      sub_1623A60((__int64)&v178, v187, 2);
      v131 = *(_QWORD *)(v169 + 48);
      if ( v131 )
        sub_161E7C0(v169 + 48, v131);
      v132 = (unsigned __int8 *)v178;
      *(_QWORD *)(v169 + 48) = v178;
      if ( v132 )
        sub_1623210((__int64)&v178, v132, v169 + 48);
    }
LABEL_66:
    v28 = v177;
    if ( !v177 && *(_BYTE *)(v169 + 16) > 0x17u )
    {
      if ( *(_QWORD *)(v169 + 40) == *(_QWORD *)(a2 + 40) )
        v28 = v169;
      v177 = v28;
    }
    v27 += 12;
  }
  while ( v168 != v27 );
LABEL_99:
  if ( !v169 )
  {
LABEL_278:
    v54 = v187;
    v58 = 0;
    goto LABEL_110;
  }
  LOWORD(v184) = 257;
  v49 = sub_159C4F0(v166);
  v50 = sub_15FB440(26, (__int64 *)v169, v49, (__int64)v183, 0);
  LOWORD(v184) = 259;
  v183[0] = "memcheck.conflict";
  if ( v188 )
  {
    v51 = v189;
    sub_157E9D0(v188 + 40, v50);
    v52 = *(_QWORD *)(v50 + 24);
    v53 = *v51;
    *(_QWORD *)(v50 + 32) = v51;
    v53 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v50 + 24) = v53 | v52 & 7;
    *(_QWORD *)(v53 + 8) = v50 + 24;
    *v51 = *v51 & 7 | (v50 + 24);
  }
  sub_164B780(v50, (__int64 *)v183);
  v54 = v187;
  if ( v187 )
  {
    v179[0] = v187;
    sub_1623A60((__int64)v179, v187, 2);
    v55 = *(_QWORD *)(v50 + 48);
    if ( v55 )
      sub_161E7C0(v50 + 48, v55);
    v56 = (unsigned __int8 *)v179[0];
    *(_QWORD *)(v50 + 48) = v179[0];
    if ( v56 )
      sub_1623210((__int64)v179, v56, v50 + 48);
    v54 = v187;
  }
  v57 = v177;
  if ( !v177 && *(_BYTE *)(v50 + 16) > 0x17u )
  {
    if ( *(_QWORD *)(v50 + 40) == *(_QWORD *)(a2 + 40) )
      v57 = v50;
    v177 = v57;
  }
  v58 = v177;
LABEL_110:
  if ( v54 )
    sub_161E7C0((__int64)&v187, v54);
  v59 = v198;
  v60 = &v198[12 * (unsigned int)v199];
  if ( v198 != v60 )
  {
    do
    {
      v61 = *(v60 - 1);
      v60 -= 12;
      if ( v61 != -8 && v61 != 0 && v61 != -16 )
        sub_1649B30(v60 + 9);
      v62 = v60[8];
      if ( v62 != -8 && v62 != 0 && v62 != -16 )
        sub_1649B30(v60 + 6);
      v63 = v60[5];
      if ( v63 != 0 && v63 != -8 && v63 != -16 )
        sub_1649B30(v60 + 3);
      v64 = v60[2];
      if ( v64 != 0 && v64 != -8 && v64 != -16 )
        sub_1649B30(v60);
    }
    while ( v59 != v60 );
    v60 = v198;
  }
  if ( v60 != (unsigned __int64 *)v200 )
    _libc_free((unsigned __int64)v60);
  if ( v235 != v237 )
    _libc_free((unsigned __int64)v235);
  if ( v230[0] )
    sub_161E7C0((__int64)v230, v230[0]);
  j___libc_free_0(v226);
  if ( v219 != v218 )
    _libc_free((unsigned __int64)v219);
  j___libc_free_0(v214);
  j___libc_free_0(v210);
  j___libc_free_0(v206);
  if ( v204 )
  {
    v65 = v202;
    v66 = &v202[5 * v204];
    do
    {
      while ( *v65 == -8 )
      {
        if ( v65[1] != -8 )
          goto LABEL_138;
        v65 += 5;
        if ( v66 == v65 )
          goto LABEL_145;
      }
      if ( *v65 != -16 || v65[1] != -16 )
      {
LABEL_138:
        v67 = v65[4];
        if ( v67 != -8 && v67 != 0 && v67 != -16 )
          sub_1649B30(v65 + 2);
      }
      v65 += 5;
    }
    while ( v66 != v65 );
  }
LABEL_145:
  j___libc_free_0((unsigned __int64)v202);
  return v58;
}
