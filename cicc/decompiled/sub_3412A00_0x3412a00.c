// Function: sub_3412A00
// Address: 0x3412a00
//
unsigned __int8 *__fastcall sub_3412A00(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  int v7; // r13d
  __int64 *v8; // r15
  __int16 *v10; // rdx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 v16; // rsi
  unsigned int v17; // edi
  unsigned __int64 v18; // r8
  __int64 v19; // r14
  _BYTE *v20; // rax
  _BYTE *v21; // rcx
  _BYTE *j; // rdx
  _BYTE **v23; // rcx
  unsigned int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r14
  _BYTE *v28; // rax
  _QWORD *v29; // rax
  _BYTE **v30; // r12
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned __int16 *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  bool v36; // al
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // r8
  __int128 v40; // rax
  __int64 v41; // r9
  unsigned __int8 *v42; // rax
  int v43; // edx
  int v44; // edi
  unsigned __int8 *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  unsigned __int8 *v49; // rax
  unsigned __int8 *v50; // rdx
  unsigned __int8 *v51; // r13
  __int64 v52; // rdx
  unsigned __int8 *v53; // r12
  unsigned __int8 **v54; // rdx
  unsigned int v55; // r14d
  unsigned int v56; // ebx
  _QWORD *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  __int64 *v63; // rax
  unsigned int v64; // eax
  __int64 *v65; // r13
  unsigned __int16 v66; // ax
  __int64 v67; // r9
  __int64 v68; // r8
  unsigned __int8 *v69; // r12
  unsigned __int64 v70; // rdi
  __int64 v72; // rbx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned __int64 v77; // rdx
  __int64 v78; // rbx
  _QWORD *v79; // r14
  __int128 v80; // rax
  __int64 v81; // r9
  unsigned int v82; // esi
  __int64 v83; // rbx
  unsigned __int8 *v84; // rax
  unsigned __int8 *v85; // rdx
  __int64 v86; // rbx
  __int64 v87; // rdx
  __int64 v88; // r13
  __int64 v89; // rcx
  _BYTE *v90; // rax
  _BYTE *v91; // rdx
  _BYTE *i; // rdx
  __int64 v93; // rdx
  int v94; // eax
  __int64 v95; // rdx
  __int64 *v96; // r13
  __int64 v97; // rax
  __int64 v98; // r10
  __int64 *v99; // rbx
  __int64 v100; // r13
  __int64 v101; // r8
  __int64 *v102; // rax
  __int128 v103; // rax
  __int64 v104; // r9
  unsigned __int8 *v105; // rax
  int v106; // edx
  int v107; // edi
  unsigned __int8 *v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rcx
  __int64 v111; // r14
  __int64 v112; // r15
  unsigned __int16 *v113; // rdx
  __int64 v114; // rdx
  __int64 v115; // rdx
  __int64 v116; // rax
  unsigned __int8 *v117; // r14
  __int64 v118; // rax
  unsigned __int8 *v119; // rdx
  unsigned __int8 *v120; // r15
  unsigned __int64 v121; // rdx
  unsigned __int8 **v122; // rax
  __int64 v123; // rax
  unsigned __int64 v124; // rdx
  unsigned __int8 **v125; // rax
  unsigned int v126; // r14d
  unsigned int v127; // r13d
  _QWORD *v128; // rax
  __int64 v129; // rdx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rdx
  __int64 *v133; // rdx
  _QWORD *v134; // rax
  __int64 v135; // rdx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // rdx
  __int64 *v139; // rdx
  unsigned int v140; // eax
  __int64 v141; // r15
  __int64 *v142; // r13
  unsigned __int16 v143; // r14
  unsigned __int16 v144; // ax
  __int64 v145; // r9
  unsigned __int16 v146; // r10
  unsigned __int8 *v147; // rax
  int v148; // edx
  int v149; // r13d
  __int64 v150; // r14
  unsigned __int8 *v151; // rax
  int v152; // edx
  __int64 v153; // r8
  __int64 v154; // rdx
  __int64 v155; // rdx
  __int64 v156; // rdx
  __int64 v157; // rdx
  __int128 v158; // [rsp-20h] [rbp-2E0h]
  __int128 v159; // [rsp-20h] [rbp-2E0h]
  __int128 v160; // [rsp-20h] [rbp-2E0h]
  _BYTE **v161; // [rsp-10h] [rbp-2D0h]
  __int128 v162; // [rsp-10h] [rbp-2D0h]
  __int128 v163; // [rsp-10h] [rbp-2D0h]
  __int128 v164; // [rsp-10h] [rbp-2D0h]
  __int128 v165; // [rsp-10h] [rbp-2D0h]
  __int64 v166; // [rsp-8h] [rbp-2C8h]
  __int64 v167; // [rsp+18h] [rbp-2A8h]
  __int64 v168; // [rsp+20h] [rbp-2A0h]
  __int64 v169; // [rsp+20h] [rbp-2A0h]
  unsigned int v170; // [rsp+34h] [rbp-28Ch]
  unsigned __int64 v171; // [rsp+38h] [rbp-288h]
  unsigned __int16 v172; // [rsp+40h] [rbp-280h]
  __int64 v173; // [rsp+48h] [rbp-278h]
  unsigned __int16 v175; // [rsp+56h] [rbp-26Ah]
  __int64 v176; // [rsp+58h] [rbp-268h]
  unsigned int v177; // [rsp+58h] [rbp-268h]
  unsigned int v178; // [rsp+58h] [rbp-268h]
  __int64 v179; // [rsp+70h] [rbp-250h]
  __int64 v180; // [rsp+78h] [rbp-248h]
  _QWORD *v181; // [rsp+80h] [rbp-240h]
  __int64 v182; // [rsp+80h] [rbp-240h]
  __int64 v183; // [rsp+88h] [rbp-238h]
  __int64 v184; // [rsp+88h] [rbp-238h]
  unsigned int v185; // [rsp+90h] [rbp-230h]
  __int64 v186; // [rsp+90h] [rbp-230h]
  __int64 v187; // [rsp+90h] [rbp-230h]
  __int64 v188; // [rsp+90h] [rbp-230h]
  __int64 v189; // [rsp+98h] [rbp-228h]
  __int64 v190; // [rsp+98h] [rbp-228h]
  unsigned __int16 v191; // [rsp+98h] [rbp-228h]
  unsigned int v192; // [rsp+98h] [rbp-228h]
  unsigned int v193; // [rsp+98h] [rbp-228h]
  _QWORD *v195; // [rsp+A0h] [rbp-220h]
  __int64 v196; // [rsp+A0h] [rbp-220h]
  _QWORD *v197; // [rsp+A0h] [rbp-220h]
  _QWORD *v198; // [rsp+A0h] [rbp-220h]
  __int64 *v199; // [rsp+A0h] [rbp-220h]
  __int64 v200; // [rsp+A0h] [rbp-220h]
  __int64 v201; // [rsp+A0h] [rbp-220h]
  __int64 v202; // [rsp+A0h] [rbp-220h]
  __int64 v203; // [rsp+A8h] [rbp-218h]
  __int64 v204; // [rsp+A8h] [rbp-218h]
  __int64 v205; // [rsp+A8h] [rbp-218h]
  __int64 v206; // [rsp+A8h] [rbp-218h]
  __int64 v207; // [rsp+A8h] [rbp-218h]
  __int64 v208; // [rsp+A8h] [rbp-218h]
  unsigned __int16 v209; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 v210; // [rsp+D8h] [rbp-1E8h]
  __int64 v211; // [rsp+E0h] [rbp-1E0h] BYREF
  int v212; // [rsp+E8h] [rbp-1D8h]
  __int16 v213; // [rsp+F0h] [rbp-1D0h] BYREF
  __int64 v214; // [rsp+F8h] [rbp-1C8h]
  __int64 v215; // [rsp+100h] [rbp-1C0h] BYREF
  __int64 v216; // [rsp+108h] [rbp-1B8h]
  unsigned __int8 *v217; // [rsp+110h] [rbp-1B0h]
  __int64 v218; // [rsp+118h] [rbp-1A8h]
  _BYTE *v219; // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v220; // [rsp+128h] [rbp-198h]
  _BYTE v221[64]; // [rsp+130h] [rbp-190h] BYREF
  _BYTE *v222; // [rsp+170h] [rbp-150h] BYREF
  __int64 v223; // [rsp+178h] [rbp-148h]
  _BYTE v224[128]; // [rsp+180h] [rbp-140h] BYREF
  _BYTE *v225; // [rsp+200h] [rbp-C0h] BYREF
  __int64 v226; // [rsp+208h] [rbp-B8h]
  _BYTE v227[176]; // [rsp+210h] [rbp-B0h] BYREF

  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v209 = v11;
  v210 = v12;
  if ( v11 )
  {
    v173 = 0;
    v175 = word_4456580[v11 - 1];
    v176 = v175;
  }
  else
  {
    v13 = sub_3009970((__int64)&v209, a2, v12, a4, a5);
    v175 = v13;
    v11 = v209;
    v173 = v14;
    v176 = v13;
    if ( !v209 )
    {
      if ( !sub_3007100((__int64)&v209) )
        goto LABEL_7;
      goto LABEL_35;
    }
  }
  if ( (unsigned __int16)(v11 - 176) > 0x34u )
  {
LABEL_4:
    v170 = word_4456340[v209 - 1];
    goto LABEL_8;
  }
LABEL_35:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v209 )
  {
    if ( (unsigned __int16)(v209 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_4;
  }
LABEL_7:
  v170 = sub_3007130((__int64)&v209, a2);
LABEL_8:
  v15 = &v211;
  v16 = *(_QWORD *)(a2 + 80);
  v211 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v211, v16, 1);
  v212 = *(_DWORD *)(a2 + 72);
  if ( a3 )
  {
    v17 = v170;
    if ( v170 > a3 )
      v17 = a3;
    v170 = v17;
  }
  else
  {
    a3 = v170;
  }
  v18 = *(unsigned int *)(a2 + 64);
  if ( *(_DWORD *)(a2 + 68) == 2 )
  {
    v88 = (unsigned int)v18;
    v89 = 0x400000000LL;
    v222 = v224;
    v223 = 0x800000000LL;
    v226 = 0x800000000LL;
    v90 = v221;
    v91 = v221;
    v225 = v227;
    v219 = v221;
    v220 = 0x400000000LL;
    if ( (_DWORD)v18 )
    {
      if ( (unsigned int)v18 > 4uLL )
      {
        v16 = (__int64)v221;
        v193 = v18;
        sub_C8D5F0((__int64)&v219, v221, (unsigned int)v18, 0x10u, v18, a6);
        v91 = v219;
        v18 = v193;
        v90 = &v219[16 * (unsigned int)v220];
      }
      for ( i = &v91[16 * v88]; i != v90; v90 += 16 )
      {
        if ( v90 )
        {
          *(_QWORD *)v90 = 0;
          *((_DWORD *)v90 + 2) = 0;
        }
      }
      LODWORD(v220) = v18;
    }
    v93 = *(_QWORD *)(a2 + 48);
    v94 = *(unsigned __int16 *)(v93 + 16);
    v95 = *(_QWORD *)(v93 + 24);
    v213 = v94;
    v214 = v95;
    if ( (_WORD)v94 )
    {
      v169 = 0;
      v172 = word_4456580[v94 - 1];
    }
    else
    {
      v172 = sub_3009970((__int64)&v213, v16, v95, v89, v18);
      v169 = v157;
    }
    v96 = v8;
    v182 = 0;
    if ( v170 )
    {
      do
      {
        v97 = *(unsigned int *)(a2 + 64);
        if ( (_DWORD)v97 )
        {
          v190 = (__int64)v15;
          v98 = 0;
          v99 = v96;
          v100 = 0;
          v180 = 40 * v97;
          do
          {
            v102 = (__int64 *)(v100 + *(_QWORD *)(a2 + 40));
            v110 = *v102;
            v111 = *v102;
            v112 = v102[1];
            v113 = (unsigned __int16 *)(*(_QWORD *)(*v102 + 48) + 16LL * *((unsigned int *)v102 + 2));
            LODWORD(v102) = *v113;
            v114 = *((_QWORD *)v113 + 1);
            LOWORD(v215) = (_WORD)v102;
            v216 = v114;
            if ( (_WORD)v102 )
            {
              v101 = 0;
              LOWORD(v102) = word_4456580[(int)v102 - 1];
            }
            else
            {
              v188 = v98;
              v102 = (__int64 *)sub_3009970((__int64)&v215, v16, v114, v110, v18);
              v98 = v188;
              v99 = v102;
              v101 = v115;
            }
            LOWORD(v99) = (_WORD)v102;
            v184 = v98;
            v100 += 40;
            v187 = v101;
            *(_QWORD *)&v103 = sub_3400EE0((__int64)a1, v182, v190, 0, a7);
            v16 = 158;
            *((_QWORD *)&v159 + 1) = v112;
            *(_QWORD *)&v159 = v111;
            v105 = sub_3406EB0(a1, 0x9Eu, v190, (unsigned int)v99, v187, v104, v159, v103);
            v107 = v106;
            v108 = v105;
            v109 = (__int64)v219;
            *(_QWORD *)&v219[v184] = v108;
            *(_DWORD *)(v109 + v184 + 8) = v107;
            v98 = v184 + 16;
          }
          while ( v180 != v100 );
          v96 = v99;
          v15 = (__int64 *)v190;
        }
        v116 = v176;
        LOWORD(v116) = v175;
        v176 = v116;
        v215 = v116;
        v216 = v173;
        LOWORD(v217) = v172;
        v218 = v169;
        v16 = *(unsigned int *)(a2 + 24);
        *((_QWORD *)&v164 + 1) = (unsigned int)v220;
        *(_QWORD *)&v164 = v219;
        v117 = sub_3411BE0(a1, v16, (__int64)v15, (unsigned __int16 *)&v215, 2, a6, v164);
        v118 = (unsigned int)v223;
        v120 = v119;
        v18 = v166;
        v121 = (unsigned int)v223 + 1LL;
        if ( v121 > HIDWORD(v223) )
        {
          v16 = (__int64)v224;
          sub_C8D5F0((__int64)&v222, v224, v121, 0x10u, v166, a6);
          v118 = (unsigned int)v223;
        }
        v122 = (unsigned __int8 **)&v222[16 * v118];
        v122[1] = v120;
        *v122 = v117;
        LODWORD(v223) = v223 + 1;
        v171 = v171 & 0xFFFFFFFF00000000LL | 1;
        v123 = (unsigned int)v226;
        v124 = (unsigned int)v226 + 1LL;
        if ( v124 > HIDWORD(v226) )
        {
          v16 = (__int64)v227;
          sub_C8D5F0((__int64)&v225, v227, v124, 0x10u, v18, a6);
          v123 = (unsigned int)v226;
        }
        v125 = (unsigned __int8 **)&v225[16 * v123];
        ++v182;
        *v125 = v117;
        v125[1] = (unsigned __int8 *)v171;
        LODWORD(v226) = v226 + 1;
      }
      while ( v170 != v182 );
    }
    if ( a3 > v170 )
    {
      HIWORD(v126) = WORD1(v176);
      v127 = v170;
      do
      {
        LOWORD(v126) = v175;
        v215 = 0;
        LODWORD(v216) = 0;
        v128 = sub_33F17F0(a1, 51, (__int64)&v215, v126, v173);
        v130 = (__int64)v128;
        v131 = v129;
        if ( v215 )
        {
          v197 = v128;
          v205 = v129;
          sub_B91220((__int64)&v215, v215);
          v130 = (__int64)v197;
          v131 = v205;
        }
        v132 = (unsigned int)v223;
        if ( (unsigned __int64)(unsigned int)v223 + 1 > HIDWORD(v223) )
        {
          v202 = v130;
          v208 = v131;
          sub_C8D5F0((__int64)&v222, v224, (unsigned int)v223 + 1LL, 0x10u, v130, v131);
          v132 = (unsigned int)v223;
          v130 = v202;
          v131 = v208;
        }
        v133 = (__int64 *)&v222[16 * v132];
        *v133 = v130;
        v133[1] = v131;
        LODWORD(v223) = v223 + 1;
        v215 = 0;
        LODWORD(v216) = 0;
        v134 = sub_33F17F0(a1, 51, (__int64)&v215, v172, v169);
        v136 = (__int64)v134;
        v137 = v135;
        if ( v215 )
        {
          v198 = v134;
          v206 = v135;
          sub_B91220((__int64)&v215, v215);
          v136 = (__int64)v198;
          v137 = v206;
        }
        v138 = (unsigned int)v226;
        if ( (unsigned __int64)(unsigned int)v226 + 1 > HIDWORD(v226) )
        {
          v201 = v136;
          v207 = v137;
          sub_C8D5F0((__int64)&v225, v227, (unsigned int)v226 + 1LL, 0x10u, v136, v137);
          v138 = (unsigned int)v226;
          v136 = v201;
          v137 = v207;
        }
        v139 = (__int64 *)&v225[16 * v138];
        ++v127;
        *v139 = v136;
        v139[1] = v137;
        LODWORD(v226) = v226 + 1;
      }
      while ( a3 != v127 );
    }
    HIWORD(v140) = WORD1(v176);
    LOWORD(v140) = v175;
    v141 = 0;
    v142 = (__int64 *)a1[8];
    v178 = v140;
    v143 = sub_2D43050(v175, a3);
    if ( !v143 )
    {
      v143 = sub_3009400(v142, v178, v173, a3, 0);
      v141 = v156;
    }
    v199 = (__int64 *)a1[8];
    v144 = sub_2D43050(v172, a3);
    v145 = 0;
    v146 = v144;
    if ( !v144 )
    {
      v146 = sub_3009400(v199, v172, v169, a3, 0);
      v145 = v155;
    }
    v191 = v146;
    *((_QWORD *)&v165 + 1) = (unsigned int)v223;
    *(_QWORD *)&v165 = v222;
    v200 = v145;
    v147 = sub_33FC220(a1, 156, (__int64)v15, v143, v141, v145, v165);
    v149 = v148;
    v150 = (__int64)v147;
    *((_QWORD *)&v160 + 1) = (unsigned int)v226;
    *(_QWORD *)&v160 = v225;
    v151 = sub_33FC220(a1, 156, (__int64)v15, v191, v200, v200, v160);
    LODWORD(v218) = v152;
    LODWORD(v216) = v149;
    v215 = v150;
    v217 = v151;
    v69 = sub_3411660((__int64)a1, (unsigned int *)&v215, 2u, (__int64)v15, v153);
    if ( v219 != v221 )
      _libc_free((unsigned __int64)v219);
    if ( v225 != v227 )
      _libc_free((unsigned __int64)v225);
    v70 = (unsigned __int64)v222;
    if ( v222 != v224 )
LABEL_59:
      _libc_free(v70);
  }
  else
  {
    v19 = (unsigned int)v18;
    v225 = v227;
    v226 = 0x800000000LL;
    v20 = v224;
    v222 = v224;
    v223 = 0x400000000LL;
    if ( (_DWORD)v18 )
    {
      v21 = v224;
      if ( (unsigned int)v18 > 4uLL )
      {
        v192 = v18;
        sub_C8D5F0((__int64)&v222, v224, (unsigned int)v18, 0x10u, v18, a6);
        v21 = v222;
        v18 = v192;
        v20 = &v222[16 * (unsigned int)v223];
      }
      for ( j = &v21[16 * v19]; j != v20; v20 += 16 )
      {
        if ( v20 )
        {
          *(_QWORD *)v20 = 0;
          *((_DWORD *)v20 + 2) = 0;
        }
      }
      LODWORD(v223) = v18;
    }
    v179 = 0;
    v23 = &v219;
    if ( v170 )
    {
      v181 = a1;
      HIWORD(v24) = HIWORD(v7);
      while ( 1 )
      {
        v25 = *(unsigned int *)(a2 + 64);
        if ( (_DWORD)v25 )
        {
          v26 = 0;
          v27 = 0;
          v189 = 40 * v25;
          do
          {
            v29 = (_QWORD *)(v26 + *(_QWORD *)(a2 + 40));
            v23 = (_BYTE **)*v29;
            v30 = (_BYTE **)*v29;
            v31 = v29[1];
            v32 = *((unsigned int *)v29 + 2);
            v33 = (unsigned __int16 *)(*(_QWORD *)(*v29 + 48LL) + 16 * v32);
            v34 = *v33;
            v35 = *((_QWORD *)v33 + 1);
            LOWORD(v219) = v34;
            v220 = v35;
            if ( (_WORD)v34 )
            {
              if ( (unsigned __int16)(v34 - 17) <= 0xD3u )
              {
                v39 = 0;
                LOWORD(v37) = word_4456580[(int)v34 - 1];
                goto LABEL_33;
              }
            }
            else
            {
              v183 = (__int64)v23;
              v185 = v32;
              v36 = sub_30070B0((__int64)&v219);
              LODWORD(v32) = v185;
              v23 = (_BYTE **)v183;
              if ( v36 )
              {
                v37 = sub_3009970((__int64)&v219, v34, v185, v183, v18);
                HIWORD(v24) = HIWORD(v37);
                v39 = v38;
LABEL_33:
                LOWORD(v24) = v37;
                v186 = v39;
                *(_QWORD *)&v40 = sub_3400EE0((__int64)v181, v179, (__int64)&v211, 0, a7);
                *((_QWORD *)&v158 + 1) = v31;
                *(_QWORD *)&v158 = v30;
                v42 = sub_3406EB0(v181, 0x9Eu, (__int64)&v211, v24, v186, v41, v158, v40);
                v44 = v43;
                v45 = v42;
                v46 = (__int64)v222;
                *(_QWORD *)&v222[v27] = v45;
                *(_DWORD *)(v46 + v27 + 8) = v44;
                goto LABEL_29;
              }
            }
            v28 = &v222[v27];
            *(_QWORD *)v28 = v23;
            *((_DWORD *)v28 + 2) = v32;
LABEL_29:
            v26 += 40;
            v27 += 16;
          }
          while ( v189 != v26 );
        }
        v47 = *(unsigned int *)(a2 + 24);
        if ( (_DWORD)v47 == 222 )
        {
          v74 = *((_QWORD *)v222 + 2);
          LODWORD(v75) = *(unsigned __int16 *)(v74 + 96);
          v76 = *(_QWORD *)(v74 + 104);
          LOWORD(v219) = v75;
          v220 = v76;
          if ( (_WORD)v75 )
          {
            LOWORD(v75) = word_4456580[(int)v75 - 1];
            v77 = 0;
          }
          else
          {
            v75 = sub_3009970((__int64)&v219, v47, v76, (__int64)v23, v18);
            v168 = v75;
          }
          v78 = v168;
          v79 = v181;
          LOWORD(v78) = v75;
          v168 = v78;
          *(_QWORD *)&v80 = sub_33F7D60(v181, (unsigned int)v78, v77);
          v82 = *(_DWORD *)(a2 + 24);
        }
        else
        {
          if ( (int)v47 > 222 )
          {
            if ( (_DWORD)v47 == 235 )
            {
              v72 = v176;
              LOWORD(v72) = v175;
              v176 = v72;
              v49 = (unsigned __int8 *)sub_33F2D30(
                                         v181,
                                         (__int64)&v211,
                                         v72,
                                         v173,
                                         *(_QWORD *)v222,
                                         *((_QWORD *)v222 + 1),
                                         *(_DWORD *)(a2 + 96),
                                         *(_DWORD *)(a2 + 100));
              goto LABEL_43;
            }
LABEL_42:
            v48 = v176;
            LOWORD(v48) = v175;
            v176 = v48;
            v49 = sub_33FBA10(
                    v181,
                    v47,
                    (__int64)&v211,
                    (unsigned int)v48,
                    v173,
                    *(_DWORD *)(a2 + 28),
                    (__int64)v222,
                    (unsigned int)v223);
            goto LABEL_43;
          }
          if ( (int)v47 > 194 )
          {
            if ( (_DWORD)v47 == 206 )
            {
              v73 = v176;
              LOWORD(v73) = v175;
              *((_QWORD *)&v163 + 1) = (unsigned int)v223;
              *(_QWORD *)&v163 = v222;
              v176 = v73;
              v49 = sub_33FC220(v181, 205, (__int64)&v211, (unsigned int)v73, v173, a6, v163);
LABEL_43:
              v51 = v50;
              v52 = (unsigned int)v226;
              v53 = v49;
              v23 = v161;
              v18 = (unsigned int)v226 + 1LL;
              if ( v18 > HIDWORD(v226) )
                goto LABEL_71;
              goto LABEL_44;
            }
            goto LABEL_42;
          }
          if ( (int)v47 <= 189 )
            goto LABEL_42;
          v86 = v167;
          v79 = v181;
          v87 = *(_QWORD *)(*(_QWORD *)v222 + 48LL) + 16LL * *((unsigned int *)v222 + 2);
          LOWORD(v86) = *(_WORD *)v87;
          v167 = v86;
          *(_QWORD *)&v80 = sub_33FB4D0(
                              (__int64)v181,
                              v86,
                              *(_QWORD *)(v87 + 8),
                              *((_QWORD *)v222 + 2),
                              *((_QWORD *)v222 + 3),
                              a7);
          v82 = *(_DWORD *)(a2 + 24);
        }
        v83 = v176;
        LOWORD(v83) = v175;
        v176 = v83;
        v84 = sub_3406EB0(v79, v82, (__int64)&v211, (unsigned int)v83, v173, v81, *(_OWORD *)v222, v80);
        v51 = v85;
        v52 = (unsigned int)v226;
        v53 = v84;
        v18 = (unsigned int)v226 + 1LL;
        if ( v18 > HIDWORD(v226) )
        {
LABEL_71:
          sub_C8D5F0((__int64)&v225, v227, v18, 0x10u, v18, a6);
          v52 = (unsigned int)v226;
        }
LABEL_44:
        v54 = (unsigned __int8 **)&v225[16 * v52];
        *v54 = v53;
        v54[1] = v51;
        ++v179;
        LODWORD(v226) = v226 + 1;
        if ( v170 == v179 )
        {
          a1 = v181;
          v15 = &v211;
          break;
        }
      }
    }
    if ( a3 > v170 )
    {
      v55 = v170;
      HIWORD(v56) = WORD1(v176);
      do
      {
        LOWORD(v56) = v175;
        v219 = 0;
        LODWORD(v220) = 0;
        v57 = sub_33F17F0(a1, 51, (__int64)&v219, v56, v173);
        v59 = (__int64)v57;
        v60 = v58;
        if ( v219 )
        {
          v195 = v57;
          v203 = v58;
          sub_B91220((__int64)&v219, (__int64)v219);
          v59 = (__int64)v195;
          v60 = v203;
        }
        v61 = (unsigned int)v226;
        v62 = (unsigned int)v226 + 1LL;
        if ( v62 > HIDWORD(v226) )
        {
          v196 = v59;
          v204 = v60;
          sub_C8D5F0((__int64)&v225, v227, v62, 0x10u, v59, v60);
          v61 = (unsigned int)v226;
          v59 = v196;
          v60 = v204;
        }
        v63 = (__int64 *)&v225[16 * v61];
        ++v55;
        *v63 = v59;
        v63[1] = v60;
        LODWORD(v226) = v226 + 1;
      }
      while ( a3 != v55 );
      v15 = &v211;
    }
    HIWORD(v64) = WORD1(v176);
    LOWORD(v64) = v175;
    v65 = (__int64 *)a1[8];
    v177 = v64;
    v66 = sub_2D43050(v175, a3);
    v68 = 0;
    if ( !v66 )
    {
      v66 = sub_3009400(v65, v177, v173, a3, 0);
      v68 = v154;
    }
    *((_QWORD *)&v162 + 1) = (unsigned int)v226;
    *(_QWORD *)&v162 = v225;
    v69 = sub_33FC220(a1, 156, (__int64)&v211, v66, v68, v67, v162);
    if ( v222 != v224 )
      _libc_free((unsigned __int64)v222);
    v70 = (unsigned __int64)v225;
    if ( v225 != v227 )
      goto LABEL_59;
  }
  if ( v211 )
    sub_B91220((__int64)v15, v211);
  return v69;
}
