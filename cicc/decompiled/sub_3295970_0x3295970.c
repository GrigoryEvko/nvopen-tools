// Function: sub_3295970
// Address: 0x3295970
//
__int64 __fastcall sub_3295970(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v6; // rax
  unsigned int v7; // ebx
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // r13
  int v14; // eax
  __int16 *v15; // rax
  unsigned __int16 v16; // r14
  __int64 v17; // rax
  __int16 *v18; // rdx
  unsigned __int16 v19; // r14
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rsi
  __int16 *v23; // rax
  unsigned __int16 v24; // r14
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rax
  bool v28; // r14
  __int16 *v29; // rax
  unsigned __int16 v30; // r14
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdi
  int v37; // r14d
  int v38; // edx
  int v39; // r15d
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 result; // rax
  int v46; // eax
  __int64 v47; // r12
  unsigned __int16 *v48; // rdx
  __int64 v49; // r14
  __int64 *v50; // rax
  __int64 v51; // r12
  __int64 v52; // rcx
  __int64 v53; // r13
  __int64 v54; // rbx
  int v55; // eax
  __int64 v56; // rdx
  unsigned int v57; // eax
  __int128 v58; // rax
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // rcx
  __int64 v63; // r10
  __int64 v64; // rbx
  __int64 v65; // rdx
  __int64 v66; // r11
  _QWORD *v67; // rax
  _QWORD *v68; // rdx
  _QWORD *v69; // rax
  _QWORD *v70; // rdx
  __int64 v71; // rax
  int v72; // ecx
  __int64 v73; // rax
  unsigned __int16 *v74; // rax
  __int64 v75; // r9
  int v76; // r14d
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rax
  unsigned __int8 v80; // al
  __int64 v81; // rax
  __int64 v82; // r12
  unsigned int v83; // ebx
  int v84; // r13d
  __int64 v85; // r8
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int64 v88; // rdx
  __int64 *v89; // rax
  __int64 v90; // rax
  _BYTE *v91; // rdx
  int v92; // eax
  __int64 v93; // rdx
  __int64 v94; // r13
  unsigned __int16 *v95; // rax
  unsigned __int16 v96; // dx
  __int64 v97; // r9
  int v98; // r14d
  __int64 v99; // rax
  __int64 v100; // r12
  __int128 v101; // rax
  __int128 v102; // rax
  int v103; // r9d
  __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // r13
  __int64 v107; // r12
  __int128 v108; // rax
  int v109; // r9d
  __int64 v110; // rdx
  __int64 v111; // rdi
  __int64 v112; // rax
  int v113; // esi
  __int64 v114; // rax
  int v115; // ecx
  unsigned __int16 *v116; // rdx
  int v117; // eax
  __int64 v118; // rdx
  __int16 v119; // ax
  __int64 v120; // rdx
  unsigned __int16 *v121; // rdx
  int v122; // eax
  __int64 v123; // rdx
  __int16 v124; // ax
  __int64 v125; // rdx
  __int64 (*v126)(); // rax
  char v127; // al
  unsigned __int16 v128; // ax
  __int64 v129; // rsi
  __int64 (__fastcall *v130)(__int64, __int64, unsigned int, __int64); // rax
  __int128 v131; // rax
  __int64 v132; // r13
  __int64 v133; // r12
  int v134; // r9d
  __int128 v135; // rax
  int v136; // r9d
  __int128 v137; // rax
  int v138; // r9d
  __int64 v139; // r12
  __int64 v140; // rdx
  __int64 v141; // r13
  __int64 v142; // rbx
  __int64 v143; // rax
  int v144; // edx
  int v145; // eax
  __int64 v146; // r13
  __int64 v147; // rax
  int v148; // edx
  __int64 v149; // r14
  __int64 v150; // r15
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // rax
  int v154; // r8d
  int v155; // r9d
  __int128 *v156; // r12
  unsigned __int64 v157; // rbx
  int v158; // r13d
  __int128 *v159; // rdx
  __int128 *v160; // rax
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // r9
  __int128 v166; // [rsp-30h] [rbp-480h]
  __int128 v167; // [rsp-10h] [rbp-460h]
  __int128 v168; // [rsp-10h] [rbp-460h]
  __int128 v169; // [rsp-10h] [rbp-460h]
  __int128 v170; // [rsp-10h] [rbp-460h]
  __int64 v171; // [rsp+0h] [rbp-450h]
  __int64 v172; // [rsp+0h] [rbp-450h]
  __int64 v173; // [rsp+8h] [rbp-448h]
  __int64 v174; // [rsp+8h] [rbp-448h]
  __int64 v175; // [rsp+10h] [rbp-440h]
  unsigned __int8 v176; // [rsp+10h] [rbp-440h]
  char v177; // [rsp+10h] [rbp-440h]
  __int64 v178; // [rsp+18h] [rbp-438h]
  char v179; // [rsp+20h] [rbp-430h]
  __int64 v180; // [rsp+20h] [rbp-430h]
  __int64 v181; // [rsp+28h] [rbp-428h]
  __int16 v182; // [rsp+32h] [rbp-41Eh]
  unsigned int v183; // [rsp+38h] [rbp-418h]
  unsigned int v184; // [rsp+38h] [rbp-418h]
  int v185; // [rsp+38h] [rbp-418h]
  const void *s1; // [rsp+40h] [rbp-410h]
  __int64 s1a; // [rsp+40h] [rbp-410h]
  __int64 s1b; // [rsp+40h] [rbp-410h]
  __int128 s1c; // [rsp+40h] [rbp-410h]
  unsigned int v190; // [rsp+50h] [rbp-400h]
  __int64 v191; // [rsp+50h] [rbp-400h]
  __int16 v192; // [rsp+50h] [rbp-400h]
  __int128 v193; // [rsp+50h] [rbp-400h]
  int v194; // [rsp+60h] [rbp-3F0h]
  __int128 v195; // [rsp+60h] [rbp-3F0h]
  __int64 v196; // [rsp+60h] [rbp-3F0h]
  __int64 v197; // [rsp+60h] [rbp-3F0h]
  __int128 v198; // [rsp+60h] [rbp-3F0h]
  __int64 v199; // [rsp+70h] [rbp-3E0h]
  __int64 v202; // [rsp+80h] [rbp-3D0h]
  __int128 v203; // [rsp+80h] [rbp-3D0h]
  __int128 *v204; // [rsp+80h] [rbp-3D0h]
  unsigned int v205; // [rsp+90h] [rbp-3C0h]
  unsigned __int16 v206; // [rsp+90h] [rbp-3C0h]
  __int128 v207; // [rsp+90h] [rbp-3C0h]
  __int64 v208; // [rsp+90h] [rbp-3C0h]
  __int64 v209; // [rsp+90h] [rbp-3C0h]
  __int64 v210; // [rsp+90h] [rbp-3C0h]
  __int64 v211; // [rsp+90h] [rbp-3C0h]
  __int64 v212; // [rsp+98h] [rbp-3B8h]
  __int64 v213; // [rsp+A0h] [rbp-3B0h]
  unsigned int v214; // [rsp+A0h] [rbp-3B0h]
  __int64 v215; // [rsp+A0h] [rbp-3B0h]
  __int64 v216; // [rsp+A0h] [rbp-3B0h]
  __int128 v217; // [rsp+A0h] [rbp-3B0h]
  int v218; // [rsp+A0h] [rbp-3B0h]
  int v219; // [rsp+A0h] [rbp-3B0h]
  __int128 *v220; // [rsp+A0h] [rbp-3B0h]
  __int16 v221; // [rsp+A2h] [rbp-3AEh]
  int v222; // [rsp+A8h] [rbp-3A8h]
  int v223; // [rsp+A8h] [rbp-3A8h]
  unsigned int v224; // [rsp+B8h] [rbp-398h] BYREF
  int v225; // [rsp+BCh] [rbp-394h] BYREF
  __m128i v226; // [rsp+C0h] [rbp-390h] BYREF
  __m128i v227; // [rsp+D0h] [rbp-380h] BYREF
  __m128i v228; // [rsp+E0h] [rbp-370h] BYREF
  _BYTE *v229; // [rsp+F0h] [rbp-360h] BYREF
  __int64 v230; // [rsp+F8h] [rbp-358h]
  _BYTE v231[256]; // [rsp+100h] [rbp-350h] BYREF
  __m128i v232; // [rsp+200h] [rbp-250h] BYREF
  _BYTE v233[256]; // [rsp+210h] [rbp-240h] BYREF
  _BYTE *v234; // [rsp+310h] [rbp-140h] BYREF
  __int64 v235; // [rsp+318h] [rbp-138h]
  _BYTE v236[304]; // [rsp+320h] [rbp-130h] BYREF

  v6 = *(unsigned __int16 **)(a2 + 48);
  v199 = *((_QWORD *)v6 + 1);
  v7 = *v6;
  v8 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v9 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v10 = *(_DWORD *)(a2 + 24);
  v11 = *(unsigned int *)(a2 + 28);
  v226 = v8;
  v205 = v10;
  v194 = v11;
  v227 = v9;
  if ( v10 <= 62 )
  {
    if ( v10 <= 58 )
      goto LABEL_3;
LABEL_38:
    v12 = v226.m128i_i64[0];
    goto LABEL_39;
  }
  if ( (unsigned int)(v10 - 65) <= 1 )
    goto LABEL_38;
LABEL_3:
  v12 = v226.m128i_i64[0];
  v13 = v227.m128i_i64[0];
  v14 = *(_DWORD *)(v227.m128i_i64[0] + 24);
  if ( *(_DWORD *)(v226.m128i_i64[0] + 24) == 165 )
  {
    if ( v14 == 165 )
    {
      v15 = *(__int16 **)(v226.m128i_i64[0] + 48);
      v16 = *v15;
      v17 = *((_QWORD *)v15 + 1);
      LOWORD(v234) = v16;
      v235 = v17;
      if ( v16 )
      {
        if ( (unsigned __int16)(v16 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        v190 = word_4456340[v16 - 1];
      }
      else
      {
        if ( sub_3007100((__int64)&v234) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        v190 = sub_3007130((__int64)&v234, v11);
      }
      v18 = *(__int16 **)(v227.m128i_i64[0] + 48);
      v19 = *v18;
      v20 = *((_QWORD *)v18 + 1);
      s1 = *(const void **)(v226.m128i_i64[0] + 96);
      LOWORD(v234) = v19;
      v235 = v20;
      if ( v19 )
      {
        if ( (unsigned __int16)(v19 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        v21 = word_4456340[v19 - 1];
      }
      else
      {
        if ( sub_3007100((__int64)&v234) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        v21 = sub_3007130((__int64)&v234, v11);
      }
      if ( v190 == v21 && (!(4LL * v190) || !memcmp(s1, *(const void **)(v227.m128i_i64[0] + 96), 4LL * v190)) )
      {
        v110 = *(_QWORD *)(v226.m128i_i64[0] + 40);
        if ( *(_DWORD *)(*(_QWORD *)(v110 + 40) + 24LL) == 51 )
        {
          v111 = *(_QWORD *)(v227.m128i_i64[0] + 40);
          if ( *(_DWORD *)(*(_QWORD *)(v111 + 40) + 24LL) == 51 )
          {
            v112 = *(_QWORD *)(v226.m128i_i64[0] + 56);
            if ( v112 )
            {
              v113 = 1;
              do
              {
                if ( v226.m128i_i32[2] == *(_DWORD *)(v112 + 8) )
                {
                  if ( !v113 )
                    goto LABEL_174;
                  v112 = *(_QWORD *)(v112 + 32);
                  if ( !v112 )
                    goto LABEL_172;
                  if ( v226.m128i_i32[2] == *(_DWORD *)(v112 + 8) )
                    goto LABEL_174;
                  v113 = 0;
                }
                v112 = *(_QWORD *)(v112 + 32);
              }
              while ( v112 );
              if ( v113 != 1 )
                goto LABEL_172;
            }
LABEL_174:
            v153 = *(_QWORD *)(v227.m128i_i64[0] + 56);
            if ( v153 )
            {
              v154 = 1;
              do
              {
                if ( v227.m128i_i32[2] == *(_DWORD *)(v153 + 8) )
                {
                  if ( !v154 )
                    goto LABEL_184;
                  v153 = *(_QWORD *)(v153 + 32);
                  if ( !v153 )
                    goto LABEL_172;
                  if ( v227.m128i_i32[2] == *(_DWORD *)(v153 + 8) )
                    goto LABEL_184;
                  v154 = 0;
                }
                v153 = *(_QWORD *)(v153 + 32);
              }
              while ( v153 );
              if ( v154 != 1 )
              {
LABEL_172:
                v145 = sub_3405C90(*a1, v205, a3, v7, v199, v194, *(_OWORD *)v110, *(_OWORD *)v111);
                v146 = *a1;
                v218 = v145;
                v147 = *(_QWORD *)(v226.m128i_i64[0] + 40);
                v222 = v148;
                v149 = *(_QWORD *)(v147 + 40);
                v150 = *(_QWORD *)(v147 + 48);
                v151 = sub_3288400(v226.m128i_i64[0], v205);
                return sub_33FCE10(v146, v7, v199, a3, v218, v222, v149, v150, v151, v152);
              }
            }
LABEL_184:
            if ( v226.m128i_i32[2] == v227.m128i_i32[2] && v226.m128i_i64[0] == v227.m128i_i64[0] )
              goto LABEL_172;
          }
        }
      }
      v22 = v227.m128i_i64[1];
      if ( sub_33DFBC0(v227.m128i_i64[0], v227.m128i_i64[1], 0, 0) )
        goto LABEL_17;
LABEL_156:
      v28 = 1;
LABEL_24:
      v11 = v226.m128i_i64[1];
      if ( sub_33DFBC0(v226.m128i_i64[0], v226.m128i_i64[1], 0, 0) && v28 )
      {
        v29 = *(__int16 **)(v13 + 48);
        v30 = *v29;
        v31 = *((_QWORD *)v29 + 1);
        LOWORD(v234) = v30;
        v235 = v31;
        if ( v30 )
        {
          if ( (unsigned __int16)(v30 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v32 = word_4456340[v30 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v234) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v32 = sub_3007130((__int64)&v234, v226.m128i_i64[1]);
        }
        v11 = v32;
        if ( sub_325F060(*(char **)(v13 + 96), v32) )
        {
          v33 = *(_QWORD *)(v13 + 56);
          if ( v33 )
          {
            if ( !*(_QWORD *)(v33 + 32) )
            {
              v34 = *(_QWORD *)(v13 + 40);
              if ( *(_DWORD *)(*(_QWORD *)(v34 + 40) + 24LL) == 51 && *(_DWORD *)(*(_QWORD *)v34 + 24LL) != 157 )
              {
                v35 = v205;
                v36 = v13;
                v37 = sub_3405C90(*a1, v205, a3, v7, v199, v194, *(_OWORD *)&v226, *(_OWORD *)v34);
                v39 = v38;
                v213 = *a1;
LABEL_36:
                v40 = sub_3288400(v36, v35);
                v42 = v41;
                v43 = sub_3288990(*a1, v7, v199);
                return sub_33FCE10(v213, v7, v199, a3, v37, v39, v43, v44, v40, v42);
              }
            }
          }
        }
      }
LABEL_39:
      v46 = *(_DWORD *)(v12 + 24);
      if ( v46 == 160 )
        goto LABEL_40;
      goto LABEL_53;
    }
    v22 = v227.m128i_i64[1];
    if ( sub_33DFBC0(v227.m128i_i64[0], v227.m128i_i64[1], 0, 0) )
    {
      v13 = 0;
LABEL_17:
      v23 = *(__int16 **)(v226.m128i_i64[0] + 48);
      v24 = *v23;
      v25 = *((_QWORD *)v23 + 1);
      LOWORD(v234) = v24;
      v235 = v25;
      if ( v24 )
      {
        if ( (unsigned __int16)(v24 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        v26 = word_4456340[v24 - 1];
      }
      else
      {
        if ( sub_3007100((__int64)&v234) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        v26 = sub_3007130((__int64)&v234, v22);
      }
      if ( sub_325F060(*(char **)(v226.m128i_i64[0] + 96), v26) )
      {
        v27 = *(_QWORD *)(v226.m128i_i64[0] + 56);
        if ( v27 )
        {
          if ( !*(_QWORD *)(v27 + 32) )
          {
            v143 = *(_QWORD *)(v226.m128i_i64[0] + 40);
            if ( *(_DWORD *)(*(_QWORD *)(v143 + 40) + 24LL) == 51 && *(_DWORD *)(*(_QWORD *)v143 + 24LL) != 157 )
            {
              v35 = v205;
              v36 = v226.m128i_i64[0];
              v37 = sub_3405C90(*a1, v205, a3, v7, v199, v194, *(_OWORD *)v143, *(_OWORD *)&v227);
              v39 = v144;
              v213 = *a1;
              goto LABEL_36;
            }
          }
        }
      }
      v28 = v13 != 0;
      goto LABEL_24;
    }
  }
  else
  {
    if ( v14 == 165 )
    {
      sub_33DFBC0(v227.m128i_i64[0], v227.m128i_i64[1], 0, 0);
      goto LABEL_156;
    }
    sub_33DFBC0(v227.m128i_i64[0], v227.m128i_i64[1], 0, 0);
  }
  v11 = v226.m128i_i64[1];
  sub_33DFBC0(v226.m128i_i64[0], v226.m128i_i64[1], 0, 0);
  v46 = *(_DWORD *)(v226.m128i_i64[0] + 24);
  if ( v46 == 160 )
  {
LABEL_40:
    v47 = *(_QWORD *)(v12 + 40);
    if ( *(_DWORD *)(*(_QWORD *)v47 + 24LL) == 51 && *(_DWORD *)(v227.m128i_i64[0] + 24) == 160 )
    {
      v94 = *(_QWORD *)(v227.m128i_i64[0] + 40);
      if ( *(_DWORD *)(*(_QWORD *)v94 + 24LL) == 51
        && *(_QWORD *)(v47 + 80) == *(_QWORD *)(v94 + 80)
        && *(_DWORD *)(v47 + 88) == *(_DWORD *)(v94 + 88)
        && ((unsigned __int8)sub_3286E00(&v226) || (unsigned __int8)sub_3286E00(&v227)) )
      {
        v95 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v47 + 40) + 48LL) + 16LL * *(unsigned int *)(v47 + 48));
        v96 = *v95;
        v97 = *((_QWORD *)v95 + 1);
        v198 = (__int128)_mm_loadu_si128((const __m128i *)(v47 + 40));
        v193 = (__int128)_mm_loadu_si128((const __m128i *)(v94 + 40));
        v98 = *v95;
        v99 = *(_QWORD *)(*(_QWORD *)(v94 + 40) + 48LL) + 16LL * *(unsigned int *)(v94 + 48);
        s1c = (__int128)_mm_loadu_si128((const __m128i *)(v47 + 80));
        if ( *(_WORD *)v99 == v96 && (*(_QWORD *)(v99 + 8) == v97 || v96) )
        {
          v11 = v205;
          v185 = v97;
          if ( (unsigned __int8)sub_328C7F0(a1[1], v205, v96, v97, *((unsigned __int8 *)a1 + 33)) )
          {
            v100 = *a1;
            *(_QWORD *)&v101 = sub_3288990(*a1, v7, v199);
            v217 = v101;
            *(_QWORD *)&v102 = sub_3288990(*a1, v7, v199);
            v104 = sub_3406EB0(v100, v205, a3, v7, v199, v103, v102, v217);
            v106 = v105;
            v107 = v104;
            *(_QWORD *)&v108 = sub_3406EB0(*a1, v205, a3, v98, v185, v185, v198, v193);
            *((_QWORD *)&v166 + 1) = v106;
            *(_QWORD *)&v166 = v107;
            return sub_340F900(*a1, 160, a3, v7, v199, v109, v166, v108, s1c);
          }
        }
      }
    }
LABEL_41:
    v48 = *(unsigned __int16 **)(a2 + 48);
    v49 = *a1;
    v50 = *(__int64 **)(a2 + 40);
    v179 = *((_BYTE *)a1 + 34);
    v51 = *v50;
    v52 = v50[1];
    v53 = v50[5];
    v54 = v50[6];
    v202 = *v50;
    v183 = *(_DWORD *)(a2 + 24);
    v55 = *v48;
    v56 = *((_QWORD *)v48 + 1);
    v228.m128i_i16[0] = v55;
    v228.m128i_i64[1] = v56;
    if ( (_WORD)v55 )
    {
      v191 = 0;
      v206 = word_4456580[v55 - 1];
    }
    else
    {
      v197 = v52;
      v92 = sub_3009970((__int64)&v228, v11, v56, v52, a5);
      v52 = v197;
      v221 = HIWORD(v92);
      v206 = v92;
      v191 = v93;
    }
    HIWORD(v57) = v221;
    LOWORD(v57) = v206;
    v214 = v57;
    v182 = HIWORD(v57);
    s1a = *(_QWORD *)(v49 + 16);
    *(_QWORD *)&v58 = sub_33F2320(v49, v51, v52, &v224);
    v59 = v58;
    v195 = v58;
    v60 = sub_33F2320(v49, v53, v54, &v225);
    v62 = 0;
    v63 = v60;
    v64 = v60;
    v66 = v65;
    if ( *(_DWORD *)(v202 + 24) == 168 )
      LOBYTE(v62) = *(_DWORD *)(v53 + 24) == 168;
    if ( !v60 || !v59 || v224 != v225 )
      return 0;
    v116 = (unsigned __int16 *)(*(_QWORD *)(v59 + 48) + 16LL * DWORD2(v195));
    v117 = *v116;
    v118 = *((_QWORD *)v116 + 1);
    v232.m128i_i16[0] = v117;
    v232.m128i_i64[1] = v118;
    if ( (_WORD)v117 )
    {
      v119 = word_4456580[v117 - 1];
      v120 = 0;
    }
    else
    {
      v171 = v63;
      v173 = v66;
      v176 = v62;
      v119 = sub_3009970((__int64)&v232, v53, v118, v62, v61);
      v63 = v171;
      v66 = v173;
      v62 = v176;
    }
    if ( v119 != v206 || !v206 && v120 != v191 )
      return 0;
    v121 = (unsigned __int16 *)(*(_QWORD *)(v64 + 48) + 16LL * (unsigned int)v66);
    v122 = *v121;
    v123 = *((_QWORD *)v121 + 1);
    LOWORD(v229) = v122;
    v230 = v123;
    if ( (_WORD)v122 )
    {
      v124 = word_4456580[v122 - 1];
      v125 = 0;
    }
    else
    {
      v172 = v63;
      v174 = v66;
      v177 = v62;
      v124 = sub_3009970((__int64)&v229, v206, v123, v62, v61);
      v63 = v172;
      v66 = v174;
      LOBYTE(v62) = v177;
    }
    if ( v206 != v124 || !v206 && v125 != v191 )
      return 0;
    if ( !(_BYTE)v62 )
    {
      v126 = *(__int64 (**)())(*(_QWORD *)s1a + 1696LL);
      if ( v126 == sub_2FE35D0 )
        return 0;
      v175 = v63;
      v178 = v66;
      v127 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD))v126)(
               s1a,
               v228.m128i_u32[0],
               v228.m128i_i64[1],
               v224);
      v63 = v175;
      v66 = v178;
      if ( !v127 )
        return 0;
    }
    v128 = v206;
    if ( !v179 )
    {
      v129 = *(_QWORD *)(v49 + 64);
      v180 = v63;
      v181 = v66;
      v130 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)s1a + 592LL);
      if ( v130 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v234, s1a, v129, v214, v191);
        v128 = v235;
      }
      else
      {
        v128 = v130(s1a, v129, v214, v191);
      }
      v63 = v180;
      v66 = v181;
    }
    if ( v128 != 1 && (!v128 || !*(_QWORD *)(s1a + 8LL * v128 + 112))
      || v183 <= 0x1F3
      && ((*(_BYTE *)(v183 + s1a + 500LL * v128 + 6414) & 0xFB) != 0
       || v183 - 172 <= 1 && (!v206 || !*(_QWORD *)(s1a + 8LL * v206 + 112))) )
    {
      return 0;
    }
    if ( *(_DWORD *)(v202 + 24) == 156 && *(_DWORD *)(v53 + 24) == 156 )
    {
      v234 = v236;
      v229 = v231;
      v219 = v63;
      v230 = 0x1000000000LL;
      v232.m128i_i64[1] = 0x1000000000LL;
      v235 = 0x1000000000LL;
      v223 = v66;
      v232.m128i_i64[0] = (__int64)v233;
      sub_3408690(v49, v195, DWORD2(v195), (unsigned int)&v229, 0, 0, 0, 0);
      sub_3408690(v49, v219, v223, (unsigned int)&v232, 0, 0, 0, 0);
      v156 = (__int128 *)v232.m128i_i64[0];
      v157 = (unsigned __int64)v229;
      HIWORD(v158) = v182;
      v220 = (__int128 *)(v232.m128i_i64[0] + 16LL * v232.m128i_u32[2]);
      v204 = (__int128 *)&v229[16 * (unsigned int)v230];
      while ( 1 )
      {
        v159 = v156;
        v160 = (__int128 *)v157;
        ++v156;
        v157 += 16LL;
        if ( v159 == v220 || v160 == v204 )
          break;
        LOWORD(v158) = v206;
        v161 = sub_3405C90(v49, v183, a3, v158, v191, *(_DWORD *)(a2 + 28), *v160, *v159);
        sub_3050D50((__int64)&v234, v161, v162, v163, v164, v165);
      }
      *((_QWORD *)&v170 + 1) = (unsigned int)v235;
      *(_QWORD *)&v170 = v234;
      result = sub_33FC220(v49, 156, a3, v228.m128i_i32[0], v228.m128i_i32[2], v155, v170);
      if ( v234 != v236 )
      {
        v209 = result;
        _libc_free((unsigned __int64)v234);
        result = v209;
      }
      if ( (_BYTE *)v232.m128i_i64[0] != v233 )
      {
        v210 = result;
        _libc_free(v232.m128i_u64[0]);
        result = v210;
      }
      if ( v229 != v231 )
      {
        v211 = result;
        _libc_free((unsigned __int64)v229);
        result = v211;
      }
LABEL_150:
      if ( result )
        return result;
      return 0;
    }
    *(_QWORD *)&v203 = v63;
    *((_QWORD *)&v203 + 1) = v66;
    *(_QWORD *)&v131 = sub_3400EE0(v49, (int)v224, a3, 0, v61);
    v132 = *((_QWORD *)&v131 + 1);
    v133 = v131;
    *(_QWORD *)&v135 = sub_3406EB0(v49, 158, a3, v214, v191, v134, v195, v131);
    *((_QWORD *)&v168 + 1) = v132;
    *(_QWORD *)&v168 = v133;
    v207 = v135;
    *(_QWORD *)&v137 = sub_3406EB0(v49, 158, a3, v214, v191, v136, v203, v168);
    v139 = sub_3405C90(v49, v183, a3, v214, v191, *(_DWORD *)(a2 + 28), v207, v137);
    v141 = v140;
    v232 = _mm_load_si128(&v228);
    if ( v228.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v228.m128i_i16[0] - 176) <= 0x34u )
      {
LABEL_147:
        if ( *(_DWORD *)(v139 + 24) == 51 )
        {
          v234 = 0;
          LODWORD(v235) = 0;
          v142 = sub_33F17F0(v49, 51, &v234, v232.m128i_u32[0], v232.m128i_i64[1]);
          if ( v234 )
            sub_B91220((__int64)&v234, (__int64)v234);
        }
        else
        {
          *((_QWORD *)&v169 + 1) = v141;
          *(_QWORD *)&v169 = v139;
          v142 = sub_33FAF80(v49, 168, a3, v232.m128i_i32[0], v232.m128i_i32[2], v138, v169);
        }
        result = v142;
        goto LABEL_150;
      }
    }
    else if ( sub_3007100((__int64)&v232) )
    {
      goto LABEL_147;
    }
    result = sub_32886A0(v49, v232.m128i_u32[0], v232.m128i_i64[1], a3, v139, v141);
    goto LABEL_150;
  }
LABEL_53:
  if ( v46 != 159 )
    goto LABEL_41;
  v67 = (_QWORD *)sub_325F230(*(_QWORD *)(v12 + 40), *(unsigned int *)(v12 + 64), 1);
  v11 = (__int64)v68;
  if ( v68 != sub_32600F0(v67, (__int64)v68) )
    goto LABEL_41;
  if ( *(_DWORD *)(v227.m128i_i64[0] + 24) != 159 )
    goto LABEL_41;
  v69 = (_QWORD *)sub_325F230(*(_QWORD *)(v227.m128i_i64[0] + 40), *(unsigned int *)(v227.m128i_i64[0] + 64), 1);
  v11 = (__int64)v70;
  if ( v70 != sub_32600F0(v69, (__int64)v70) )
    goto LABEL_41;
  v71 = *(_QWORD *)(v12 + 56);
  if ( !v71 )
    goto LABEL_109;
  v72 = 1;
  do
  {
    while ( v226.m128i_i32[2] != *(_DWORD *)(v71 + 8) )
    {
      v71 = *(_QWORD *)(v71 + 32);
      if ( !v71 )
        goto LABEL_65;
    }
    if ( !v72 )
      goto LABEL_109;
    v73 = *(_QWORD *)(v71 + 32);
    if ( !v73 )
      goto LABEL_66;
    if ( v226.m128i_i32[2] == *(_DWORD *)(v73 + 8) )
      goto LABEL_109;
    v71 = *(_QWORD *)(v73 + 32);
    v72 = 0;
  }
  while ( v71 );
LABEL_65:
  if ( v72 == 1 )
  {
LABEL_109:
    v114 = *(_QWORD *)(v227.m128i_i64[0] + 56);
    if ( !v114 )
      goto LABEL_41;
    v115 = 1;
    do
    {
      if ( v227.m128i_i32[2] == *(_DWORD *)(v114 + 8) )
      {
        if ( !v115 )
          goto LABEL_41;
        v114 = *(_QWORD *)(v114 + 32);
        if ( !v114 )
          goto LABEL_66;
        if ( v227.m128i_i32[2] == *(_DWORD *)(v114 + 8) )
          goto LABEL_41;
        v115 = 0;
      }
      v114 = *(_QWORD *)(v114 + 32);
    }
    while ( v114 );
    if ( v115 == 1 )
      goto LABEL_41;
  }
LABEL_66:
  v74 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v12 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v12 + 40) + 8LL));
  LODWORD(v75) = *v74;
  a5 = *((_QWORD *)v74 + 1);
  v76 = (unsigned __int16)v75;
  v77 = *(_QWORD *)(**(_QWORD **)(v227.m128i_i64[0] + 40) + 48LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(v227.m128i_i64[0] + 40) + 8LL);
  if ( *(_WORD *)v77 != (_WORD)v75 || *(_QWORD *)(v77 + 8) != a5 && !(_WORD)v75 )
    goto LABEL_41;
  v78 = a1[1];
  if ( (_WORD)v75 == 1 )
  {
    v79 = 1;
    if ( v205 > 0x1F3 )
      goto LABEL_73;
  }
  else
  {
    if ( !(_WORD)v75 )
      goto LABEL_41;
    v79 = (unsigned __int16)v75;
    if ( !*(_QWORD *)(v78 + 8LL * (unsigned __int16)v75 + 112) )
      goto LABEL_41;
    if ( v205 > 0x1F3 )
      goto LABEL_73;
  }
  v80 = *(_BYTE *)(v205 + v78 + 500 * v79 + 6414);
  if ( v80 > 1u && v80 != 4 )
    goto LABEL_41;
LABEL_73:
  v81 = *(unsigned int *)(v12 + 64);
  v234 = v236;
  v235 = 0x400000000LL;
  if ( (_DWORD)v81 )
  {
    v192 = v75;
    v196 = v12;
    v82 = 0;
    v215 = 40 * v81;
    v184 = v7;
    v83 = v205;
    s1b = v227.m128i_i64[0];
    v84 = a5;
    do
    {
      LOWORD(v76) = v192;
      v85 = sub_3406EB0(
              *a1,
              v83,
              a3,
              v76,
              v84,
              v75,
              *(_OWORD *)(v82 + *(_QWORD *)(v196 + 40)),
              *(_OWORD *)(v82 + *(_QWORD *)(s1b + 40)));
      v86 = (unsigned int)v235;
      v75 = v87;
      v88 = (unsigned int)v235 + 1LL;
      if ( v88 > HIDWORD(v235) )
      {
        v208 = v85;
        v212 = v75;
        sub_C8D5F0((__int64)&v234, v236, v88, 0x10u, v85, v75);
        v86 = (unsigned int)v235;
        v85 = v208;
        v75 = v212;
      }
      v89 = (__int64 *)&v234[16 * v86];
      v82 += 40;
      *v89 = v85;
      v89[1] = v75;
      v90 = (unsigned int)(v235 + 1);
      LODWORD(v235) = v235 + 1;
    }
    while ( v215 != v82 );
    v7 = v184;
    v91 = v234;
  }
  else
  {
    v91 = v236;
    v90 = 0;
  }
  *((_QWORD *)&v167 + 1) = v90;
  *(_QWORD *)&v167 = v91;
  result = sub_33FC220(*a1, 159, a3, v7, v199, *a1, v167);
  if ( v234 != v236 )
  {
    v216 = result;
    _libc_free((unsigned __int64)v234);
    return v216;
  }
  return result;
}
