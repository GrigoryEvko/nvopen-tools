// Function: sub_EFA6B0
// Address: 0xefa6b0
//
_QWORD *__fastcall sub_EFA6B0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // rcx
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r14
  _QWORD *v11; // r15
  char v12; // r13
  unsigned __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // ecx
  _QWORD *v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  unsigned int v23; // eax
  _QWORD *v24; // r15
  _QWORD *v25; // rbx
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r12
  unsigned int v30; // eax
  _QWORD *v31; // r9
  _QWORD *v32; // r14
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r15
  unsigned int v37; // eax
  __int64 v38; // r13
  __int64 v39; // r15
  unsigned int v40; // ecx
  unsigned int v41; // edx
  __int64 v42; // r13
  __int64 v43; // rax
  char v44; // r13
  unsigned __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  unsigned int v49; // ecx
  _QWORD *v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r14
  unsigned int v55; // eax
  _QWORD *v56; // r15
  _QWORD *v57; // rbx
  __int64 v58; // r12
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r12
  unsigned int v62; // eax
  _QWORD *v63; // r9
  _QWORD *v64; // r14
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r15
  unsigned int v69; // eax
  __int64 v70; // r13
  __int64 v71; // r15
  unsigned int v72; // ecx
  unsigned int v73; // edx
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rax
  __int64 v77; // rsi
  unsigned __int64 *v78; // rbx
  unsigned __int64 v79; // r12
  unsigned __int64 v80; // rax
  char v81; // r13
  __int64 v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rax
  unsigned int v86; // ecx
  _QWORD *v87; // r15
  __int64 v88; // r12
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r12
  unsigned int v92; // eax
  _QWORD *v93; // r14
  _QWORD *v94; // r15
  __int64 v95; // rbx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rbx
  unsigned int v99; // eax
  _QWORD *v100; // r9
  _QWORD *v101; // r12
  __int64 v102; // r15
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // r14
  unsigned int v106; // eax
  __int64 v107; // r13
  __int64 v108; // r14
  __int64 v109; // rax
  __int64 v110; // rbx
  char v111; // r12
  __int64 v112; // r13
  __int64 v114; // rax
  int *v115; // r12
  __int64 v116; // rdx
  __int64 v117; // r13
  unsigned int v118; // eax
  _QWORD *v119; // r15
  __int64 v120; // r14
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // r14
  unsigned int v124; // eax
  _QWORD *v125; // r9
  __int64 v126; // rbx
  __int64 v127; // rax
  __int64 v128; // rcx
  __int64 v129; // rbx
  unsigned int v130; // eax
  __int64 v131; // r10
  __int64 v132; // rax
  __m128i *v133; // rax
  __m128i *v134; // r12
  __int64 v135; // rax
  __m128i v136; // xmm0
  __m128i v137; // xmm1
  const __m128i *v138; // rdi
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __m128i *v144; // rax
  __m128i *v145; // rcx
  __m128i *v146; // rdx
  __m128i *v147; // rax
  __int64 v148; // rdx
  __int64 v149; // rax
  int *v150; // rdi
  __int64 v151; // rax
  __int64 v152; // rcx
  __int64 v153; // rdx
  __int64 v154; // rax
  __int64 v155; // rdx
  __int64 v156; // rax
  unsigned __int64 v157; // r13
  unsigned __int64 v158; // r14
  _QWORD *v159; // rax
  __int64 v160; // [rsp+0h] [rbp-1B0h]
  __int64 v161; // [rsp+8h] [rbp-1A8h]
  __int64 v162; // [rsp+8h] [rbp-1A8h]
  _QWORD *v163; // [rsp+8h] [rbp-1A8h]
  _QWORD *v164; // [rsp+10h] [rbp-1A0h]
  _QWORD *v165; // [rsp+10h] [rbp-1A0h]
  _QWORD *v166; // [rsp+10h] [rbp-1A0h]
  _QWORD *v167; // [rsp+18h] [rbp-198h]
  _QWORD *v168; // [rsp+18h] [rbp-198h]
  __int64 v169; // [rsp+18h] [rbp-198h]
  __int64 v170; // [rsp+20h] [rbp-190h]
  __int64 v171; // [rsp+20h] [rbp-190h]
  _QWORD *v172; // [rsp+20h] [rbp-190h]
  _QWORD *v173; // [rsp+28h] [rbp-188h]
  _QWORD *v174; // [rsp+28h] [rbp-188h]
  unsigned __int64 v175; // [rsp+28h] [rbp-188h]
  __int64 v176; // [rsp+30h] [rbp-180h]
  _QWORD *v177; // [rsp+38h] [rbp-178h]
  _QWORD *v178; // [rsp+38h] [rbp-178h]
  _QWORD *v179; // [rsp+38h] [rbp-178h]
  char v180; // [rsp+47h] [rbp-169h]
  char v181; // [rsp+47h] [rbp-169h]
  char v182; // [rsp+47h] [rbp-169h]
  _QWORD *v183; // [rsp+48h] [rbp-168h]
  _QWORD *v184; // [rsp+48h] [rbp-168h]
  _QWORD *v185; // [rsp+48h] [rbp-168h]
  __int64 v186; // [rsp+50h] [rbp-160h]
  _QWORD *v187; // [rsp+58h] [rbp-158h]
  _QWORD *v188; // [rsp+58h] [rbp-158h]
  _QWORD *v189; // [rsp+58h] [rbp-158h]
  unsigned __int64 v190; // [rsp+60h] [rbp-150h]
  unsigned __int64 v191; // [rsp+60h] [rbp-150h]
  __int64 v192; // [rsp+60h] [rbp-150h]
  _QWORD *v194; // [rsp+80h] [rbp-130h]
  _QWORD *v195; // [rsp+88h] [rbp-128h]
  __int64 v196; // [rsp+88h] [rbp-128h]
  unsigned __int64 v197; // [rsp+90h] [rbp-120h]
  unsigned __int64 v198; // [rsp+90h] [rbp-120h]
  __int64 v199; // [rsp+90h] [rbp-120h]
  _QWORD *v200; // [rsp+98h] [rbp-118h]
  _QWORD *v202; // [rsp+A0h] [rbp-110h]
  _QWORD *v203; // [rsp+A8h] [rbp-108h]
  _QWORD *v204; // [rsp+B0h] [rbp-100h]
  __int64 v205; // [rsp+B0h] [rbp-100h]
  __int64 v206; // [rsp+B8h] [rbp-F8h]
  _QWORD *v207; // [rsp+B8h] [rbp-F8h]
  size_t v208; // [rsp+B8h] [rbp-F8h]
  _QWORD *v209; // [rsp+B8h] [rbp-F8h]
  bool v210; // [rsp+CFh] [rbp-E1h] BYREF
  __int64 *v211[2]; // [rsp+D0h] [rbp-E0h] BYREF
  __m128i v212[13]; // [rsp+E0h] [rbp-D0h] BYREF

  v2 = a2;
  if ( *(_DWORD *)(a2 + 48) )
  {
    v3 = sub_C1B290(*(__int64 **)(a2 + 32), (__int64 *)(*(_QWORD *)(a2 + 32) + 24LL * *(_QWORD *)(a2 + 40)));
  }
  else
  {
    v115 = *(int **)(a2 + 16);
    v3 = *(_QWORD *)(a2 + 24);
    if ( v115 )
    {
      v208 = *(_QWORD *)(a2 + 24);
      sub_C7D030(v212);
      sub_C7D280(v212[0].m128i_i32, v115, v208);
      sub_C7D290(v212, v211);
      v3 = (unsigned __int64)v211[0];
    }
  }
  v212[0].m128i_i64[0] = v3;
  v4 = sub_C1DD00(a1, v3 % a1[1], v212, v3);
  if ( v4 )
  {
    v5 = (_QWORD *)*v4;
    v200 = v5;
    if ( v5 )
    {
      v206 = a2 + 80;
      v6 = (__int64)(v5 + 12);
      v204 = v5 + 11;
      if ( *(_QWORD *)(a2 + 96) == a2 + 80 )
        goto LABEL_19;
      v7 = *(_QWORD *)(a2 + 96);
      while ( 1 )
      {
        v212[0].m128i_i64[0] = *(_QWORD *)(v7 + 32);
        v8 = v200[13];
        if ( !v8 )
        {
          v9 = v6;
          goto LABEL_222;
        }
        v9 = v6;
        do
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(v8 + 32) < v212[0].m128i_i32[0] )
            {
              v8 = *(_QWORD *)(v8 + 24);
              goto LABEL_13;
            }
            if ( *(_DWORD *)(v8 + 32) == v212[0].m128i_i32[0] && *(_DWORD *)(v8 + 36) < v212[0].m128i_i32[1] )
              break;
            v9 = v8;
            v8 = *(_QWORD *)(v8 + 16);
            if ( !v8 )
              goto LABEL_14;
          }
          v8 = *(_QWORD *)(v8 + 24);
LABEL_13:
          ;
        }
        while ( v8 );
LABEL_14:
        if ( v9 == v6
          || *(_DWORD *)(v9 + 32) > v212[0].m128i_i32[0]
          || *(_DWORD *)(v9 + 32) == v212[0].m128i_i32[0] && *(_DWORD *)(v9 + 36) > v212[0].m128i_i32[1] )
        {
LABEL_222:
          v211[0] = (__int64 *)v212;
          v9 = sub_EFA4B0(v204, v9, v211);
        }
        sub_C1CFB0((unsigned __int64 *)(v9 + 40), (unsigned __int64 *)(v7 + 40), 1u);
        v7 = sub_220EF30(v7);
        if ( v7 == v206 )
        {
          v2 = a2;
          goto LABEL_19;
        }
      }
    }
  }
  v133 = (__m128i *)sub_22077B0(200);
  v134 = v133;
  if ( v133 )
    v133->m128i_i64[0] = 0;
  v135 = v212[0].m128i_i64[0];
  v136 = _mm_loadu_si128((const __m128i *)(a2 + 16));
  v134[6].m128i_i32[0] = 0;
  v137 = _mm_loadu_si128((const __m128i *)(a2 + 32));
  v138 = *(const __m128i **)(a2 + 88);
  v134[7].m128i_i64[0] = (__int64)v134[6].m128i_i64;
  v134->m128i_i64[1] = v135;
  v139 = *(_QWORD *)a2;
  v134[6].m128i_i64[1] = 0;
  v134[1].m128i_i64[0] = v139;
  v140 = *(_QWORD *)(a2 + 8);
  v134[7].m128i_i64[1] = (__int64)v134[6].m128i_i64;
  v134[1].m128i_i64[1] = v140;
  v141 = *(_QWORD *)(a2 + 48);
  v134[8].m128i_i64[0] = 0;
  v134[4].m128i_i64[0] = v141;
  v142 = *(_QWORD *)(a2 + 56);
  v134[2] = v136;
  v134[4].m128i_i64[1] = v142;
  v143 = *(_QWORD *)(a2 + 64);
  v134[3] = v137;
  v134[5].m128i_i64[0] = v143;
  if ( v138 )
  {
    v144 = sub_EF8780(v138, (__int64)v134[6].m128i_i64);
    v145 = v144;
    do
    {
      v146 = v144;
      v144 = (__m128i *)v144[1].m128i_i64[0];
    }
    while ( v144 );
    v134[7].m128i_i64[0] = (__int64)v146;
    v147 = v145;
    do
    {
      v148 = (__int64)v147;
      v147 = (__m128i *)v147[1].m128i_i64[1];
    }
    while ( v147 );
    v149 = *(_QWORD *)(a2 + 112);
    v134[7].m128i_i64[1] = v148;
    v134[6].m128i_i64[1] = (__int64)v145;
    v134[8].m128i_i64[0] = v149;
  }
  v150 = *(int **)(a2 + 136);
  v134[9].m128i_i32[0] = 0;
  v134[9].m128i_i64[1] = 0;
  v134[10].m128i_i64[0] = (__int64)v134[9].m128i_i64;
  v134[10].m128i_i64[1] = (__int64)v134[9].m128i_i64;
  v134[11].m128i_i64[0] = 0;
  if ( v150 )
  {
    v151 = sub_EF8D70(v150, (__int64)v134[9].m128i_i64);
    v152 = v151;
    do
    {
      v153 = v151;
      v151 = *(_QWORD *)(v151 + 16);
    }
    while ( v151 );
    v134[10].m128i_i64[0] = v153;
    v154 = v152;
    do
    {
      v155 = v154;
      v154 = *(_QWORD *)(v154 + 24);
    }
    while ( v154 );
    v156 = *(_QWORD *)(a2 + 160);
    v134[10].m128i_i64[1] = v155;
    v134[9].m128i_i64[1] = v152;
    v134[11].m128i_i64[0] = v156;
  }
  v157 = v134->m128i_u64[1];
  v134[11].m128i_i64[1] = *(_QWORD *)(a2 + 168);
  v158 = v157 % a1[1];
  v159 = sub_C1DD00(a1, v158, &v134->m128i_i64[1], v157);
  if ( v159 && (v200 = (_QWORD *)*v159) != 0 )
  {
    sub_EF8F20((_QWORD *)v134[9].m128i_i64[1]);
    sub_EF88E0((_QWORD *)v134[6].m128i_i64[1]);
    j_j___libc_free_0(v134, 200);
  }
  else
  {
    v200 = sub_EF8460(a1, v158, v157, v134, 1);
  }
  sub_EF8F20((_QWORD *)v200[19]);
  v200[19] = 0;
  v200[20] = v200 + 18;
  v200[21] = v200 + 18;
  v200[22] = 0;
  v200[9] = 0;
LABEL_19:
  v197 = *(_QWORD *)(v2 + 56);
  v176 = v2 + 128;
  if ( *(_QWORD *)(v2 + 144) != v2 + 128 )
  {
    v186 = *(_QWORD *)(v2 + 144);
    v194 = v200 + 11;
    v207 = v200 + 12;
    do
    {
      v10 = v186;
      v11 = *(_QWORD **)(v186 + 64);
      if ( (_QWORD *)(v186 + 48) == v11 )
        goto LABEL_174;
      do
      {
        v195 = v11 + 6;
        v12 = unk_4F838D3;
        if ( !unk_4F838D3 || (v13 = v11[14]) == 0 )
        {
          v14 = v11[26];
          if ( v11[20] )
          {
            v15 = v11[18];
            if ( !v14
              || (v16 = v11[24], v17 = *(_DWORD *)(v16 + 32), *(_DWORD *)(v15 + 32) < v17)
              || *(_DWORD *)(v15 + 32) == v17 && *(_DWORD *)(v15 + 36) < *(_DWORD *)(v16 + 36) )
            {
              v13 = *(_QWORD *)(v15 + 40);
LABEL_64:
              if ( v13 )
                goto LABEL_65;
              goto LABEL_196;
            }
LABEL_28:
            v18 = *(_QWORD **)(v16 + 64);
            v187 = (_QWORD *)(v16 + 48);
            if ( v18 != (_QWORD *)(v16 + 48) )
            {
              v173 = v11;
              v170 = v10;
              v190 = 0;
              while ( 2 )
              {
                if ( v12 )
                {
                  v19 = v18[14];
                  if ( v19 )
                    goto LABEL_62;
                }
                v20 = v18[26];
                if ( v18[20] )
                {
                  v21 = v18[18];
                  if ( !v20
                    || (v22 = v18[24], v23 = *(_DWORD *)(v22 + 32), *(_DWORD *)(v21 + 32) < v23)
                    || *(_DWORD *)(v21 + 32) == v23 && *(_DWORD *)(v21 + 36) < *(_DWORD *)(v22 + 36) )
                  {
                    v19 = *(_QWORD *)(v21 + 40);
LABEL_61:
                    if ( v19 )
                      goto LABEL_62;
LABEL_198:
                    v19 = v18[13] != 0;
LABEL_62:
                    v190 += v19;
                    v18 = (_QWORD *)sub_220EF30(v18);
                    if ( v187 == v18 )
                    {
                      v11 = v173;
                      v10 = v170;
                      v13 = v190;
                      goto LABEL_64;
                    }
                    continue;
                  }
                }
                else
                {
                  if ( !v20 )
                    goto LABEL_198;
                  v22 = v18[24];
                }
                break;
              }
              v24 = *(_QWORD **)(v22 + 64);
              v183 = (_QWORD *)(v22 + 48);
              if ( v24 == (_QWORD *)(v22 + 48) )
                goto LABEL_198;
              v167 = v18;
              v19 = 0;
              v25 = v24;
              while ( 2 )
              {
                if ( v12 )
                {
                  v26 = v25[14];
                  if ( v26 )
                  {
LABEL_59:
                    v19 += v26;
                    v25 = (_QWORD *)sub_220EF30(v25);
                    if ( v183 == v25 )
                    {
                      v18 = v167;
                      goto LABEL_61;
                    }
                    continue;
                  }
                }
                break;
              }
              v27 = v25[26];
              if ( v25[20] )
              {
                v28 = v25[18];
                if ( !v27
                  || (v29 = v25[24], v30 = *(_DWORD *)(v29 + 32), *(_DWORD *)(v28 + 32) < v30)
                  || *(_DWORD *)(v28 + 32) == v30 && *(_DWORD *)(v28 + 36) < *(_DWORD *)(v29 + 36) )
                {
                  v26 = *(_QWORD *)(v28 + 40);
LABEL_58:
                  if ( v26 )
                    goto LABEL_59;
LABEL_211:
                  v26 = v25[13] != 0;
                  goto LABEL_59;
                }
              }
              else
              {
                if ( !v27 )
                  goto LABEL_211;
                v29 = v25[24];
              }
              v31 = *(_QWORD **)(v29 + 64);
              v177 = (_QWORD *)(v29 + 48);
              if ( v31 == (_QWORD *)(v29 + 48) )
                goto LABEL_211;
              v180 = v12;
              v26 = 0;
              v164 = v25;
              v161 = v19;
              v32 = v31;
              while ( 2 )
              {
                if ( v180 )
                {
                  v33 = v32[14];
                  if ( v33 )
                    goto LABEL_56;
                }
                v34 = v32[26];
                if ( v32[20] )
                {
                  v35 = v32[18];
                  if ( !v34
                    || (v36 = v32[24], v37 = *(_DWORD *)(v36 + 32), *(_DWORD *)(v35 + 32) < v37)
                    || *(_DWORD *)(v35 + 32) == v37 && *(_DWORD *)(v35 + 36) < *(_DWORD *)(v36 + 36) )
                  {
                    v33 = *(_QWORD *)(v35 + 40);
                    goto LABEL_55;
                  }
LABEL_52:
                  v38 = *(_QWORD *)(v36 + 64);
                  v39 = v36 + 48;
                  if ( v38 != v39 )
                  {
                    v33 = 0;
                    do
                    {
                      v33 += sub_EF9210((_QWORD *)(v38 + 48));
                      v38 = sub_220EF30(v38);
                    }
                    while ( v39 != v38 );
LABEL_55:
                    if ( v33 )
                    {
LABEL_56:
                      v26 += v33;
                      v32 = (_QWORD *)sub_220EF30(v32);
                      if ( v177 == v32 )
                      {
                        v12 = v180;
                        v25 = v164;
                        v19 = v161;
                        goto LABEL_58;
                      }
                      continue;
                    }
                  }
                }
                else if ( v34 )
                {
                  v36 = v32[24];
                  goto LABEL_52;
                }
                break;
              }
              v33 = v32[13] != 0;
              goto LABEL_56;
            }
          }
          else if ( v14 )
          {
            v16 = v11[24];
            goto LABEL_28;
          }
LABEL_196:
          v13 = v11[13] != 0;
        }
LABEL_65:
        v40 = *(_DWORD *)(v10 + 36);
        v41 = *(_DWORD *)(v10 + 32);
        v42 = (__int64)(v200 + 12);
        v212[0].m128i_i64[0] = __PAIR64__(v40, v41);
        v43 = v200[13];
        if ( !v43 )
          goto LABEL_190;
        do
        {
          while ( 1 )
          {
            if ( v41 > *(_DWORD *)(v43 + 32) )
            {
              v43 = *(_QWORD *)(v43 + 24);
              goto LABEL_71;
            }
            if ( v41 == *(_DWORD *)(v43 + 32) && v40 > *(_DWORD *)(v43 + 36) )
              break;
            v42 = v43;
            v43 = *(_QWORD *)(v43 + 16);
            if ( !v43 )
              goto LABEL_72;
          }
          v43 = *(_QWORD *)(v43 + 24);
LABEL_71:
          ;
        }
        while ( v43 );
LABEL_72:
        if ( (_QWORD *)v42 == v207
          || v41 < *(_DWORD *)(v42 + 32)
          || v41 == *(_DWORD *)(v42 + 32) && v40 < *(_DWORD *)(v42 + 36) )
        {
LABEL_190:
          v211[0] = (__int64 *)v212;
          v42 = sub_EFA5B0(v194, v42, v211);
        }
        *(_QWORD *)(v42 + 40) = sub_C1B1E0(v13, 1u, *(_QWORD *)(v42 + 40), (bool *)v211);
        v44 = unk_4F838D3;
        if ( !unk_4F838D3 || (v45 = v11[14]) == 0 )
        {
          v46 = v11[26];
          if ( v11[20] )
          {
            v47 = v11[18];
            if ( !v46
              || (v48 = v11[24], v49 = *(_DWORD *)(v48 + 32), *(_DWORD *)(v47 + 32) < v49)
              || *(_DWORD *)(v47 + 32) == v49 && *(_DWORD *)(v47 + 36) < *(_DWORD *)(v48 + 36) )
            {
              v45 = *(_QWORD *)(v47 + 40);
LABEL_117:
              if ( v45 )
                goto LABEL_118;
LABEL_194:
              v45 = v11[13] != 0;
              goto LABEL_118;
            }
          }
          else
          {
            if ( !v46 )
              goto LABEL_194;
            v48 = v11[24];
          }
          v50 = *(_QWORD **)(v48 + 64);
          v188 = (_QWORD *)(v48 + 48);
          if ( v50 == (_QWORD *)(v48 + 48) )
            goto LABEL_194;
          v174 = v11;
          v171 = v10;
          v191 = 0;
          while ( 2 )
          {
            if ( v44 )
            {
              v51 = v50[14];
              if ( v51 )
              {
LABEL_115:
                v191 += v51;
                v50 = (_QWORD *)sub_220EF30(v50);
                if ( v188 == v50 )
                {
                  v11 = v174;
                  v10 = v171;
                  v45 = v191;
                  goto LABEL_117;
                }
                continue;
              }
            }
            break;
          }
          v52 = v50[26];
          if ( v50[20] )
          {
            v53 = v50[18];
            if ( !v52
              || (v54 = v50[24], v55 = *(_DWORD *)(v54 + 32), *(_DWORD *)(v53 + 32) < v55)
              || *(_DWORD *)(v53 + 32) == v55 && *(_DWORD *)(v53 + 36) < *(_DWORD *)(v54 + 36) )
            {
              v51 = *(_QWORD *)(v53 + 40);
LABEL_114:
              if ( v51 )
                goto LABEL_115;
LABEL_202:
              v51 = v50[13] != 0;
              goto LABEL_115;
            }
          }
          else
          {
            if ( !v52 )
              goto LABEL_202;
            v54 = v50[24];
          }
          v56 = *(_QWORD **)(v54 + 64);
          v184 = (_QWORD *)(v54 + 48);
          if ( v56 == (_QWORD *)(v54 + 48) )
            goto LABEL_202;
          v168 = v50;
          v51 = 0;
          v57 = v56;
          while ( 2 )
          {
            if ( v44 )
            {
              v58 = v57[14];
              if ( v58 )
              {
LABEL_112:
                v51 += v58;
                v57 = (_QWORD *)sub_220EF30(v57);
                if ( v184 == v57 )
                {
                  v50 = v168;
                  goto LABEL_114;
                }
                continue;
              }
            }
            break;
          }
          v59 = v57[26];
          if ( v57[20] )
          {
            v60 = v57[18];
            if ( !v59
              || (v61 = v57[24], v62 = *(_DWORD *)(v61 + 32), *(_DWORD *)(v60 + 32) < v62)
              || *(_DWORD *)(v60 + 32) == v62 && *(_DWORD *)(v60 + 36) < *(_DWORD *)(v61 + 36) )
            {
              v58 = *(_QWORD *)(v60 + 40);
LABEL_111:
              if ( v58 )
                goto LABEL_112;
LABEL_207:
              v58 = v57[13] != 0;
              goto LABEL_112;
            }
          }
          else
          {
            if ( !v59 )
              goto LABEL_207;
            v61 = v57[24];
          }
          v63 = *(_QWORD **)(v61 + 64);
          v178 = (_QWORD *)(v61 + 48);
          if ( v63 == (_QWORD *)(v61 + 48) )
            goto LABEL_207;
          v181 = v44;
          v58 = 0;
          v165 = v57;
          v162 = v51;
          v64 = v63;
          while ( 2 )
          {
            if ( v181 )
            {
              v65 = v64[14];
              if ( v65 )
                goto LABEL_109;
            }
            v66 = v64[26];
            if ( v64[20] )
            {
              v67 = v64[18];
              if ( !v66
                || (v68 = v64[24], v69 = *(_DWORD *)(v68 + 32), *(_DWORD *)(v67 + 32) < v69)
                || *(_DWORD *)(v67 + 32) == v69 && *(_DWORD *)(v67 + 36) < *(_DWORD *)(v68 + 36) )
              {
                v65 = *(_QWORD *)(v67 + 40);
                goto LABEL_108;
              }
LABEL_105:
              v70 = *(_QWORD *)(v68 + 64);
              v71 = v68 + 48;
              if ( v70 != v71 )
              {
                v65 = 0;
                do
                {
                  v65 += sub_EF9210((_QWORD *)(v70 + 48));
                  v70 = sub_220EF30(v70);
                }
                while ( v71 != v70 );
LABEL_108:
                if ( v65 )
                {
LABEL_109:
                  v58 += v65;
                  v64 = (_QWORD *)sub_220EF30(v64);
                  if ( v178 == v64 )
                  {
                    v44 = v181;
                    v57 = v165;
                    v51 = v162;
                    goto LABEL_111;
                  }
                  continue;
                }
              }
            }
            else if ( v66 )
            {
              v68 = v64[24];
              goto LABEL_105;
            }
            break;
          }
          v65 = v64[13] != 0;
          goto LABEL_109;
        }
LABEL_118:
        v72 = *(_DWORD *)(v10 + 36);
        v73 = *(_DWORD *)(v10 + 32);
        v74 = v11[8];
        v75 = v11[9];
        v211[0] = (__int64 *)__PAIR64__(v72, v73);
        v76 = v200[13];
        v77 = (__int64)(v200 + 12);
        if ( !v76 )
          goto LABEL_188;
        while ( 1 )
        {
LABEL_122:
          if ( v73 > *(_DWORD *)(v76 + 32) )
          {
            v76 = *(_QWORD *)(v76 + 24);
            goto LABEL_124;
          }
          if ( v73 == *(_DWORD *)(v76 + 32) && v72 > *(_DWORD *)(v76 + 36) )
            break;
          v77 = v76;
          v76 = *(_QWORD *)(v76 + 16);
          if ( !v76 )
            goto LABEL_125;
        }
        v76 = *(_QWORD *)(v76 + 24);
LABEL_124:
        if ( v76 )
          goto LABEL_122;
LABEL_125:
        if ( (_QWORD *)v77 == v207
          || v73 < *(_DWORD *)(v77 + 32)
          || v73 == *(_DWORD *)(v77 + 32) && v72 < *(_DWORD *)(v77 + 36) )
        {
LABEL_188:
          v212[0].m128i_i64[0] = (__int64)v211;
          v77 = sub_EFA5B0(v194, v77, (__int64 **)v212);
        }
        v212[0].m128i_i64[0] = v74;
        v212[0].m128i_i64[1] = v75;
        v78 = sub_C1CD30((_QWORD *)(v77 + 48), v212);
        *v78 = sub_C1B1E0(v45, 1u, *v78, &v210);
        v79 = v11[13];
        v80 = 0;
        if ( v79 <= v197 )
          v80 = v197 - v79;
        v198 = v80;
        v81 = unk_4F838D3;
        if ( !unk_4F838D3 || (v82 = v11[14]) == 0 )
        {
          v83 = v11[26];
          if ( v11[20] )
          {
            v84 = v11[18];
            if ( !v83
              || (v85 = v11[24], v86 = *(_DWORD *)(v85 + 32), *(_DWORD *)(v84 + 32) < v86)
              || *(_DWORD *)(v84 + 32) == v86 && *(_DWORD *)(v84 + 36) < *(_DWORD *)(v85 + 36) )
            {
              v82 = *(_QWORD *)(v84 + 40);
LABEL_172:
              if ( v82 )
                goto LABEL_173;
LABEL_192:
              v82 = v79 != 0;
              goto LABEL_173;
            }
          }
          else
          {
            if ( !v83 )
              goto LABEL_192;
            v85 = v11[24];
          }
          v189 = (_QWORD *)(v85 + 48);
          if ( *(_QWORD *)(v85 + 64) == v85 + 48 )
            goto LABEL_192;
          v175 = v11[13];
          v169 = v10;
          v192 = 0;
          v172 = v11;
          v87 = *(_QWORD **)(v85 + 64);
          while ( 2 )
          {
            if ( v81 )
            {
              v88 = v87[14];
              if ( v88 )
              {
LABEL_170:
                v192 += v88;
                v87 = (_QWORD *)sub_220EF30(v87);
                if ( v189 == v87 )
                {
                  v79 = v175;
                  v11 = v172;
                  v10 = v169;
                  v82 = v192;
                  goto LABEL_172;
                }
                continue;
              }
            }
            break;
          }
          v89 = v87[26];
          if ( v87[20] )
          {
            v90 = v87[18];
            if ( !v89
              || (v91 = v87[24], v92 = *(_DWORD *)(v91 + 32), *(_DWORD *)(v90 + 32) < v92)
              || *(_DWORD *)(v90 + 32) == v92 && *(_DWORD *)(v90 + 36) < *(_DWORD *)(v91 + 36) )
            {
              v88 = *(_QWORD *)(v90 + 40);
LABEL_169:
              if ( v88 )
                goto LABEL_170;
LABEL_200:
              v88 = v87[13] != 0;
              goto LABEL_170;
            }
          }
          else
          {
            if ( !v89 )
              goto LABEL_200;
            v91 = v87[24];
          }
          v93 = *(_QWORD **)(v91 + 64);
          v185 = (_QWORD *)(v91 + 48);
          if ( v93 == (_QWORD *)(v91 + 48) )
            goto LABEL_200;
          v166 = v87;
          v88 = 0;
          v94 = v93;
          while ( 2 )
          {
            if ( v81 )
            {
              v95 = v94[14];
              if ( v95 )
              {
LABEL_167:
                v88 += v95;
                v94 = (_QWORD *)sub_220EF30(v94);
                if ( v185 == v94 )
                {
                  v87 = v166;
                  goto LABEL_169;
                }
                continue;
              }
            }
            break;
          }
          v96 = v94[26];
          if ( v94[20] )
          {
            v97 = v94[18];
            if ( !v96
              || (v98 = v94[24], v99 = *(_DWORD *)(v98 + 32), *(_DWORD *)(v97 + 32) < v99)
              || *(_DWORD *)(v97 + 32) == v99 && *(_DWORD *)(v97 + 36) < *(_DWORD *)(v98 + 36) )
            {
              v95 = *(_QWORD *)(v97 + 40);
LABEL_166:
              if ( v95 )
                goto LABEL_167;
LABEL_209:
              v95 = v94[13] != 0;
              goto LABEL_167;
            }
          }
          else
          {
            if ( !v96 )
              goto LABEL_209;
            v98 = v94[24];
          }
          v100 = *(_QWORD **)(v98 + 64);
          v179 = (_QWORD *)(v98 + 48);
          if ( v100 == (_QWORD *)(v98 + 48) )
            goto LABEL_209;
          v182 = v81;
          v95 = 0;
          v163 = v94;
          v160 = v88;
          v101 = v100;
          while ( 2 )
          {
            if ( v182 )
            {
              v102 = v101[14];
              if ( v102 )
                goto LABEL_164;
            }
            v103 = v101[26];
            if ( v101[20] )
            {
              v104 = v101[18];
              if ( !v103
                || (v105 = v101[24], v106 = *(_DWORD *)(v105 + 32), *(_DWORD *)(v104 + 32) < v106)
                || *(_DWORD *)(v104 + 32) == v106 && *(_DWORD *)(v104 + 36) < *(_DWORD *)(v105 + 36) )
              {
                v102 = *(_QWORD *)(v104 + 40);
                goto LABEL_163;
              }
LABEL_160:
              v107 = *(_QWORD *)(v105 + 64);
              v108 = v105 + 48;
              if ( v107 != v108 )
              {
                v102 = 0;
                do
                {
                  v102 += sub_EF9210((_QWORD *)(v107 + 48));
                  v107 = sub_220EF30(v107);
                }
                while ( v108 != v107 );
LABEL_163:
                if ( v102 )
                {
LABEL_164:
                  v95 += v102;
                  v101 = (_QWORD *)sub_220EF30(v101);
                  if ( v179 == v101 )
                  {
                    v81 = v182;
                    v94 = v163;
                    v88 = v160;
                    goto LABEL_166;
                  }
                  continue;
                }
              }
            }
            else if ( v103 )
            {
              v105 = v101[24];
              goto LABEL_160;
            }
            break;
          }
          v102 = v101[13] != 0;
          goto LABEL_164;
        }
LABEL_173:
        v197 = v198 + v82;
        sub_EFA6B0(a1, v195);
        v11 = (_QWORD *)sub_220EF30(v11);
      }
      while ( (_QWORD *)(v186 + 48) != v11 );
LABEL_174:
      v186 = sub_220EF30(v186);
    }
    while ( v176 != v186 );
  }
  v109 = sub_C1B1E0(v197, 1u, v200[9], (bool *)v212[0].m128i_i8);
  v200[9] = v109;
  v110 = v109;
  v111 = unk_4F838D3;
  if ( !unk_4F838D3 || (v112 = v200[10]) == 0 )
  {
    v114 = v200[22];
    if ( v200[16] )
    {
      v116 = v200[14];
      if ( !v114
        || (v117 = v200[20], v118 = *(_DWORD *)(v117 + 32), *(_DWORD *)(v116 + 32) < v118)
        || *(_DWORD *)(v116 + 32) == v118 && *(_DWORD *)(v116 + 36) < *(_DWORD *)(v117 + 36) )
      {
        v112 = *(_QWORD *)(v116 + 40);
LABEL_286:
        if ( v112 )
          goto LABEL_177;
LABEL_186:
        v112 = v110 != 0;
        goto LABEL_177;
      }
    }
    else
    {
      if ( !v114 )
        goto LABEL_186;
      v117 = v200[20];
    }
    v119 = *(_QWORD **)(v117 + 64);
    v209 = (_QWORD *)(v117 + 48);
    if ( v119 == (_QWORD *)(v117 + 48) )
      goto LABEL_186;
    v196 = v110;
    v112 = 0;
    while ( 1 )
    {
      if ( v111 )
      {
        v120 = v119[14];
        if ( v120 )
          goto LABEL_282;
      }
      v121 = v119[26];
      if ( v119[20] )
      {
        v122 = v119[18];
        if ( !v121
          || (v123 = v119[24], v124 = *(_DWORD *)(v123 + 32), *(_DWORD *)(v122 + 32) < v124)
          || *(_DWORD *)(v122 + 32) == v124 && *(_DWORD *)(v122 + 36) < *(_DWORD *)(v123 + 36) )
        {
          v120 = *(_QWORD *)(v122 + 40);
LABEL_281:
          if ( v120 )
            goto LABEL_282;
          goto LABEL_306;
        }
      }
      else
      {
        if ( !v121 )
          goto LABEL_306;
        v123 = v119[24];
      }
      v125 = *(_QWORD **)(v123 + 64);
      v202 = (_QWORD *)(v123 + 48);
      if ( v125 != (_QWORD *)(v123 + 48) )
      {
        v120 = 0;
        while ( 1 )
        {
          if ( v111 )
          {
            v126 = v125[14];
            if ( v126 )
              goto LABEL_280;
          }
          v127 = v125[26];
          if ( v125[20] )
          {
            v128 = v125[18];
            if ( !v127
              || (v129 = v125[24], v130 = *(_DWORD *)(v129 + 32), *(_DWORD *)(v128 + 32) < v130)
              || *(_DWORD *)(v128 + 32) == v130 && *(_DWORD *)(v128 + 36) < *(_DWORD *)(v129 + 36) )
            {
              v126 = *(_QWORD *)(v128 + 40);
              goto LABEL_279;
            }
          }
          else
          {
            if ( !v127 )
            {
LABEL_307:
              v126 = v125[13] != 0;
              goto LABEL_280;
            }
            v129 = v125[24];
          }
          v131 = *(_QWORD *)(v129 + 64);
          v199 = v129 + 48;
          if ( v131 == v129 + 48 )
            goto LABEL_307;
          v126 = 0;
          do
          {
            v203 = v125;
            v205 = v131;
            v126 += sub_EF9210((_QWORD *)(v131 + 48));
            v132 = sub_220EF30(v205);
            v125 = v203;
            v131 = v132;
          }
          while ( v199 != v132 );
LABEL_279:
          if ( !v126 )
            goto LABEL_307;
LABEL_280:
          v120 += v126;
          v125 = (_QWORD *)sub_220EF30(v125);
          if ( v202 == v125 )
            goto LABEL_281;
        }
      }
LABEL_306:
      v120 = v119[13] != 0;
LABEL_282:
      v112 += v120;
      v119 = (_QWORD *)sub_220EF30(v119);
      if ( v209 == v119 )
      {
        v110 = v196;
        goto LABEL_286;
      }
    }
  }
LABEL_177:
  v200[10] = v112;
  return v200;
}
