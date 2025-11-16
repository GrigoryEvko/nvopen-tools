// Function: sub_1CAB590
// Address: 0x1cab590
//
__int64 __fastcall sub_1CAB590(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 i; // rbx
  _BYTE *v8; // rsi
  unsigned __int64 *v9; // r15
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // rax
  unsigned __int8 v14; // dl
  unsigned __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _BOOL4 v18; // r14d
  __int64 v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r8
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rdx
  _QWORD *v29; // r14
  __int64 v30; // rsi
  __int64 v31; // rcx
  int *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  _QWORD *v39; // rdi
  unsigned __int64 *v40; // rdx
  unsigned __int64 *v41; // r8
  unsigned __int64 *v42; // rax
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rsi
  unsigned __int64 *v45; // r8
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rax
  const __m128i *v48; // r8
  __m128i *v49; // rax
  __m128i v50; // kr00_16
  __m128i *v51; // r8
  __m128i *j; // rcx
  __m128i *v53; // rdx
  unsigned __int64 v54; // rdx
  bool v55; // si
  _QWORD *v56; // r8
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // r12
  __int64 v60; // rdi
  _QWORD *v61; // rbx
  const __m128i *v62; // r15
  char *v63; // r13
  unsigned __int64 v64; // r14
  __int64 v65; // r15
  __int64 v66; // rdi
  __int64 v67; // rcx
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r8
  unsigned __int64 v72; // rdx
  _QWORD *v73; // r14
  const __m128i *v74; // rax
  const __m128i *v75; // rdx
  __int64 v76; // rbx
  __int64 v77; // rdi
  __int64 v78; // rax
  __int64 v79; // r13
  __int64 v80; // r12
  __int64 v81; // rdi
  __int64 *v82; // r12
  __int64 v83; // rdi
  __int64 result; // rax
  int v85; // eax
  __int64 v86; // rax
  unsigned int v87; // eax
  __int64 v88; // rsi
  __int64 v89; // r9
  unsigned __int64 v90; // r10
  _QWORD *v91; // rax
  int v92; // eax
  __int64 v93; // rax
  unsigned int v94; // esi
  int v95; // eax
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 v98; // r8
  __m128i *v99; // r14
  __m128i *v100; // rdx
  unsigned __int64 v101; // rdx
  bool v102; // cl
  unsigned __int64 v103; // rdx
  bool v104; // al
  __int64 v105; // r12
  __int64 *v106; // rbx
  _QWORD *v107; // r15
  __int64 v108; // r15
  _QWORD *v109; // rax
  __int64 *v110; // rdx
  _BOOL4 v111; // r9d
  __int64 v112; // rax
  __m128i *v113; // r14
  __m128i *v114; // rdx
  unsigned __int64 v115; // rdx
  bool v116; // cl
  unsigned __int64 v117; // rdx
  bool v118; // al
  _QWORD *v119; // r12
  __int64 v120; // rsi
  __int64 v121; // rdx
  int *v122; // rdi
  __int64 v123; // rax
  __int64 v124; // rcx
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rax
  _QWORD *v129; // rdi
  __int64 v130; // rax
  unsigned int v131; // esi
  unsigned __int64 *v132; // [rsp+20h] [rbp-150h]
  __int64 v133; // [rsp+20h] [rbp-150h]
  _QWORD *v134; // [rsp+28h] [rbp-148h]
  _QWORD *v135; // [rsp+30h] [rbp-140h]
  __int64 v136; // [rsp+30h] [rbp-140h]
  _QWORD *v137; // [rsp+38h] [rbp-138h]
  __int64 *v138; // [rsp+40h] [rbp-130h]
  _QWORD *v139; // [rsp+48h] [rbp-128h]
  __int64 v140; // [rsp+48h] [rbp-128h]
  _QWORD *v141; // [rsp+50h] [rbp-120h]
  __int64 v142; // [rsp+50h] [rbp-120h]
  unsigned __int64 v143; // [rsp+50h] [rbp-120h]
  unsigned __int64 *v144; // [rsp+58h] [rbp-118h]
  __int64 v145; // [rsp+58h] [rbp-118h]
  __int64 v146; // [rsp+58h] [rbp-118h]
  __int64 v147; // [rsp+60h] [rbp-110h]
  const __m128i *v148; // [rsp+60h] [rbp-110h]
  __int64 k; // [rsp+60h] [rbp-110h]
  __int64 v150; // [rsp+60h] [rbp-110h]
  __int64 v151; // [rsp+60h] [rbp-110h]
  __int64 v152; // [rsp+60h] [rbp-110h]
  __int64 v153; // [rsp+60h] [rbp-110h]
  const __m128i *v154; // [rsp+68h] [rbp-108h]
  __int64 v155; // [rsp+78h] [rbp-F8h]
  _QWORD *v156; // [rsp+78h] [rbp-F8h]
  const __m128i *v157; // [rsp+78h] [rbp-F8h]
  const __m128i *v158; // [rsp+78h] [rbp-F8h]
  _BOOL4 v159; // [rsp+78h] [rbp-F8h]
  __int64 v160; // [rsp+78h] [rbp-F8h]
  unsigned __int64 v161; // [rsp+88h] [rbp-E8h] BYREF
  _QWORD *v162; // [rsp+90h] [rbp-E0h] BYREF
  unsigned __int64 v163; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v164; // [rsp+A0h] [rbp-D0h] BYREF
  _BYTE *v165; // [rsp+A8h] [rbp-C8h]
  _BYTE *v166; // [rsp+B0h] [rbp-C0h]
  __m128i v167; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v168; // [rsp+D0h] [rbp-A0h]
  __int64 v169; // [rsp+E0h] [rbp-90h] BYREF
  int v170; // [rsp+E8h] [rbp-88h] BYREF
  int *v171; // [rsp+F0h] [rbp-80h]
  int *v172; // [rsp+F8h] [rbp-78h]
  int *v173; // [rsp+100h] [rbp-70h]
  __int64 v174; // [rsp+108h] [rbp-68h]
  __m128i v175; // [rsp+110h] [rbp-60h] BYREF
  int *m128i_i32; // [rsp+120h] [rbp-50h]
  __int64 *v177; // [rsp+128h] [rbp-48h]
  __int64 *v178; // [rsp+130h] [rbp-40h]
  __int64 v179; // [rsp+138h] [rbp-38h]

  v2 = a1;
  v135 = a1 + 38;
  sub_1C96D30(a1[40]);
  a1[40] = 0;
  v3 = (_QWORD *)a1[34];
  v2[41] = v2 + 39;
  v2[42] = v2 + 39;
  v2[43] = 0;
  v144 = v2 + 39;
  v137 = v2 + 32;
  sub_1C97640(v3);
  v2[34] = 0;
  v4 = (_QWORD *)v2[58];
  v2[35] = v2 + 33;
  v2[36] = v2 + 33;
  v2[37] = 0;
  v154 = (const __m128i *)(v2 + 33);
  v141 = v2 + 56;
  sub_1C973F0(v4);
  v2[58] = 0;
  v2[59] = v2 + 57;
  v2[60] = v2 + 57;
  v2[61] = 0;
  v5 = *(_QWORD *)(a2 + 80);
  v164 = 0;
  v165 = 0;
  v166 = 0;
  if ( !v5 )
    BUG();
  v6 = *(_QWORD *)(v5 + 24);
  for ( i = v5 + 16; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 8) == 53 && (unsigned __int8)sub_1C95B10(v6 - 24) )
    {
      v175.m128i_i64[0] = v6 - 24;
      v175.m128i_i64[1] = v6 - 24;
      m128i_i32 = 0;
      sub_1C99A10((__int64)v135, v175.m128i_i64);
      sub_1C997D0((__int64)v2, v6 - 24, v6 - 24, 0, (__int64)v135);
      v175.m128i_i64[0] = v6 - 24;
      v8 = v165;
      if ( v165 == v166 )
      {
        sub_12879C0((__int64)&v164, v165, &v175);
      }
      else
      {
        if ( v165 )
        {
          *(_QWORD *)v165 = v6 - 24;
          v8 = v165;
        }
        v165 = v8 + 8;
      }
    }
  }
  v9 = &v163;
  v147 = v2[41];
  if ( (unsigned __int64 *)v147 != v144 )
  {
    v10 = v2 + 57;
    v139 = v2;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v147 + 32);
      v171 = 0;
      v170 = 0;
      v172 = &v170;
      v173 = &v170;
      v174 = 0;
      v12 = *(_QWORD *)(v11 + 8);
      v161 = v11;
      if ( !v12 )
        goto LABEL_87;
      do
      {
        while ( 1 )
        {
          v13 = sub_1648700(v12);
          v14 = *((_BYTE *)v13 + 16);
          if ( v14 <= 0x17u )
            BUG();
          if ( v14 != 55 )
            goto LABEL_49;
          v162 = v13;
          v15 = *(v13 - 6);
          v163 = v15;
          v16 = sub_1819210((__int64)&v169, v9);
          if ( v17 )
          {
            v18 = v16 || (int *)v17 == &v170 || v15 < *(_QWORD *)(v17 + 32);
            v155 = v17;
            v19 = sub_22077B0(40);
            *(_QWORD *)(v19 + 32) = v163;
            sub_220F040(v18, v19, v155, &v170);
            ++v174;
          }
          v20 = (_QWORD *)v139[58];
          if ( v20 )
          {
            v21 = v10;
            v22 = (_QWORD *)v139[58];
            do
            {
              while ( 1 )
              {
                v23 = v22[2];
                v24 = v22[3];
                if ( v22[4] >= v163 )
                  break;
                v22 = (_QWORD *)v22[3];
                if ( !v24 )
                  goto LABEL_26;
              }
              v21 = v22;
              v22 = (_QWORD *)v22[2];
            }
            while ( v23 );
LABEL_26:
            if ( v10 != v21 && v21[4] <= v163 )
              break;
          }
          v175.m128i_i32[2] = 0;
          m128i_i32 = 0;
          v177 = &v175.m128i_i64[1];
          v178 = &v175.m128i_i64[1];
          v179 = 0;
          sub_1C98BA0(&v175, (unsigned __int64 *)&v162);
          v25 = (_QWORD *)v139[58];
          if ( !v25 )
          {
            v26 = (__int64)v10;
LABEL_35:
            v167.m128i_i64[0] = (__int64)v9;
            v26 = sub_1C9B520(v141, (_QWORD *)v26, (unsigned __int64 **)&v167);
            goto LABEL_36;
          }
          v26 = (__int64)v10;
          do
          {
            while ( 1 )
            {
              v27 = v25[2];
              v28 = v25[3];
              if ( v25[4] >= v163 )
                break;
              v25 = (_QWORD *)v25[3];
              if ( !v28 )
                goto LABEL_33;
            }
            v26 = (__int64)v25;
            v25 = (_QWORD *)v25[2];
          }
          while ( v27 );
LABEL_33:
          if ( v10 == (_QWORD *)v26 || *(_QWORD *)(v26 + 32) > v163 )
            goto LABEL_35;
LABEL_36:
          if ( (__m128i *)(v26 + 40) != &v175 )
          {
            v29 = *(_QWORD **)(v26 + 56);
            v30 = v26 + 48;
            v167.m128i_i64[0] = (__int64)v29;
            v31 = *(_QWORD *)(v26 + 72);
            v168 = v26 + 40;
            v167.m128i_i64[1] = v31;
            if ( v29 )
            {
              v29[1] = 0;
              if ( *(_QWORD *)(v31 + 16) )
                v167.m128i_i64[1] = *(_QWORD *)(v31 + 16);
              *(_QWORD *)(v26 + 56) = 0;
              *(_QWORD *)(v26 + 64) = v30;
              *(_QWORD *)(v26 + 72) = v30;
              *(_QWORD *)(v26 + 80) = 0;
              v32 = m128i_i32;
              if ( m128i_i32 )
              {
LABEL_41:
                v156 = (_QWORD *)v26;
                v33 = sub_1C95BA0(v32, v30, &v167);
                v34 = v33;
                do
                {
                  v35 = v33;
                  v33 = *(_QWORD *)(v33 + 16);
                }
                while ( v33 );
                v156[8] = v35;
                v36 = v34;
                do
                {
                  v37 = v36;
                  v36 = *(_QWORD *)(v36 + 24);
                }
                while ( v36 );
                v156[9] = v37;
                v38 = v179;
                v156[7] = v34;
                v156[10] = v38;
                v29 = (_QWORD *)v167.m128i_i64[0];
                if ( !v167.m128i_i64[0] )
                  goto LABEL_47;
              }
              do
              {
                sub_1C97220(v29[3]);
                v39 = v29;
                v29 = (_QWORD *)v29[2];
                j_j___libc_free_0(v39, 40);
              }
              while ( v29 );
              goto LABEL_47;
            }
            v167 = 0u;
            *(_QWORD *)(v26 + 56) = 0;
            *(_QWORD *)(v26 + 64) = v30;
            *(_QWORD *)(v26 + 72) = v30;
            *(_QWORD *)(v26 + 80) = 0;
            v32 = m128i_i32;
            if ( !m128i_i32 )
              goto LABEL_48;
            goto LABEL_41;
          }
LABEL_47:
          v32 = m128i_i32;
LABEL_48:
          sub_1C97220((__int64)v32);
LABEL_49:
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            goto LABEL_50;
        }
        v56 = v10;
        do
        {
          while ( 1 )
          {
            v57 = v20[2];
            v58 = v20[3];
            if ( v20[4] >= v163 )
              break;
            v20 = (_QWORD *)v20[3];
            if ( !v58 )
              goto LABEL_79;
          }
          v56 = v20;
          v20 = (_QWORD *)v20[2];
        }
        while ( v57 );
LABEL_79:
        if ( v10 == v56 || v56[4] > v163 )
        {
          v175.m128i_i64[0] = (__int64)v9;
          v56 = (_QWORD *)sub_1C9B520(v141, v56, (unsigned __int64 **)&v175);
        }
        sub_1C98BA0(v56 + 5, (unsigned __int64 *)&v162);
        v12 = *(_QWORD *)(v12 + 8);
      }
      while ( v12 );
LABEL_50:
      if ( !v174 )
        goto LABEL_85;
      v40 = (unsigned __int64 *)v139[40];
      if ( !v40 )
        goto LABEL_85;
      v41 = v144;
      v42 = (unsigned __int64 *)v139[40];
      do
      {
        while ( 1 )
        {
          v43 = v42[2];
          v44 = v42[3];
          if ( v42[4] >= v161 )
            break;
          v42 = (unsigned __int64 *)v42[3];
          if ( !v44 )
            goto LABEL_56;
        }
        v41 = v42;
        v42 = (unsigned __int64 *)v42[2];
      }
      while ( v43 );
LABEL_56:
      if ( v144 != v41 && v41[4] <= v161 )
      {
        v45 = v144;
        do
        {
          while ( 1 )
          {
            v46 = v40[2];
            v47 = v40[3];
            if ( v40[4] >= v161 )
              break;
            v40 = (unsigned __int64 *)v40[3];
            if ( !v47 )
              goto LABEL_62;
          }
          v45 = v40;
          v40 = (unsigned __int64 *)v40[2];
        }
        while ( v46 );
LABEL_62:
        if ( v144 == v45 || v45[4] > v161 )
        {
          v175.m128i_i64[0] = (__int64)&v161;
          v45 = sub_1C9AC70(v135, v45, (unsigned __int64 **)&v175);
        }
        v167 = _mm_loadu_si128((const __m128i *)(v45 + 5));
        v48 = (const __m128i *)sub_1C98E50((__int64)v137, (unsigned __int64 *)&v167);
        v49 = (__m128i *)v139[34];
        if ( v154 != v48 )
        {
          if ( !v49 )
          {
            v51 = (__m128i *)v154;
            goto LABEL_161;
          }
          v50 = v167;
          v51 = (__m128i *)v154;
          for ( j = (__m128i *)v139[34]; ; j = v53 )
          {
            v54 = j[2].m128i_u64[0];
            v55 = v54 < v167.m128i_i64[0];
            if ( v54 == v167.m128i_i64[0] )
              v55 = j[2].m128i_i64[1] < (unsigned __int64)v167.m128i_i64[1];
            v53 = (__m128i *)j[1].m128i_i64[1];
            if ( !v55 )
            {
              v53 = (__m128i *)j[1].m128i_i64[0];
              v51 = j;
            }
            if ( !v53 )
              break;
          }
          if ( v154 == v51 )
          {
LABEL_161:
            v175.m128i_i64[0] = (__int64)&v167;
            v98 = sub_1C9CF10(v137, v51, (const __m128i **)&v175) + 56;
            v49 = (__m128i *)v139[34];
            if ( v49 )
            {
              v50 = v167;
              goto LABEL_153;
            }
            v99 = (__m128i *)v154;
          }
          else
          {
            if ( v51[2].m128i_i64[0] != v167.m128i_i64[0] )
            {
              if ( v51[2].m128i_i64[0] <= (unsigned __int64)v167.m128i_i64[0] )
                goto LABEL_152;
              goto LABEL_161;
            }
            if ( v51[2].m128i_i64[1] > (unsigned __int64)v167.m128i_i64[1] )
              goto LABEL_161;
LABEL_152:
            v98 = (__int64)&v51[3].m128i_i64[1];
LABEL_153:
            v99 = (__m128i *)v154;
            while ( 1 )
            {
              v101 = v49[2].m128i_u64[0];
              v102 = v101 < v50.m128i_i64[0];
              if ( v101 == v50.m128i_i64[0] )
                v102 = v49[2].m128i_i64[1] < (unsigned __int64)v50.m128i_i64[1];
              v100 = (__m128i *)v49[1].m128i_i64[1];
              if ( !v102 )
              {
                v100 = (__m128i *)v49[1].m128i_i64[0];
                v99 = v49;
              }
              if ( !v100 )
                break;
              v49 = v100;
            }
            if ( v154 != v99 )
            {
              v103 = v99[2].m128i_u64[0];
              v104 = v103 > v50.m128i_i64[0];
              if ( v103 == v50.m128i_i64[0] )
                v104 = v99[2].m128i_i64[1] > (unsigned __int64)v50.m128i_i64[1];
              if ( !v104 )
              {
LABEL_167:
                v105 = (__int64)v172;
                if ( v172 != &v170 )
                {
                  v134 = v10;
                  v106 = &v99[3].m128i_i64[1];
                  v132 = v9;
                  v107 = (_QWORD *)v98;
                  do
                  {
                    v109 = sub_1819AD0((__m128i *)v99[3].m128i_i64, v107, (unsigned __int64 *)(v105 + 32));
                    v108 = (__int64)v109;
                    if ( v110 )
                    {
                      v111 = 1;
                      if ( !v109 && v110 != v106 )
                        v111 = *(_QWORD *)(v105 + 32) < (unsigned __int64)v110[4];
                      v138 = v110;
                      v159 = v111;
                      v108 = sub_22077B0(40);
                      *(_QWORD *)(v108 + 32) = *(_QWORD *)(v105 + 32);
                      sub_220F040(v159, v108, v138, v106);
                      ++v99[5].m128i_i64[1];
                    }
                    v107 = (_QWORD *)sub_220EF30(v108);
                    v105 = sub_220EF30(v105);
                  }
                  while ( (int *)v105 != &v170 );
                  v10 = v134;
                  v9 = v132;
                }
                goto LABEL_85;
              }
            }
          }
          v160 = v98;
          v175.m128i_i64[0] = (__int64)&v167;
          v112 = sub_1C9CF10(v137, v99, (const __m128i **)&v175);
          v98 = v160;
          v99 = (__m128i *)v112;
          goto LABEL_167;
        }
        if ( !v49 )
        {
          v113 = (__m128i *)v154;
          goto LABEL_191;
        }
        v113 = (__m128i *)v154;
        while ( 1 )
        {
          v115 = v49[2].m128i_u64[0];
          v116 = v115 < v167.m128i_i64[0];
          if ( v115 == v167.m128i_i64[0] )
            v116 = v49[2].m128i_i64[1] < (unsigned __int64)v167.m128i_i64[1];
          v114 = (__m128i *)v49[1].m128i_i64[1];
          if ( !v116 )
          {
            v114 = (__m128i *)v49[1].m128i_i64[0];
            v113 = v49;
          }
          if ( !v114 )
            break;
          v49 = v114;
        }
        if ( v154 == v113 )
          goto LABEL_191;
        v117 = v113[2].m128i_u64[0];
        v118 = v117 > v167.m128i_i64[0];
        if ( v117 == v167.m128i_i64[0] )
          v118 = v113[2].m128i_i64[1] > (unsigned __int64)v167.m128i_i64[1];
        if ( v118 )
        {
LABEL_191:
          v175.m128i_i64[0] = (__int64)&v167;
          v113 = (__m128i *)sub_1C9CF10(v137, v113, (const __m128i **)&v175);
        }
        if ( &v113[3] == (__m128i *)&v169 )
          goto LABEL_85;
        v119 = (_QWORD *)v113[4].m128i_i64[0];
        v120 = (__int64)&v113[3].m128i_i64[1];
        v175.m128i_i64[0] = (__int64)v119;
        v121 = v113[5].m128i_i64[0];
        m128i_i32 = v113[3].m128i_i32;
        v175.m128i_i64[1] = v121;
        if ( v119 )
        {
          v119[1] = 0;
          if ( *(_QWORD *)(v121 + 16) )
            v175.m128i_i64[1] = *(_QWORD *)(v121 + 16);
          v113[4].m128i_i64[0] = 0;
          v113[4].m128i_i64[1] = v120;
          v113[5].m128i_i64[0] = v120;
          v113[5].m128i_i64[1] = 0;
          v122 = v171;
          if ( v171 )
            goto LABEL_197;
          do
          {
LABEL_202:
            sub_1C97470(v119[3]);
            v129 = v119;
            v119 = (_QWORD *)v119[2];
            j_j___libc_free_0(v129, 40);
          }
          while ( v119 );
          goto LABEL_85;
        }
        v175 = 0u;
        v113[4].m128i_i64[0] = 0;
        v113[4].m128i_i64[1] = v120;
        v113[5].m128i_i64[0] = v120;
        v113[5].m128i_i64[1] = 0;
        v122 = v171;
        if ( v171 )
        {
LABEL_197:
          v123 = sub_1C95DD0(v122, v120, &v175);
          v124 = v123;
          do
          {
            v125 = v123;
            v123 = *(_QWORD *)(v123 + 16);
          }
          while ( v123 );
          v113[4].m128i_i64[1] = v125;
          v126 = v124;
          do
          {
            v127 = v126;
            v126 = *(_QWORD *)(v126 + 24);
          }
          while ( v126 );
          v113[5].m128i_i64[0] = v127;
          v128 = v174;
          v113[4].m128i_i64[0] = v124;
          v113[5].m128i_i64[1] = v128;
          v119 = (_QWORD *)v175.m128i_i64[0];
          if ( !v175.m128i_i64[0] )
            goto LABEL_85;
          goto LABEL_202;
        }
      }
      else
      {
LABEL_85:
        v59 = (__int64)v171;
        if ( v171 )
        {
          do
          {
            sub_1C97470(*(_QWORD *)(v59 + 24));
            v60 = v59;
            v59 = *(_QWORD *)(v59 + 16);
            j_j___libc_free_0(v60, 40);
          }
          while ( v59 );
        }
      }
LABEL_87:
      v147 = sub_220EEE0(v147);
      if ( (unsigned __int64 *)v147 == v144 )
      {
        v2 = v139;
        break;
      }
    }
  }
  v171 = 0;
  v61 = v2;
  v174 = 0;
  v62 = (const __m128i *)v2[35];
  v170 = 0;
  v172 = &v170;
  v173 = &v170;
  if ( v62 == v154 )
  {
    v175.m128i_i32[2] = 0;
    v83 = 0;
    m128i_i32 = 0;
    v177 = &v175.m128i_i64[1];
    v178 = &v175.m128i_i64[1];
    v179 = 0;
  }
  else
  {
    do
    {
      v63 = &v62[3].m128i_i8[8];
      v64 = 0;
      v175 = _mm_loadu_si128(v62 + 2);
      if ( &v62[3].m128i_u64[1] != (unsigned __int64 *)v62[4].m128i_i64[1] )
      {
        v157 = v62;
        v65 = v62[4].m128i_i64[1];
        do
        {
          v66 = v61[44];
          v67 = 1;
          v68 = **(_QWORD **)(v65 + 32);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v68 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v86 = *(_QWORD *)(v68 + 32);
                v68 = *(_QWORD *)(v68 + 24);
                v67 *= v86;
                continue;
              case 1:
                v69 = 16;
                break;
              case 2:
                v69 = 32;
                break;
              case 3:
              case 9:
                v69 = 64;
                break;
              case 4:
                v69 = 80;
                break;
              case 5:
              case 6:
                v69 = 128;
                break;
              case 7:
                v153 = v67;
                v92 = sub_15A9520(v66, 0);
                v67 = v153;
                v69 = (unsigned int)(8 * v92);
                break;
              case 0xB:
                v69 = *(_DWORD *)(v68 + 8) >> 8;
                break;
              case 0xD:
                v152 = v67;
                v91 = (_QWORD *)sub_15A9930(v66, v68);
                v67 = v152;
                v69 = 8LL * *v91;
                break;
              case 0xE:
                v140 = v67;
                v142 = *(_QWORD *)(v68 + 24);
                v151 = *(_QWORD *)(v68 + 32);
                v87 = sub_15A9FE0(v66, v142);
                v88 = v142;
                v89 = 1;
                v67 = v140;
                v90 = v87;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v88 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v96 = *(_QWORD *)(v88 + 32);
                      v88 = *(_QWORD *)(v88 + 24);
                      v89 *= v96;
                      continue;
                    case 1:
                      v93 = 16;
                      goto LABEL_136;
                    case 2:
                      v93 = 32;
                      goto LABEL_136;
                    case 3:
                    case 9:
                      v93 = 64;
                      goto LABEL_136;
                    case 4:
                      v93 = 80;
                      goto LABEL_136;
                    case 5:
                    case 6:
                      v93 = 128;
                      goto LABEL_136;
                    case 7:
                      v94 = 0;
                      v143 = v90;
                      v145 = v89;
                      goto LABEL_139;
                    case 0xB:
                      v93 = *(_DWORD *)(v88 + 8) >> 8;
                      goto LABEL_136;
                    case 0xD:
                      JUMPOUT(0x1CAC1FD);
                    case 0xE:
                      v136 = *(_QWORD *)(v88 + 24);
                      sub_15A9FE0(v66, v136);
                      v146 = 1;
                      v97 = v136;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v97 + 8) )
                        {
                          case 0:
                          case 8:
                          case 0xA:
                          case 0xC:
                          case 0x10:
                            v130 = v146 * *(_QWORD *)(v97 + 32);
                            v97 = *(_QWORD *)(v97 + 24);
                            v146 = v130;
                            continue;
                          case 1:
                          case 2:
                          case 4:
                          case 5:
                          case 6:
                          case 0xB:
                            goto LABEL_223;
                          case 3:
                          case 9:
                            JUMPOUT(0x1CAC6FA);
                          case 7:
                            v131 = 0;
                            goto LABEL_209;
                          case 0xD:
                            sub_15A9930(v66, v97);
                            goto LABEL_223;
                          case 0xE:
                            v133 = *(_QWORD *)(v97 + 24);
                            sub_15A9FE0(v66, v133);
                            sub_127FA20(v66, v133);
                            goto LABEL_223;
                          case 0xF:
                            v131 = *(_DWORD *)(v97 + 8) >> 8;
LABEL_209:
                            sub_15A9520(v66, v131);
LABEL_223:
                            JUMPOUT(0x1CAC6FF);
                        }
                      }
                    case 0xF:
                      v143 = v90;
                      v145 = v89;
                      v94 = *(_DWORD *)(v88 + 8) >> 8;
LABEL_139:
                      v95 = sub_15A9520(v66, v94);
                      v89 = v145;
                      v90 = v143;
                      v67 = v140;
                      v93 = (unsigned int)(8 * v95);
LABEL_136:
                      v69 = 8 * v151 * v90 * ((v90 + ((unsigned __int64)(v93 * v89 + 7) >> 3) - 1) / v90);
                      break;
                  }
                  break;
                }
                break;
              case 0xF:
                v150 = v67;
                v85 = sub_15A9520(v66, *(_DWORD *)(v68 + 8) >> 8);
                v67 = v150;
                v69 = (unsigned int)(8 * v85);
                break;
            }
            break;
          }
          v70 = v69 * v67;
          if ( v64 < (unsigned __int64)(v70 + 7) >> 3 )
            v64 = (unsigned __int64)(v70 + 7) >> 3;
          v65 = sub_220EF30(v65);
        }
        while ( v63 != (char *)v65 );
        v62 = v157;
      }
      if ( (const __m128i *)v61[35] != v154 )
      {
        v71 = v61[35];
        do
        {
          while ( 1 )
          {
            if ( v62 != (const __m128i *)v71 && *(_QWORD *)(v71 + 32) == v62[2].m128i_i64[0] )
            {
              v72 = *(_QWORD *)(v71 + 40);
              if ( v175.m128i_i64[1] <= v72 && v64 + v175.m128i_i64[1] > v72 )
                break;
            }
            v71 = sub_220EEE0(v71);
            if ( (const __m128i *)v71 == v154 )
              goto LABEL_107;
          }
          v148 = (const __m128i *)v71;
          sub_1C98EE0(&v169, &v175);
          sub_1C98EE0(&v169, v148 + 2);
          v71 = sub_220EEE0(v148);
        }
        while ( (const __m128i *)v71 != v154 );
      }
LABEL_107:
      v62 = (const __m128i *)sub_220EEE0(v62);
    }
    while ( v62 != v154 );
    v73 = v61;
    for ( k = (__int64)v172; (int *)k != &v170; k = sub_220EF30(k) )
    {
      v74 = (const __m128i *)sub_1C9B0F0((__int64)v137, (unsigned __int64 *)(k + 32));
      v158 = v75;
      v76 = (__int64)v74;
      if ( (const __m128i *)v73[35] == v74 && v154 == v75 )
      {
        sub_1C97640((_QWORD *)v73[34]);
        v73[35] = v154;
        v73[34] = 0;
        v73[36] = v154;
        v73[37] = 0;
      }
      else if ( v75 != v74 )
      {
        do
        {
          v77 = v76;
          v76 = sub_220EF30(v76);
          v78 = sub_220F330(v77, v154);
          v79 = *(_QWORD *)(v78 + 64);
          v80 = v78;
          while ( v79 )
          {
            sub_1C97470(*(_QWORD *)(v79 + 24));
            v81 = v79;
            v79 = *(_QWORD *)(v79 + 16);
            j_j___libc_free_0(v81, 40);
          }
          j_j___libc_free_0(v80, 96);
          --v73[37];
        }
        while ( v158 != (const __m128i *)v76 );
      }
    }
    v82 = (__int64 *)v73[35];
    v175.m128i_i32[2] = 0;
    m128i_i32 = 0;
    v177 = &v175.m128i_i64[1];
    v178 = &v175.m128i_i64[1];
    v179 = 0;
    if ( v154 == (const __m128i *)v82 )
    {
      v83 = 0;
    }
    else
    {
      do
      {
        if ( (unsigned __int64)v82[11] > 1 )
          sub_1C9C2F0(v73, v82 + 6, v82[4], v82[5], &v175);
        v82 = (__int64 *)sub_220EEE0(v82);
      }
      while ( v82 != (__int64 *)v154 );
      v83 = (__int64)m128i_i32;
    }
  }
  sub_1C961D0(v83);
  result = sub_1C963A0((__int64)v171);
  if ( v164 )
    return j_j___libc_free_0(v164, &v166[-v164]);
  return result;
}
