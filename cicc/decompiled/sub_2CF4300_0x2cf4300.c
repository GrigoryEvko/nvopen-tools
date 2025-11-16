// Function: sub_2CF4300
// Address: 0x2cf4300
//
void __fastcall sub_2CF4300(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 i; // rbx
  __int64 v8; // r9
  _BYTE *v9; // rsi
  unsigned __int64 *v10; // r15
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rax
  __int64 v13; // r12
  _BYTE *v14; // rdx
  unsigned __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 *v17; // rdx
  char v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r9
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // r9
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
  unsigned __int64 v39; // rdi
  unsigned __int64 *v40; // rdx
  unsigned __int64 *v41; // r8
  unsigned __int64 *v42; // rax
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rsi
  unsigned __int64 *v45; // r8
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rax
  __m128i *v48; // r14
  __m128i *v49; // r8
  __m128i *v50; // rax
  __m128i v51; // kr00_16
  __m128i *j; // rcx
  __m128i *v53; // rdx
  unsigned __int64 v54; // rdx
  bool v55; // si
  __int64 v56; // r9
  __int64 v57; // rsi
  __int64 v58; // rax
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rdi
  const __m128i *v61; // r12
  unsigned __int64 v62; // rbx
  __int64 k; // r13
  __int64 v64; // rax
  __int64 v65; // rdx
  unsigned __int64 v66; // rax
  const __m128i *m; // r13
  unsigned __int64 v68; // rdx
  __m128i *v69; // rax
  __m128i *v70; // rdx
  __int64 v71; // rbx
  int *v72; // rdi
  int *v73; // rax
  unsigned __int64 v74; // r13
  unsigned __int64 v75; // r12
  unsigned __int64 v76; // rdi
  __m128i *v77; // r12
  unsigned __int64 v78; // rdi
  __int64 v79; // r14
  __m128i *v80; // r9
  __m128i *v81; // rdx
  unsigned __int64 v82; // rdx
  bool v83; // cl
  unsigned __int64 v84; // rdx
  bool v85; // al
  __int64 *v86; // r12
  __m128i *v87; // r13
  __m128i *v88; // rbx
  __int64 v89; // r15
  __int64 v90; // r15
  _QWORD *v91; // rax
  __int64 *v92; // rdx
  char v93; // r14
  __m128i *v94; // r14
  __m128i *v95; // rdx
  unsigned __int64 v96; // rdx
  bool v97; // cl
  unsigned __int64 v98; // rdx
  bool v99; // al
  unsigned __int64 *v100; // r12
  __int64 v101; // rsi
  __int64 v102; // rdx
  int *v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rcx
  __int64 v106; // rdx
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rax
  unsigned __int64 v110; // rdi
  unsigned __int64 *v111; // [rsp+0h] [rbp-150h]
  _QWORD *v112; // [rsp+8h] [rbp-148h]
  _QWORD *v113; // [rsp+10h] [rbp-140h]
  __int64 *v114; // [rsp+18h] [rbp-138h]
  _QWORD *v115; // [rsp+20h] [rbp-130h]
  _QWORD *v116; // [rsp+28h] [rbp-128h]
  _QWORD *v117; // [rsp+30h] [rbp-120h]
  unsigned __int64 *v118; // [rsp+38h] [rbp-118h]
  __int64 v119; // [rsp+40h] [rbp-110h]
  __int64 *n; // [rsp+40h] [rbp-110h]
  __int64 *v121; // [rsp+50h] [rbp-100h]
  _QWORD *v122; // [rsp+50h] [rbp-100h]
  __m128i *v123; // [rsp+50h] [rbp-100h]
  __int64 *v124; // [rsp+50h] [rbp-100h]
  __m128i *v125; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v126; // [rsp+68h] [rbp-E8h] BYREF
  unsigned __int64 v127; // [rsp+70h] [rbp-E0h] BYREF
  unsigned __int64 v128; // [rsp+78h] [rbp-D8h] BYREF
  unsigned __int64 v129; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE *v130; // [rsp+88h] [rbp-C8h]
  _BYTE *v131; // [rsp+90h] [rbp-C0h]
  __m128i v132; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+B0h] [rbp-A0h]
  __int64 v134; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-88h] BYREF
  int *v136; // [rsp+D0h] [rbp-80h]
  __int64 *v137; // [rsp+D8h] [rbp-78h]
  __int64 *v138; // [rsp+E0h] [rbp-70h]
  __int64 v139; // [rsp+E8h] [rbp-68h]
  __m128i *v140; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v141; // [rsp+F8h] [rbp-58h] BYREF
  int *m128i_i32; // [rsp+100h] [rbp-50h]
  __int64 *v143; // [rsp+108h] [rbp-48h]
  __int64 *v144; // [rsp+110h] [rbp-40h]
  __int64 v145; // [rsp+118h] [rbp-38h]

  v2 = a1;
  v113 = a1 + 38;
  sub_2CDEC30(a1[40]);
  a1[40] = 0;
  v3 = (_QWORD *)a1[34];
  v2[41] = v2 + 39;
  v2[42] = v2 + 39;
  v2[43] = 0;
  v118 = v2 + 39;
  v115 = v2 + 32;
  sub_2CDF550(v3);
  v2[34] = 0;
  v4 = (_QWORD *)v2[58];
  v2[35] = v2 + 33;
  v2[36] = v2 + 33;
  v2[37] = 0;
  v125 = (__m128i *)(v2 + 33);
  v117 = v2 + 56;
  sub_2CDF2A0(v4);
  v2[58] = 0;
  v2[59] = v2 + 57;
  v2[60] = v2 + 57;
  v2[61] = 0;
  v5 = *(_QWORD *)(a2 + 80);
  v129 = 0;
  v130 = 0;
  v131 = 0;
  if ( !v5 )
    BUG();
  v6 = *(_QWORD *)(v5 + 32);
  for ( i = v5 + 24; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 24) == 60 && (unsigned __int8)sub_2CDDA20(v6 - 24) )
    {
      v140 = (__m128i *)(v6 - 24);
      v141 = v6 - 24;
      m128i_i32 = 0;
      sub_2CE1820((__int64)v113, (__int64 *)&v140);
      sub_2CE1620((__int64)v2, v6 - 24, v6 - 24, 0, (__int64)v113, v8);
      v140 = (__m128i *)(v6 - 24);
      v9 = v130;
      if ( v130 == v131 )
      {
        sub_928380((__int64)&v129, v130, &v140);
      }
      else
      {
        if ( v130 )
        {
          *(_QWORD *)v130 = v6 - 24;
          v9 = v130;
        }
        v130 = v9 + 8;
      }
    }
  }
  v10 = &v128;
  v119 = v2[41];
  if ( (unsigned __int64 *)v119 != v118 )
  {
    v11 = v2 + 57;
    v116 = v2;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v119 + 32);
      v136 = 0;
      LODWORD(v135) = 0;
      v137 = &v135;
      v138 = &v135;
      v139 = 0;
      v13 = *(_QWORD *)(v12 + 16);
      v126 = v12;
      if ( !v13 )
        goto LABEL_87;
      do
      {
        while ( 1 )
        {
          v14 = *(_BYTE **)(v13 + 24);
          if ( *v14 <= 0x1Cu )
            BUG();
          if ( *v14 != 62 )
            goto LABEL_49;
          v127 = *(_QWORD *)(v13 + 24);
          v15 = *((_QWORD *)v14 - 8);
          v128 = v15;
          v16 = sub_23FDE00((__int64)&v134, v10);
          if ( v17 )
          {
            v18 = v16 || v17 == &v135 || v15 < v17[4];
            v121 = v17;
            v19 = sub_22077B0(0x28u);
            *(_QWORD *)(v19 + 32) = v128;
            sub_220F040(v18, v19, v121, &v135);
            ++v139;
          }
          v20 = (_QWORD *)v116[58];
          if ( v20 )
          {
            v21 = v11;
            v22 = (_QWORD *)v116[58];
            do
            {
              while ( 1 )
              {
                v23 = v22[2];
                v24 = v22[3];
                if ( v22[4] >= v128 )
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
            if ( v11 != v21 && v21[4] <= v128 )
              break;
          }
          LODWORD(v141) = 0;
          m128i_i32 = 0;
          v143 = &v141;
          v144 = &v141;
          v145 = 0;
          sub_2CE0B60(&v140, &v127);
          v25 = (_QWORD *)v116[58];
          if ( !v25 )
          {
            v26 = (__int64)v11;
LABEL_35:
            v132.m128i_i64[0] = (__int64)v10;
            v26 = sub_2CE3350(v117, v26, (unsigned __int64 **)&v132);
            goto LABEL_36;
          }
          v26 = (__int64)v11;
          do
          {
            while ( 1 )
            {
              v27 = v25[2];
              v28 = v25[3];
              if ( v25[4] >= v128 )
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
          if ( v11 == (_QWORD *)v26 || *(_QWORD *)(v26 + 32) > v128 )
            goto LABEL_35;
LABEL_36:
          if ( (__m128i **)(v26 + 40) != &v140 )
          {
            v29 = *(_QWORD **)(v26 + 56);
            v30 = v26 + 48;
            v132.m128i_i64[0] = (__int64)v29;
            v31 = *(_QWORD *)(v26 + 72);
            v133 = v26 + 40;
            v132.m128i_i64[1] = v31;
            if ( v29 )
            {
              v29[1] = 0;
              if ( *(_QWORD *)(v31 + 16) )
                v132.m128i_i64[1] = *(_QWORD *)(v31 + 16);
              *(_QWORD *)(v26 + 56) = 0;
              *(_QWORD *)(v26 + 64) = v30;
              *(_QWORD *)(v26 + 72) = v30;
              *(_QWORD *)(v26 + 80) = 0;
              v32 = m128i_i32;
              if ( m128i_i32 )
              {
LABEL_41:
                v122 = (_QWORD *)v26;
                v33 = sub_2CDDAA0(v32, v30, &v132);
                v34 = v33;
                do
                {
                  v35 = v33;
                  v33 = *(_QWORD *)(v33 + 16);
                }
                while ( v33 );
                v122[8] = v35;
                v36 = v34;
                do
                {
                  v37 = v36;
                  v36 = *(_QWORD *)(v36 + 24);
                }
                while ( v36 );
                v122[9] = v37;
                v38 = v145;
                v122[7] = v34;
                v122[10] = v38;
                v29 = (_QWORD *)v132.m128i_i64[0];
                if ( !v132.m128i_i64[0] )
                  goto LABEL_47;
              }
              do
              {
                sub_2CDF0D0(v29[3]);
                v39 = (unsigned __int64)v29;
                v29 = (_QWORD *)v29[2];
                j_j___libc_free_0(v39);
              }
              while ( v29 );
              goto LABEL_47;
            }
            v132 = 0u;
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
          sub_2CDF0D0((unsigned __int64)v32);
LABEL_49:
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            goto LABEL_50;
        }
        v56 = (__int64)v11;
        do
        {
          while ( 1 )
          {
            v57 = v20[2];
            v58 = v20[3];
            if ( v20[4] >= v128 )
              break;
            v20 = (_QWORD *)v20[3];
            if ( !v58 )
              goto LABEL_79;
          }
          v56 = (__int64)v20;
          v20 = (_QWORD *)v20[2];
        }
        while ( v57 );
LABEL_79:
        if ( v11 == (_QWORD *)v56 || *(_QWORD *)(v56 + 32) > v128 )
        {
          v140 = (__m128i *)v10;
          v56 = sub_2CE3350(v117, v56, (unsigned __int64 **)&v140);
        }
        sub_2CE0B60((_QWORD *)(v56 + 40), &v127);
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v13 );
LABEL_50:
      if ( !v139 )
        goto LABEL_85;
      v40 = (unsigned __int64 *)v116[40];
      if ( !v40 )
        goto LABEL_85;
      v41 = v118;
      v42 = (unsigned __int64 *)v116[40];
      do
      {
        while ( 1 )
        {
          v43 = v42[2];
          v44 = v42[3];
          if ( v42[4] >= v126 )
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
      if ( v118 != v41 && v41[4] <= v126 )
      {
        v45 = v118;
        do
        {
          while ( 1 )
          {
            v46 = v40[2];
            v47 = v40[3];
            if ( v40[4] >= v126 )
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
        if ( v118 == v45 || v45[4] > v126 )
        {
          v140 = (__m128i *)&v126;
          v45 = sub_2CE2AA0(v113, (__int64)v45, (unsigned __int64 **)&v140);
        }
        v132 = _mm_loadu_si128((const __m128i *)(v45 + 5));
        v48 = v125;
        v49 = (__m128i *)sub_2CE0E10((__int64)v115, (unsigned __int64 *)&v132);
        v50 = (__m128i *)v116[34];
        if ( v125 != v49 )
        {
          if ( !v50 )
          {
            v48 = v125;
            goto LABEL_132;
          }
          v51 = v132;
          for ( j = (__m128i *)v116[34]; ; j = v53 )
          {
            v54 = j[2].m128i_u64[0];
            v55 = v54 < v132.m128i_i64[0];
            if ( v54 == v132.m128i_i64[0] )
              v55 = j[2].m128i_i64[1] < (unsigned __int64)v132.m128i_i64[1];
            v53 = (__m128i *)j[1].m128i_i64[1];
            if ( !v55 )
            {
              v53 = (__m128i *)j[1].m128i_i64[0];
              v48 = j;
            }
            if ( !v53 )
              break;
          }
          if ( v125 == v48 )
          {
LABEL_132:
            v140 = &v132;
            v79 = sub_2CE4CC0(v115, v48, (const __m128i **)&v140) + 56;
            v50 = (__m128i *)v116[34];
            if ( v50 )
            {
              v51 = v132;
              goto LABEL_124;
            }
            v80 = v125;
          }
          else
          {
            if ( v48[2].m128i_i64[0] != v132.m128i_i64[0] )
            {
              if ( v48[2].m128i_i64[0] <= (unsigned __int64)v132.m128i_i64[0] )
                goto LABEL_123;
              goto LABEL_132;
            }
            if ( v48[2].m128i_i64[1] > (unsigned __int64)v132.m128i_i64[1] )
              goto LABEL_132;
LABEL_123:
            v79 = (__int64)&v48[3].m128i_i64[1];
LABEL_124:
            v80 = v125;
            while ( 1 )
            {
              v82 = v50[2].m128i_u64[0];
              v83 = v82 < v51.m128i_i64[0];
              if ( v82 == v51.m128i_i64[0] )
                v83 = v50[2].m128i_i64[1] < (unsigned __int64)v51.m128i_i64[1];
              v81 = (__m128i *)v50[1].m128i_i64[1];
              if ( !v83 )
              {
                v81 = (__m128i *)v50[1].m128i_i64[0];
                v80 = v50;
              }
              if ( !v81 )
                break;
              v50 = v81;
            }
            if ( v125 != v80 )
            {
              v84 = v80[2].m128i_u64[0];
              v85 = v84 > v51.m128i_i64[0];
              if ( v84 == v51.m128i_i64[0] )
                v85 = v80[2].m128i_i64[1] > (unsigned __int64)v51.m128i_i64[1];
              if ( !v85 )
              {
LABEL_138:
                v86 = v137;
                v87 = v80 + 3;
                v114 = &v80[3].m128i_i64[1];
                if ( v137 != &v135 )
                {
                  v112 = v11;
                  v88 = v80;
                  v111 = v10;
                  v89 = v79;
                  do
                  {
                    v91 = sub_23FE670(v87, v89, (unsigned __int64 *)v86 + 4);
                    v90 = (__int64)v91;
                    if ( v92 )
                    {
                      v93 = 1;
                      if ( !v91 && v92 != v114 )
                        v93 = v86[4] < (unsigned __int64)v92[4];
                      v124 = v92;
                      v90 = sub_22077B0(0x28u);
                      *(_QWORD *)(v90 + 32) = v86[4];
                      sub_220F040(v93, v90, v124, v114);
                      ++v88[5].m128i_i64[1];
                    }
                    v89 = sub_220EF30(v90);
                    v86 = (__int64 *)sub_220EF30((__int64)v86);
                  }
                  while ( v86 != &v135 );
                  v11 = v112;
                  v10 = v111;
                }
                goto LABEL_85;
              }
            }
          }
          v140 = &v132;
          v80 = (__m128i *)sub_2CE4CC0(v115, v80, (const __m128i **)&v140);
          goto LABEL_138;
        }
        if ( !v50 )
        {
          v94 = v125;
          goto LABEL_162;
        }
        v94 = v125;
        while ( 1 )
        {
          v96 = v50[2].m128i_u64[0];
          v97 = v96 < v132.m128i_i64[0];
          if ( v96 == v132.m128i_i64[0] )
            v97 = v50[2].m128i_i64[1] < (unsigned __int64)v132.m128i_i64[1];
          v95 = (__m128i *)v50[1].m128i_i64[1];
          if ( !v97 )
          {
            v95 = (__m128i *)v50[1].m128i_i64[0];
            v94 = v50;
          }
          if ( !v95 )
            break;
          v50 = v95;
        }
        if ( v125 == v94 )
          goto LABEL_162;
        v98 = v94[2].m128i_u64[0];
        v99 = v98 > v132.m128i_i64[0];
        if ( v98 == v132.m128i_i64[0] )
          v99 = v94[2].m128i_i64[1] > (unsigned __int64)v132.m128i_i64[1];
        if ( v99 )
        {
LABEL_162:
          v140 = &v132;
          v94 = (__m128i *)sub_2CE4CC0(v115, v94, (const __m128i **)&v140);
        }
        if ( &v94[3] == (__m128i *)&v134 )
          goto LABEL_85;
        v100 = (unsigned __int64 *)v94[4].m128i_i64[0];
        v101 = (__int64)&v94[3].m128i_i64[1];
        v140 = (__m128i *)v100;
        v102 = v94[5].m128i_i64[0];
        m128i_i32 = v94[3].m128i_i32;
        v141 = v102;
        if ( v100 )
        {
          v100[1] = 0;
          if ( *(_QWORD *)(v102 + 16) )
            v141 = *(_QWORD *)(v102 + 16);
          v94[4].m128i_i64[0] = 0;
          v94[4].m128i_i64[1] = v101;
          v94[5].m128i_i64[0] = v101;
          v94[5].m128i_i64[1] = 0;
          v103 = v136;
          if ( v136 )
            goto LABEL_168;
          do
          {
LABEL_173:
            sub_2CDF380(v100[3]);
            v110 = (unsigned __int64)v100;
            v100 = (unsigned __int64 *)v100[2];
            j_j___libc_free_0(v110);
          }
          while ( v100 );
          goto LABEL_85;
        }
        v141 = 0;
        v94[4].m128i_i64[0] = 0;
        v94[4].m128i_i64[1] = v101;
        v94[5].m128i_i64[0] = v101;
        v94[5].m128i_i64[1] = 0;
        v103 = v136;
        if ( v136 )
        {
LABEL_168:
          v104 = sub_2CDDCD0(v103, v101, &v140);
          v105 = v104;
          do
          {
            v106 = v104;
            v104 = *(_QWORD *)(v104 + 16);
          }
          while ( v104 );
          v94[4].m128i_i64[1] = v106;
          v107 = v105;
          do
          {
            v108 = v107;
            v107 = *(_QWORD *)(v107 + 24);
          }
          while ( v107 );
          v94[5].m128i_i64[0] = v108;
          v109 = v139;
          v94[4].m128i_i64[0] = v105;
          v94[5].m128i_i64[1] = v109;
          v100 = (unsigned __int64 *)v140;
          if ( !v140 )
            goto LABEL_85;
          goto LABEL_173;
        }
      }
      else
      {
LABEL_85:
        v59 = (unsigned __int64)v136;
        if ( v136 )
        {
          do
          {
            sub_2CDF380(*(_QWORD *)(v59 + 24));
            v60 = v59;
            v59 = *(_QWORD *)(v59 + 16);
            j_j___libc_free_0(v60);
          }
          while ( v59 );
        }
      }
LABEL_87:
      v119 = sub_220EEE0(v119);
      if ( (unsigned __int64 *)v119 == v118 )
      {
        v2 = v116;
        break;
      }
    }
  }
  v136 = 0;
  v139 = 0;
  v61 = (const __m128i *)v2[35];
  v137 = &v135;
  v138 = &v135;
  LODWORD(v135) = 0;
  if ( v61 == v125 )
  {
    LODWORD(v141) = 0;
    v78 = 0;
    m128i_i32 = 0;
    v143 = &v141;
    v144 = &v141;
    v145 = 0;
  }
  else
  {
    do
    {
      v62 = 0;
      v132 = _mm_loadu_si128(v61 + 2);
      for ( k = v61[4].m128i_i64[1]; &v61[3].m128i_u64[1] != (unsigned __int64 *)k; k = sub_220EF30(k) )
      {
        v64 = sub_9208B0(v2[44], *(_QWORD *)(*(_QWORD *)(k + 32) + 8LL));
        v141 = v65;
        v140 = (__m128i *)((unsigned __int64)(v64 + 7) >> 3);
        v66 = sub_CA1930(&v140);
        if ( v62 < v66 )
          v62 = v66;
      }
      for ( m = (const __m128i *)v2[35]; m != v125; m = (const __m128i *)sub_220EEE0((__int64)m) )
      {
        if ( m != v61 && m[2].m128i_i64[0] == v61[2].m128i_i64[0] )
        {
          v68 = m[2].m128i_u64[1];
          if ( v132.m128i_i64[1] <= v68 && v62 + v132.m128i_i64[1] > v68 )
          {
            sub_2CE0EA0(&v134, &v132);
            sub_2CE0EA0(&v134, m + 2);
          }
        }
      }
      v61 = (const __m128i *)sub_220EEE0((__int64)v61);
    }
    while ( v61 != v125 );
    for ( n = v137; n != &v135; n = (__int64 *)sub_220EF30((__int64)n) )
    {
      v69 = (__m128i *)sub_2CE2F20((__int64)v115, (unsigned __int64 *)n + 4);
      v123 = v70;
      v71 = (__int64)v69;
      if ( (__m128i *)v2[35] == v69 && v125 == v70 )
      {
        sub_2CDF550((_QWORD *)v2[34]);
        v2[35] = v125;
        v2[34] = 0;
        v2[36] = v125;
        v2[37] = 0;
      }
      else if ( v70 != v69 )
      {
        do
        {
          v72 = (int *)v71;
          v71 = sub_220EF30(v71);
          v73 = sub_220F330(v72, v125);
          v74 = *((_QWORD *)v73 + 8);
          v75 = (unsigned __int64)v73;
          while ( v74 )
          {
            sub_2CDF380(*(_QWORD *)(v74 + 24));
            v76 = v74;
            v74 = *(_QWORD *)(v74 + 16);
            j_j___libc_free_0(v76);
          }
          j_j___libc_free_0(v75);
          --v2[37];
        }
        while ( v123 != (__m128i *)v71 );
      }
    }
    v77 = (__m128i *)v2[35];
    LODWORD(v141) = 0;
    m128i_i32 = 0;
    v143 = &v141;
    v144 = &v141;
    v145 = 0;
    if ( v125 == v77 )
    {
      v78 = 0;
    }
    else
    {
      do
      {
        if ( v77[5].m128i_i64[1] > 1uLL )
          sub_2CE40A0(v2, (unsigned __int64 *)&v77[3], v77[2].m128i_i64[0], v77[2].m128i_i64[1], &v140);
        v77 = (__m128i *)sub_220EEE0((__int64)v77);
      }
      while ( v77 != v125 );
      v78 = (unsigned __int64)m128i_i32;
    }
  }
  sub_2CDE0D0(v78);
  sub_2CDDF00((unsigned __int64)v136);
  if ( v129 )
    j_j___libc_free_0(v129);
}
