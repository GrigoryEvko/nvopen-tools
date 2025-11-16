// Function: sub_115D7A0
// Address: 0x115d7a0
//
unsigned __int8 *__fastcall sub_115D7A0(__m128i *a1, __int64 a2)
{
  __m128i v4; // xmm1
  unsigned __int64 v5; // xmm2_8
  __m128i v6; // xmm3
  __int64 v7; // rax
  unsigned __int8 *v8; // r13
  __int64 v9; // r14
  char v10; // al
  __int64 v11; // rdx
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 *v17; // r9
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int **v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // edx
  unsigned int v28; // ecx
  unsigned __int64 v29; // rdx
  int v30; // edx
  int v31; // eax
  int v32; // eax
  _BYTE *v33; // rax
  _BYTE *v34; // rdi
  unsigned __int8 v35; // al
  __int64 v36; // rax
  __int64 v37; // r14
  _BYTE *v38; // r13
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  _BYTE *v43; // rdi
  unsigned __int8 *v44; // rax
  __int64 v45; // rsi
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r10
  unsigned int **v49; // r13
  char v50; // r9
  _BYTE *v51; // rax
  __int64 v52; // rax
  unsigned __int8 *v53; // rdx
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 *v56; // r13
  __int64 v57; // rax
  bool v58; // zf
  __int64 v59; // rax
  unsigned __int8 *v60; // r8
  __int64 v61; // rax
  _BYTE *v62; // r8
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rsi
  unsigned int **v66; // r14
  int v67; // edx
  int v68; // eax
  __int64 *v69; // rax
  __int64 v70; // r13
  _BYTE *v71; // rbx
  __int64 v72; // rax
  __int64 v73; // r13
  __int64 *v74; // rax
  __int64 v75; // rax
  unsigned __int8 *v76; // r12
  unsigned int *v77; // r13
  __int64 v78; // r14
  __int64 v79; // rdx
  __int64 v80; // rdx
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 *v83; // r14
  __int64 v84; // r13
  bool v85; // al
  unsigned __int8 *v86; // rax
  __int64 v87; // rax
  __int64 *v88; // rdi
  _BYTE *v89; // rsi
  __int64 v90; // r14
  _QWORD *v91; // rax
  __int64 v92; // r13
  __int64 v93; // rax
  _BYTE *v94; // rsi
  __int64 *v95; // rdi
  _QWORD *v96; // rax
  __int64 v97; // r13
  __int64 v98; // rax
  __int64 v99; // rax
  char v100; // al
  __m128i *v101; // rsi
  __m128i *v102; // rdi
  __int64 i; // rcx
  __m128i *v104; // rsi
  __int64 v105; // rcx
  __m128i *v106; // rdi
  char v107; // al
  __int64 v108; // rcx
  __m128i *v109; // rdi
  __m128i *v110; // rsi
  __int64 v111; // rcx
  unsigned __int8 *v112; // r13
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned int **v115; // rdi
  __int64 v116; // rax
  __int64 *v117; // rdi
  __int64 v118; // rax
  __int64 *v119; // r15
  __m128i v120; // rax
  int v121; // r15d
  __int64 *v122; // rdi
  __int64 v123; // rax
  __int64 v124; // rsi
  unsigned __int8 *v125; // rax
  unsigned __int8 v126; // al
  unsigned int v127; // eax
  __int64 *v128; // r13
  __m128i v129; // rax
  unsigned __int8 *v130; // rax
  __int64 v131; // rbx
  __int64 v132; // r14
  __int64 *v133; // rax
  __int64 (__fastcall *v134)(__int64 **, __int64); // rdx
  __int64 v135; // r13
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  unsigned int **v139; // rdi
  __int64 v140; // rax
  char v141; // al
  _QWORD *v142; // rax
  __int64 v143; // r15
  __int64 v144; // r14
  __int64 v145; // rdx
  unsigned int v146; // esi
  __int64 v147; // r14
  __int64 v148; // rax
  unsigned int **v149; // rdi
  __int64 v150; // r13
  __int64 v151; // rax
  char v152; // al
  unsigned int **v153; // rdi
  __int64 v154; // rax
  bool v155; // r10
  __int64 v156; // r14
  unsigned int **v157; // rdi
  unsigned __int8 *v158; // r13
  __int64 v159; // [rsp-8h] [rbp-138h]
  __int64 v160; // [rsp-8h] [rbp-138h]
  unsigned __int8 *v161; // [rsp+8h] [rbp-128h]
  __int64 v162; // [rsp+8h] [rbp-128h]
  __int64 v163; // [rsp+10h] [rbp-120h]
  bool v164; // [rsp+18h] [rbp-118h]
  __int64 v165; // [rsp+18h] [rbp-118h]
  __int64 v166; // [rsp+18h] [rbp-118h]
  unsigned int v167; // [rsp+18h] [rbp-118h]
  unsigned int v168; // [rsp+20h] [rbp-110h]
  char v169; // [rsp+25h] [rbp-10Bh]
  char v170; // [rsp+26h] [rbp-10Ah]
  unsigned __int8 v171; // [rsp+27h] [rbp-109h]
  _QWORD **v172; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v173; // [rsp+30h] [rbp-100h]
  char v174; // [rsp+30h] [rbp-100h]
  unsigned int v175; // [rsp+30h] [rbp-100h]
  bool v176; // [rsp+38h] [rbp-F8h]
  __int64 v177; // [rsp+38h] [rbp-F8h]
  __int64 v178; // [rsp+38h] [rbp-F8h]
  unsigned __int8 *v179; // [rsp+38h] [rbp-F8h]
  _BYTE *v180; // [rsp+38h] [rbp-F8h]
  __int64 v181; // [rsp+38h] [rbp-F8h]
  unsigned int v182; // [rsp+38h] [rbp-F8h]
  bool v183; // [rsp+38h] [rbp-F8h]
  bool v184; // [rsp+38h] [rbp-F8h]
  _BYTE *v185; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE *v186; // [rsp+48h] [rbp-E8h] BYREF
  _BYTE *v187; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v188; // [rsp+58h] [rbp-D8h] BYREF
  __int64 *v189; // [rsp+60h] [rbp-D0h] BYREF
  __int64 *v190; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v191; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v192; // [rsp+78h] [rbp-B8h]
  __int64 *v193; // [rsp+80h] [rbp-B0h] BYREF
  __int64 (__fastcall *v194)(__int64 **, __int64); // [rsp+88h] [rbp-A8h]
  __int64 **v195; // [rsp+90h] [rbp-A0h]
  __int64 v196; // [rsp+98h] [rbp-98h]
  __int64 v197; // [rsp+A0h] [rbp-90h]
  __m128i v198; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v199; // [rsp+C0h] [rbp-70h]
  __int64 **v200; // [rsp+D0h] [rbp-60h]
  __int64 v201; // [rsp+D8h] [rbp-58h]
  __m128i v202; // [rsp+E0h] [rbp-50h]
  __int64 v203; // [rsp+F0h] [rbp-40h]

  v4 = _mm_loadu_si128(a1 + 7);
  v173 = (unsigned __int8 *)a2;
  v5 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v6 = _mm_loadu_si128(a1 + 9);
  v7 = a1[10].m128i_i64[0];
  v8 = *(unsigned __int8 **)(a2 - 64);
  v198 = _mm_loadu_si128(a1 + 6);
  v9 = *(_QWORD *)(a2 - 32);
  v200 = (__int64 **)v5;
  v199 = v4;
  v201 = a2;
  v202 = v6;
  v203 = v7;
  v176 = sub_B448F0(a2);
  v10 = sub_B44900(a2);
  v11 = (__int64)sub_101E7A0(v8, (unsigned __int8 *)v9, v10, v176, &v198);
  if ( v11 )
    return sub_F162A0((__int64)a1, a2, v11);
  if ( !(unsigned __int8)sub_F29CA0(a1, (unsigned __int8 *)a2) )
  {
    v173 = sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
    if ( !v173 )
    {
      v173 = (unsigned __int8 *)sub_F11DB0(a1->m128i_i64, (unsigned __int8 *)a2);
      if ( !v173 )
      {
        v11 = (__int64)sub_F0DE90(a1, (unsigned __int8 *)a2);
        if ( v11 )
          return sub_F162A0((__int64)a1, a2, v11);
        v172 = *(_QWORD ***)(a2 + 8);
        v168 = sub_BCB060((__int64)v172);
        v171 = sub_B44900(a2);
        v13 = sub_B448F0(a2);
        v198.m128i_i64[0] = 0;
        v170 = v13;
        if ( (unsigned __int8)sub_995B10(&v198, v9) )
        {
          v18 = (__int64)v8;
          LOWORD(v200) = 257;
          if ( v171 )
            return sub_B505E0((__int64)v8, (__int64)&v198, 0, 0);
          return (unsigned __int8 *)sub_B50550(v18, (__int64)&v198, 0, 0);
        }
        if ( *(_BYTE *)a2 == 46 )
        {
          v33 = *(_BYTE **)(a2 - 64);
          if ( *v33 == 54 )
          {
            v15 = *((_QWORD *)v33 - 8);
            v177 = v15;
            if ( v15 )
            {
              v34 = (_BYTE *)*((_QWORD *)v33 - 4);
              if ( *v34 <= 0x15u && *v34 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v34) )
              {
                v35 = **(_BYTE **)(a2 - 32);
                if ( v35 <= 0x15u && v35 != 5 )
                {
                  v165 = *(_QWORD *)(a2 - 32);
                  if ( !(unsigned __int8)sub_AD6CA0(v165) )
                  {
                    v198.m128i_i8[8] = 0;
                    v198.m128i_i64[0] = (__int64)&v191;
                    if ( (unsigned __int8)sub_1157D90((__int64)&v198, v165, v14) )
                    {
                      v36 = sub_96E6C0(0x19u, v165, v34, a1[5].m128i_i64[1]);
                      v37 = *(_QWORD *)(a2 - 64);
                      v38 = (_BYTE *)v36;
                      v39 = v177;
                      LOWORD(v200) = 257;
                      v173 = (unsigned __int8 *)sub_B504D0(17, v177, v36, (__int64)&v198, 0, 0);
                      if ( v170 && sub_B448F0(v37) )
                      {
                        v39 = 1;
                        sub_B447F0(v173, 1);
                      }
                      if ( !v171 || !sub_B44900(v37) || !(unsigned __int8)sub_AD7CA0(v38, v39, v40, v41, v42) )
                        return v173;
                      goto LABEL_63;
                    }
                  }
                }
              }
            }
          }
        }
        if ( *(_BYTE *)a2 == 46 )
        {
          v178 = *(_QWORD *)(a2 - 64);
          if ( v178 )
          {
            v43 = *(_BYTE **)(a2 - 32);
            if ( *v43 <= 0x15u )
            {
              v44 = sub_AD8AC0((__int64)v43);
              v45 = v178;
              if ( v44 )
              {
                LOWORD(v200) = 257;
                v179 = v44;
                v46 = (unsigned __int8 *)sub_B504D0(25, v45, (__int64)v44, (__int64)&v198, 0, 0);
                v48 = (__int64)v179;
                v173 = v46;
                if ( v170 )
                {
                  sub_B447F0(v46, 1);
                  v48 = (__int64)v179;
                }
                if ( !v171 )
                  return v173;
                v198.m128i_i8[8] = 0;
                v198.m128i_i64[0] = (__int64)&v193;
                if ( !(unsigned __int8)sub_1157D90((__int64)&v198, v48, v47)
                  || sub_D94970((__int64)v193, (_QWORD *)(unsigned int)(*((_DWORD *)v193 + 2) - 1)) )
                {
                  return v173;
                }
LABEL_63:
                sub_B44850(v173, 1);
                return v173;
              }
            }
          }
        }
        if ( v168 <= 2
          || (v199.m128i_i8[0] = 0,
              LOBYTE(v200) = 0,
              v198.m128i_i64[0] = (__int64)&v188,
              v198.m128i_i64[1] = (__int64)&v189,
              v199.m128i_i64[1] = (__int64)&v190,
              !sub_115D560((__int64)&v198, 17, (unsigned __int8 *)a2)) )
        {
LABEL_23:
          v25 = *((_QWORD *)v8 + 2);
          if ( v25 )
          {
            if ( !*(_QWORD *)(v25 + 8) )
            {
              v198.m128i_i64[0] = 0;
              if ( (unsigned __int8)sub_1157E10((__int64 **)&v198, v9) )
              {
                v180 = (_BYTE *)sub_116CB70(1, v171, v8, a1);
                if ( v180 )
                {
                  v49 = (unsigned int **)a1[2].m128i_i64[0];
                  v50 = v171;
                  if ( v171 )
                    v50 = sub_AD7CA0((_BYTE *)v9, v171, v14, v15, v16);
                  v174 = v50;
                  LOWORD(v200) = 257;
                  v51 = (_BYTE *)sub_AD6890(v9, 0);
                  v52 = sub_A81850(v49, v180, v51, (__int64)&v198, 0, v174);
                  goto LABEL_79;
                }
                v126 = *v8;
                if ( *v8 > 0x1Cu && (v126 == 68 || v126 == 69) )
                {
                  v16 = *((_QWORD *)v8 - 4);
                  if ( v16 )
                  {
                    v181 = *((_QWORD *)v8 - 4);
                    v198.m128i_i64[0] = (__int64)&v191;
                    v198.m128i_i8[8] = 1;
                    if ( (unsigned __int8)sub_991580((__int64)&v198, v9) )
                    {
                      v166 = v181;
                      v175 = sub_BCB060(*(_QWORD *)(v181 + 8));
                      v127 = sub_D949C0(v191);
                      v14 = v175;
                      v16 = v181;
                      v15 = v127;
                      v182 = v127;
                      if ( v168 - v175 <= v127 )
                      {
                        v128 = (__int64 *)a1[2].m128i_i64[0];
                        v129.m128i_i64[0] = (__int64)sub_BD5D20(v166);
                        LOWORD(v200) = 773;
                        v198 = v129;
                        v199.m128i_i64[0] = (__int64)".neg";
                        v130 = sub_10A0530(v128, v166, (__int64)&v198, 0);
                        v131 = a1[2].m128i_i64[0];
                        v132 = (__int64)v130;
                        v133 = (__int64 *)sub_BD5D20((__int64)v130);
                        LOWORD(v197) = 773;
                        v193 = v133;
                        v194 = v134;
                        v195 = (__int64 **)".z";
                        if ( v172 == *(_QWORD ***)(v132 + 8) )
                        {
                          v135 = v132;
                        }
                        else
                        {
                          v135 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v131 + 80) + 120LL))(
                                   *(_QWORD *)(v131 + 80),
                                   39,
                                   v132);
                          if ( !v135 )
                          {
                            LOWORD(v200) = 257;
                            v142 = sub_BD2C40(72, unk_3F10A14);
                            v135 = (__int64)v142;
                            if ( v142 )
                              sub_B515B0((__int64)v142, v132, (__int64)v172, (__int64)&v198, 0, 0);
                            (*(void (__fastcall **)(_QWORD, __int64, __int64 **, _QWORD, _QWORD))(**(_QWORD **)(v131 + 88)
                                                                                                + 16LL))(
                              *(_QWORD *)(v131 + 88),
                              v135,
                              &v193,
                              *(_QWORD *)(v131 + 56),
                              *(_QWORD *)(v131 + 64));
                            v143 = *(_QWORD *)v131;
                            v144 = *(_QWORD *)v131 + 16LL * *(unsigned int *)(v131 + 8);
                            while ( v143 != v144 )
                            {
                              v145 = *(_QWORD *)(v143 + 8);
                              v146 = *(_DWORD *)v143;
                              v143 += 16;
                              sub_B99FD0(v135, v146, v145);
                            }
                          }
                        }
                        LOWORD(v200) = 257;
                        v136 = sub_AD64C0((__int64)v172, v182, 0);
                        return (unsigned __int8 *)sub_B504D0(25, v135, v136, (__int64)&v198, 0, 0);
                      }
                    }
                  }
                }
              }
            }
          }
          v173 = sub_F28360((__int64)a1, (_BYTE *)a2, v14, v15, v16, v17);
          if ( v173 )
            return v173;
          v11 = sub_115CCA0((unsigned __int8 *)a2, a1[2].m128i_i64[0]);
          if ( !v11 )
          {
            v198.m128i_i64[0] = (__int64)&v185;
            if ( (unsigned __int8)sub_F11D70(&v198, (_BYTE *)v9) )
            {
              v198.m128i_i64[0] = (__int64)&v191;
              v198.m128i_i64[1] = (__int64)&v193;
              v199.m128i_i64[1] = (__int64)&v191;
              v200 = &v193;
              v152 = sub_11580F0(&v198, (char *)v8);
              if ( v152 )
              {
                v153 = (unsigned int **)a1[2].m128i_i64[0];
                v183 = v152;
                LOWORD(v200) = 257;
                v154 = sub_A81850(v153, v193, v185, (__int64)&v198, 0, 0);
                v155 = v183;
                v156 = v154;
                if ( *v8 != 58 )
                  v155 = sub_B448F0((__int64)v8);
                v157 = (unsigned int **)a1[2].m128i_i64[0];
                v184 = v155;
                LOWORD(v200) = 257;
                v158 = (unsigned __int8 *)sub_A81850(v157, (_BYTE *)v191, v185, (__int64)&v198, 0, 0);
                LOWORD(v200) = 257;
                v173 = (unsigned __int8 *)sub_B504D0(13, (__int64)v158, v156, (__int64)&v198, 0, 0);
                if ( v184 && v170 )
                {
                  if ( (unsigned __int8)(*v158 - 42) <= 0x11u )
                    sub_B447F0(v158, 1);
                  sub_B447F0(v173, 1);
                }
                return v173;
              }
            }
            if ( v8 == (unsigned __int8 *)v9 )
            {
              v198.m128i_i32[0] = 1;
              v198.m128i_i32[2] = 0;
              v199.m128i_i64[0] = (__int64)&v186;
              if ( (unsigned __int8)sub_10E25C0((__int64)&v198, (__int64)v8) )
              {
                LOWORD(v200) = 257;
                return (unsigned __int8 *)sub_B504D0(17, (__int64)v186, (__int64)v186, (__int64)&v198, 0, 0);
              }
            }
            if ( sub_B44900(a2) )
            {
              LODWORD(v193) = 1;
              LODWORD(v194) = 0;
              v195 = (__int64 **)&v186;
              LODWORD(v196) = 1;
              v197 = 0;
              if ( (unsigned __int8)sub_11581D0((__int64)&v193, (__int64)v8) )
              {
                v198.m128i_i32[0] = 1;
                v198.m128i_i32[2] = 0;
                v199.m128i_i64[0] = (__int64)&v190;
                v199.m128i_i32[2] = 1;
                v200 = 0;
                if ( (unsigned __int8)sub_11581D0((__int64)&v198, v9) )
                {
                  v147 = a1[2].m128i_i64[0];
                  LOWORD(v200) = 257;
                  HIDWORD(v191) = 0;
                  v148 = sub_ACD6D0(*(__int64 **)(v147 + 72));
                  v149 = (unsigned int **)a1[2].m128i_i64[0];
                  v150 = v148;
                  LOWORD(v197) = 257;
                  v151 = sub_A81850(v149, v186, v190, (__int64)&v193, 0, 1);
                  v52 = sub_B33C40(v147, 1u, v151, v150, v191, (__int64)&v198);
LABEL_79:
                  v53 = (unsigned __int8 *)v52;
                  return sub_F162A0((__int64)a1, a2, (__int64)v53);
                }
              }
            }
            v198.m128i_i64[0] = 0;
            v198.m128i_i64[1] = (__int64)&v186;
            if ( (unsigned __int8)sub_10A7530((__int64 **)&v198, 15, v8) && *(_BYTE *)v9 <= 0x15u )
            {
              LOWORD(v200) = 257;
              v26 = sub_AD6890(v9, 0);
              return (unsigned __int8 *)sub_B504D0(17, (__int64)v186, v26, (__int64)&v198, 0, 0);
            }
            v193 = 0;
            v194 = (__int64 (__fastcall *)(__int64 **, __int64))&v186;
            if ( (unsigned __int8)sub_10A7530(&v193, 15, v8) )
            {
              v198.m128i_i64[0] = 0;
              v198.m128i_i64[1] = (__int64)&v187;
              if ( (unsigned __int8)sub_10A7530((__int64 **)&v198, 15, (unsigned __int8 *)v9) )
              {
                LOWORD(v200) = 257;
                v125 = (unsigned __int8 *)sub_B504D0(17, (__int64)v186, (__int64)v187, (__int64)&v198, 0, 0);
                v173 = v125;
                if ( v171 && (v8[1] & 4) != 0 && (*(_BYTE *)(v9 + 1) & 4) != 0 )
                  sub_B44850(v125, 1);
                return v173;
              }
            }
            v58 = *(_BYTE *)a2 == 46;
            v198.m128i_i64[0] = 0;
            v198.m128i_i64[1] = (__int64)&v186;
            v199.m128i_i64[0] = (__int64)&v187;
            if ( !v58 )
            {
LABEL_94:
              v198.m128i_i64[0] = 0;
              v198.m128i_i64[1] = (__int64)&v186;
              if ( (unsigned __int8)sub_10A7530((__int64 **)&v198, 15, (unsigned __int8 *)v9) )
              {
                v124 = sub_116CB70(0, 0, v8, a1);
                if ( v124 )
                {
                  LOWORD(v200) = 257;
                  return (unsigned __int8 *)sub_B504D0(17, v124, (__int64)v186, (__int64)&v198, 0, 0);
                }
              }
              v59 = *((_QWORD *)v8 + 2);
              if ( v59 )
              {
                if ( !*(_QWORD *)(v59 + 8) )
                {
                  LOBYTE(v192) = 0;
                  v189 = &v188;
                  v190 = &v188;
                  v191 = (__int64)&v188;
                  if ( (unsigned __int8)sub_991580((__int64)&v191, v9) )
                  {
                    v196 = 0;
                    v193 = (__int64 *)&v186;
                    v194 = sub_1155230;
                    v195 = &v189;
                    if ( (unsigned __int8)sub_11583C0((__int64)&v193, v8)
                      || (v198.m128i_i64[0] = (__int64)&v186,
                          v198.m128i_i64[1] = (__int64)sub_11552A0,
                          v199 = (__m128i)(unsigned __int64)&v190,
                          (unsigned __int8)sub_1158510((__int64)&v198, v8)) )
                    {
                      v121 = *v8;
                      v122 = (__int64 *)a1[2].m128i_i64[0];
                      LOWORD(v200) = 257;
                      LOWORD(v197) = 257;
                      v121 -= 29;
                      v123 = sub_1155FA0(v122, v121, *((_QWORD *)v8 - 4), v9, v191, 0, (__int64)&v193, 0);
                      v173 = (unsigned __int8 *)sub_B504D0(v121, (__int64)v186, v123, (__int64)&v198, 0, 0);
                      sub_B448B0((__int64)v173, 1);
                      return v173;
                    }
                  }
                }
              }
              if ( (unsigned __int8)(*v8 - 48) > 1u )
              {
                if ( (unsigned __int8)(*(_BYTE *)v9 - 42) > 0x11u )
                {
                  sub_F0E5E0((__int64)a1, (__int64)v8);
                  goto LABEL_99;
                }
                v163 = (__int64)v8;
                v60 = (unsigned __int8 *)v9;
              }
              else
              {
                v163 = v9;
                v60 = v8;
              }
              v161 = v60;
              v61 = sub_F0E5E0((__int64)a1, v163);
              v62 = v161;
              v63 = *((_QWORD *)v161 + 2);
              if ( v63 )
              {
                if ( !*(_QWORD *)(v63 + 8) )
                {
                  v111 = *((_QWORD *)v161 - 4);
                  v162 = v111;
                  if ( v111 == v163 || v111 == v61 )
                  {
                    v169 = *v62;
                    if ( (unsigned __int8)(*v62 - 48) <= 1u )
                    {
                      v112 = (unsigned __int8 *)*((_QWORD *)v62 - 8);
                      if ( sub_B44E60((__int64)v62) )
                      {
                        if ( v162 == v163 )
                        {
                          return sub_F162A0((__int64)a1, a2, (__int64)v112);
                        }
                        else
                        {
                          LOWORD(v200) = 257;
                          return (unsigned __int8 *)sub_B50550((__int64)v112, (__int64)&v198, 0, 0);
                        }
                      }
                      else
                      {
                        if ( !sub_98EF80(v112, 0, 0, 0, 0) )
                        {
                          v119 = (__int64 *)a1[2].m128i_i64[0];
                          v120.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v112);
                          LOWORD(v200) = 773;
                          v198 = v120;
                          v199.m128i_i64[0] = (__int64)".fr";
                          v112 = (unsigned __int8 *)sub_1156690(v119, (__int64)v112, (__int64)&v198);
                        }
                        v117 = (__int64 *)a1[2].m128i_i64[0];
                        LOWORD(v200) = 257;
                        v118 = sub_1155FA0(
                                 v117,
                                 (unsigned int)(v169 != 48) + 22,
                                 (__int64)v112,
                                 v162,
                                 (int)v193,
                                 0,
                                 (__int64)&v198,
                                 0);
                        LOWORD(v200) = 257;
                        if ( v162 == v163 )
                          return (unsigned __int8 *)sub_B504D0(15, (__int64)v112, v118, (__int64)&v198, 0, 0);
                        else
                          return (unsigned __int8 *)sub_B504D0(15, v118, (__int64)v112, (__int64)&v198, 0, 0);
                      }
                    }
                  }
                }
              }
LABEL_99:
              if ( sub_1001970((__int64)v172, 1)
                || (v194 = 0, *v8 == 57)
                && (unsigned __int8)sub_1155650((__int64)&v193, (__int64)v8)
                && (v198.m128i_i64[1] = 0, *(_BYTE *)v9 == 57)
                && (unsigned __int8)sub_1155650((__int64)&v198, v9) )
              {
                LOWORD(v200) = 257;
                return (unsigned __int8 *)sub_B504D0(28, (__int64)v8, v9, (__int64)&v198, 0, 0);
              }
              v53 = sub_11572C0(a2, 0, (__int64 *)a1[2].m128i_i64[0]);
              if ( v53 )
                return sub_F162A0((__int64)a1, a2, (__int64)v53);
              v53 = sub_11572C0(a2, 1, (__int64 *)a1[2].m128i_i64[0]);
              if ( v53 )
                return sub_F162A0((__int64)a1, a2, (__int64)v53);
              v190 = (__int64 *)&v186;
              if ( (unsigned __int8)sub_10A3FA0(&v190, v8)
                && (v191 = (__int64)&v187, (unsigned __int8)sub_10A3FA0((_QWORD **)&v191, (_BYTE *)v9))
                || (v193 = (__int64 *)&v186, (unsigned __int8)sub_10E4070(&v193, v8))
                && (v198.m128i_i64[0] = (__int64)&v187, (unsigned __int8)sub_10E4070(&v198, (_BYTE *)v9)) )
              {
                if ( sub_1001970(*((_QWORD *)v186 + 1), 1)
                  && *((_QWORD *)v187 + 1) == *((_QWORD *)v186 + 1)
                  && ((v137 = *((_QWORD *)v8 + 2)) != 0 && !*(_QWORD *)(v137 + 8)
                   || (v138 = *(_QWORD *)(v9 + 16)) != 0 && !*(_QWORD *)(v138 + 8)
                   || v186 == v187) )
                {
                  v139 = (unsigned int **)a1[2].m128i_i64[0];
                  v198.m128i_i64[0] = (__int64)"mulbool";
                  LOWORD(v200) = 259;
                  v140 = sub_A82350(v139, v186, v187, (__int64)&v198);
                  LOWORD(v200) = 257;
                  return (unsigned __int8 *)sub_B51D30(39, v140, (__int64)v172, (__int64)&v198, 0, 0);
                }
              }
              v190 = (__int64 *)&v186;
              if ( (unsigned __int8)sub_10E4070(&v190, v8)
                && (v191 = (__int64)&v187, (unsigned __int8)sub_10A3FA0((_QWORD **)&v191, (_BYTE *)v9))
                || (v193 = (__int64 *)&v186, (unsigned __int8)sub_10A3FA0(&v193, v8))
                && (v198.m128i_i64[0] = (__int64)&v187, (unsigned __int8)sub_10E4070(&v198, (_BYTE *)v9)) )
              {
                if ( sub_1001970(*((_QWORD *)v186 + 1), 1)
                  && *((_QWORD *)v187 + 1) == *((_QWORD *)v186 + 1)
                  && ((v113 = *((_QWORD *)v8 + 2)) != 0 && !*(_QWORD *)(v113 + 8)
                   || (v114 = *(_QWORD *)(v9 + 16)) != 0 && !*(_QWORD *)(v114 + 8)) )
                {
                  v115 = (unsigned int **)a1[2].m128i_i64[0];
                  v198.m128i_i64[0] = (__int64)"mulbool";
                  LOWORD(v200) = 259;
                  v116 = sub_A82350(v115, v186, v187, (__int64)&v198);
                  LOWORD(v200) = 257;
                  return (unsigned __int8 *)sub_B51D30(40, v116, (__int64)v172, (__int64)&v198, 0, 0);
                }
              }
              v198.m128i_i64[0] = (__int64)&v186;
              if ( (unsigned __int8)sub_10A3FA0(&v198, v8) && sub_1001970(*((_QWORD *)v186 + 1), 1) )
              {
                LOWORD(v200) = 257;
                v160 = sub_AD6530((__int64)v172, 1);
                return sub_109FEA0((__int64)v186, v9, v160, (const char **)&v198, 0, 0, 0);
              }
              v198.m128i_i64[0] = (__int64)&v186;
              if ( !(unsigned __int8)sub_10A3FA0(&v198, (_BYTE *)v9)
                || (v64 = 1, !sub_1001970(*((_QWORD *)v186 + 1), 1)) )
              {
                v198.m128i_i64[0] = (__int64)&v186;
                v198.m128i_i64[1] = (__int64)&v187;
                if ( (unsigned __int8)sub_115D710(&v198, 17, (unsigned __int8 *)a2)
                  && sub_1001970(*((_QWORD *)v186 + 1), 1) )
                {
                  LOWORD(v200) = 257;
                  v82 = sub_AD6530(*((_QWORD *)v8 + 1), 1);
                  v83 = (__int64 *)a1[2].m128i_i64[0];
                  v84 = v82;
                  v85 = sub_B44900(a2);
                  LOWORD(v197) = 257;
                  v86 = sub_10A0530(v83, (__int64)v187, (__int64)&v193, v85);
                  return sub_109FEA0((__int64)v186, (__int64)v86, v84, (const char **)&v198, 0, 0, 0);
                }
                v198.m128i_i64[0] = (__int64)&v189;
                if ( !(unsigned __int8)sub_F11D70(&v198, (_BYTE *)v9) )
                {
LABEL_122:
                  v199.m128i_i8[0] = 0;
                  v198.m128i_i64[0] = (__int64)&v186;
                  v198.m128i_i64[1] = (__int64)&v190;
                  v199.m128i_i64[1] = (__int64)&v187;
                  if ( (unsigned __int8)sub_1158710((__int64)&v198, a2)
                    && sub_D94970((__int64)v190, (_QWORD *)(unsigned int)(*((_DWORD *)v190 + 2) - 1)) )
                  {
                    v94 = v186;
                    v95 = (__int64 *)a1[2].m128i_i64[0];
                    v198.m128i_i64[0] = (__int64)"isneg";
                    LOWORD(v200) = 259;
                    v96 = sub_10A0880(v95, (__int64)v186, (__int64)&v198);
                    LOWORD(v200) = 257;
                    v97 = (__int64)v96;
                    v98 = sub_AD6530((__int64)v172, (__int64)v94);
                    return sub_109FEA0(v97, (__int64)v187, v98, (const char **)&v198, 0, 0, 0);
                  }
                  v65 = a2;
                  v198 = (__m128i)(unsigned __int64)&v186;
                  v199.m128i_i64[0] = (__int64)&v187;
                  if ( (unsigned __int8)sub_1158870(&v198, a2) )
                  {
                    v66 = (unsigned int **)a1[2].m128i_i64[0];
                    LOWORD(v197) = 257;
                    v67 = *((unsigned __int8 *)v172 + 8);
                    if ( (unsigned int)(v67 - 17) > 1 )
                    {
                      v70 = sub_BCB2A0(*v172);
                    }
                    else
                    {
                      v68 = *((_DWORD *)v172 + 8);
                      BYTE4(v191) = (_BYTE)v67 == 18;
                      LODWORD(v191) = v68;
                      v69 = (__int64 *)sub_BCB2A0(*v172);
                      v65 = v191;
                      v70 = sub_BCE1B0(v69, v191);
                    }
                    v71 = v186;
                    if ( *((_QWORD *)v186 + 1) != v70 )
                    {
                      v65 = 38;
                      v72 = (*(__int64 (__fastcall **)(unsigned int *, __int64, _BYTE *, __int64))(*(_QWORD *)v66[10]
                                                                                                 + 120LL))(
                              v66[10],
                              38,
                              v186,
                              v70);
                      if ( v72 )
                      {
                        v71 = (_BYTE *)v72;
                      }
                      else
                      {
                        LOWORD(v200) = 257;
                        v71 = (_BYTE *)sub_B51D30(38, (__int64)v71, v70, (__int64)&v198, 0, 0);
                        v65 = (__int64)v71;
                        (*(void (__fastcall **)(unsigned int *, _BYTE *, __int64 **, unsigned int *, unsigned int *))(*(_QWORD *)v66[11] + 16LL))(
                          v66[11],
                          v71,
                          &v193,
                          v66[7],
                          v66[8]);
                        v77 = *v66;
                        v78 = (__int64)&(*v66)[4 * *((unsigned int *)v66 + 2)];
                        while ( v77 != (unsigned int *)v78 )
                        {
                          v79 = *((_QWORD *)v77 + 1);
                          v65 = *v77;
                          v77 += 4;
                          sub_B99FD0((__int64)v71, v65, v79);
                        }
                      }
                    }
                    LOWORD(v200) = 257;
                    v159 = sub_AD6530((__int64)v172, v65);
                    return sub_109FEA0((__int64)v71, (__int64)v187, v159, (const char **)&v198, 0, 0, 0);
                  }
                  v199.m128i_i64[0] = 0;
                  v198.m128i_i64[0] = (__int64)&v186;
                  v198.m128i_i64[1] = v168 - 1;
                  v199.m128i_i64[1] = (__int64)&v186;
                  if ( sub_1158C00((__int64)&v198, a2) )
                  {
                    v73 = a1[2].m128i_i64[0];
                    LOWORD(v200) = 257;
                    HIDWORD(v193) = 0;
                    v74 = (__int64 *)sub_BD5C60(a2);
                    v75 = sub_ACD760(v74, v171);
                    v76 = (unsigned __int8 *)sub_B33C40(v73, 1u, (__int64)v186, v75, (__int64)v193, (__int64)&v198);
                    sub_BD6B90(v76, (unsigned __int8 *)a2);
                    return sub_F162A0((__int64)a1, a2, (__int64)v76);
                  }
                  v173 = (unsigned __int8 *)sub_F10620(a1, (unsigned __int8 *)a2);
                  if ( v173 )
                    return v173;
                  v173 = sub_F12440((__int64)a1, (unsigned __int8 *)a2);
                  if ( v173 )
                    return v173;
                  v80 = sub_11571F0((__int64)a1, (__int64)v8, 0);
                  if ( v80 )
                  {
                    LOWORD(v200) = 257;
                    v81 = v9;
LABEL_142:
                    v173 = (unsigned __int8 *)sub_B504D0(25, v81, v80, (__int64)&v198, 0, 0);
                    sub_B447F0(v173, v170);
                    return v173;
                  }
                  v80 = sub_11571F0((__int64)a1, v9, 0);
                  if ( v80 )
                  {
                    LOWORD(v200) = 257;
                    v81 = (__int64)v8;
                    goto LABEL_142;
                  }
                  if ( v171 )
                    goto LABEL_259;
                  v104 = a1 + 6;
                  v105 = 18;
                  v106 = &v198;
                  while ( v105 )
                  {
                    v106->m128i_i32[0] = v104->m128i_i32[0];
                    v104 = (__m128i *)((char *)v104 + 4);
                    v106 = (__m128i *)((char *)v106 + 4);
                    --v105;
                  }
                  v201 = a2;
                  if ( (unsigned int)sub_9AF960((__int64)v8, v9, &v198) != 3 )
                  {
LABEL_259:
                    if ( v170 )
                      return 0;
                    v100 = sub_B44900(a2);
                    v101 = a1 + 6;
                    v102 = &v198;
                    for ( i = 18; i; --i )
                    {
                      v102->m128i_i32[0] = v101->m128i_i32[0];
                      v101 = (__m128i *)((char *)v101 + 4);
                      v102 = (__m128i *)((char *)v102 + 4);
                    }
                    v201 = a2;
                    if ( (unsigned int)sub_9AC590((__int64)v8, v9, &v198, v100) != 3 )
                      return 0;
                  }
                  else
                  {
                    sub_B44850((unsigned __int8 *)a2, 1);
                    if ( v170 )
                      return (unsigned __int8 *)a2;
                    v107 = sub_B44900(a2);
                    v108 = 18;
                    v109 = &v198;
                    v110 = a1 + 6;
                    while ( v108 )
                    {
                      v109->m128i_i32[0] = v110->m128i_i32[0];
                      v110 = (__m128i *)((char *)v110 + 4);
                      v109 = (__m128i *)((char *)v109 + 4);
                      --v108;
                    }
                    v201 = a2;
                    if ( (unsigned int)sub_9AC590((__int64)v8, v9, &v198, v107) != 3 )
                      return (unsigned __int8 *)a2;
                  }
                  sub_B447F0((unsigned __int8 *)a2, 1);
                  return (unsigned __int8 *)a2;
                }
                v198.m128i_i64[0] = (__int64)&v186;
                if ( !(unsigned __int8)sub_10E4070(&v198, v8) || !sub_1001970(*((_QWORD *)v186 + 1), 1) )
                {
                  v199.m128i_i8[0] = 0;
                  v198.m128i_i64[0] = (__int64)&v186;
                  v198.m128i_i64[1] = (__int64)&v193;
                  if ( (unsigned __int8)sub_1158660((__int64)&v198, (__int64)v8)
                    && sub_D94970((__int64)v193, (_QWORD *)(unsigned int)(*((_DWORD *)v193 + 2) - 1)) )
                  {
                    v87 = sub_AD6890((__int64)v189, 0);
                    v88 = (__int64 *)a1[2].m128i_i64[0];
                    v89 = v186;
                    v90 = v87;
                    LOWORD(v200) = 259;
                    v198.m128i_i64[0] = (__int64)"isneg";
                    v91 = sub_10A0880(v88, (__int64)v186, (__int64)&v198);
                    LOWORD(v200) = 257;
                    v92 = (__int64)v91;
                    v93 = sub_AD6530((__int64)v172, (__int64)v89);
                    return sub_109FEA0(v92, v90, v93, (const char **)&v198, 0, 0, 0);
                  }
                  goto LABEL_122;
                }
                v64 = 0;
                v8 = (unsigned __int8 *)sub_AD6890((__int64)v189, 0);
              }
              LOWORD(v200) = 257;
              v99 = sub_AD6530((__int64)v172, v64);
              return sub_109FEA0((__int64)v186, (__int64)v8, v99, (const char **)&v198, 0, 0, 0);
            }
            v19 = *(_QWORD *)(a2 - 64);
            v20 = *(_QWORD *)(v19 + 16);
            if ( !v20 || *(_QWORD *)(v20 + 8) )
            {
              v21 = *(_QWORD *)(a2 - 32);
            }
            else
            {
              v141 = sub_10A7530((__int64 **)&v198, 15, (unsigned __int8 *)v19);
              v21 = *(_QWORD *)(a2 - 32);
              if ( v141 && v21 )
              {
                *(_QWORD *)v199.m128i_i64[0] = v21;
                goto LABEL_18;
              }
            }
            v22 = *(_QWORD *)(v21 + 16);
            if ( !v22 )
              goto LABEL_94;
            if ( *(_QWORD *)(v22 + 8) )
              goto LABEL_94;
            if ( !(unsigned __int8)sub_10A7530((__int64 **)&v198, 15, (unsigned __int8 *)v21) )
              goto LABEL_94;
            v23 = *(_QWORD *)(a2 - 64);
            if ( !v23 )
              goto LABEL_94;
            *(_QWORD *)v199.m128i_i64[0] = v23;
LABEL_18:
            v24 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v200) = 257;
            LOWORD(v197) = 257;
            v18 = sub_A81850(v24, v186, v187, (__int64)&v193, 0, 0);
            return (unsigned __int8 *)sub_B50550(v18, (__int64)&v198, 0, 0);
          }
          return sub_F162A0((__int64)a1, a2, v11);
        }
        v192 = *((_DWORD *)v190 + 2);
        if ( v192 > 0x40 )
          sub_C43780((__int64)&v191, (const void **)v190);
        else
          v191 = *v190;
        sub_C46F20((__int64)&v191, 1u);
        v27 = v192;
        v192 = 0;
        LODWORD(v194) = v27;
        v193 = (__int64 *)v191;
        if ( v27 > 0x40 )
        {
          if ( (unsigned int)sub_C44630((__int64)&v193) == 1 )
            goto LABEL_39;
        }
        else if ( v191 && (v191 & (v191 - 1)) == 0 )
        {
LABEL_39:
          v28 = *((_DWORD *)v190 + 2);
          if ( v28 > 0x40 )
          {
            v167 = *((_DWORD *)v190 + 2);
            v32 = sub_C444A0((__int64)v190);
            v28 = v167;
          }
          else
          {
            _BitScanReverse64(&v29, *v190);
            v30 = v29 ^ 0x3F;
            v31 = 64;
            if ( *v190 )
              v31 = v30;
            v32 = v28 + v31 - 64;
          }
          v164 = sub_D94970((__int64)v189, (_QWORD *)(v28 - 1 - v32));
          sub_969240((__int64 *)&v193);
          sub_969240(&v191);
          if ( v164 )
          {
            if ( v170 )
            {
              if ( *v8 == 56 )
              {
                v55 = *((_QWORD *)v8 + 2);
                if ( v55 )
                {
                  if ( !*(_QWORD *)(v55 + 8) )
                  {
                    v56 = (__int64 *)a1[2].m128i_i64[0];
                    LOWORD(v200) = 257;
                    v57 = sub_AD8D80((__int64)v172, (__int64)v189);
                    v8 = (unsigned __int8 *)sub_F94560(v56, v188, v57, (__int64)&v198, 1u);
                  }
                }
              }
              LOWORD(v200) = 257;
              v173 = (unsigned __int8 *)sub_B504D0(13, v188, (__int64)v8, (__int64)&v198, 0, 0);
              if ( !v171 )
                goto LABEL_47;
            }
            else
            {
              LOWORD(v200) = 257;
              v173 = (unsigned __int8 *)sub_B504D0(13, v188, (__int64)v8, (__int64)&v198, 0, 0);
              if ( !v171 )
                goto LABEL_47;
              if ( *v8 != 55 )
              {
                v54 = (_QWORD *)*v189;
                if ( *((_DWORD *)v189 + 2) > 0x40u )
                  v54 = (_QWORD *)*v54;
                if ( v168 - 1 <= (unsigned __int64)v54 )
                  goto LABEL_47;
              }
            }
            sub_B44850(v173, 1);
LABEL_47:
            sub_B447F0(v173, v170);
            return v173;
          }
          goto LABEL_23;
        }
        sub_969240((__int64 *)&v193);
        sub_969240(&v191);
        goto LABEL_23;
      }
    }
  }
  return v173;
}
