// Function: sub_F29CA0
// Address: 0xf29ca0
//
__int64 __fastcall sub_F29CA0(const __m128i *a1, unsigned __int8 *a2)
{
  unsigned int v3; // r12d
  char v4; // al
  __int64 v5; // rbx
  _BYTE *v6; // r8
  __int8 v7; // bl
  unsigned __int8 *v8; // r12
  __int64 v9; // r13
  _BYTE *v11; // r8
  unsigned __int8 *v12; // r10
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  _BYTE *v17; // rsi
  __int64 v18; // rax
  _BYTE *v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 *v27; // rax
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int8 *v42; // rax
  __int64 v43; // r12
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int8 *v50; // rax
  __int64 v51; // r12
  __int64 v52; // rcx
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rdx
  __m128i v58; // xmm1
  __int64 v59; // rsi
  unsigned __int64 v60; // xmm2_8
  __int64 v61; // rax
  __m128i v62; // xmm3
  __int64 v63; // rax
  unsigned __int8 *v64; // rdx
  __int64 v65; // rbx
  __int64 v66; // rcx
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // r12
  __int64 v70; // rax
  unsigned __int8 *v71; // rax
  __int64 v72; // rbx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rsi
  __m128i v79; // xmm5
  unsigned __int64 v80; // xmm6_8
  __int64 v81; // rdx
  __m128i v82; // xmm7
  __int64 v83; // rax
  __int64 v84; // rbx
  unsigned __int8 *v85; // rax
  __int64 v86; // r12
  __int64 v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // r13
  __int64 v91; // rax
  unsigned __int8 *v92; // rax
  __int64 v93; // r12
  __int64 v94; // rdx
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // r13
  __int64 v98; // rax
  __int64 v99; // rsi
  __m128i v100; // xmm5
  unsigned __int64 v101; // xmm6_8
  __int64 v102; // rax
  __m128i v103; // xmm7
  __int64 v104; // rdx
  __int64 v105; // rdx
  __m128i v106; // xmm1
  unsigned __int64 v107; // xmm2_8
  __m128i v108; // xmm3
  __int64 v109; // rax
  int v110; // ebx
  __int64 v111; // r8
  __int64 v112; // rcx
  __int64 v113; // r9
  _BYTE *v114; // rsi
  _BYTE *v115; // rdx
  __int64 v116; // rbx
  int v117; // r12d
  int v118; // eax
  unsigned __int8 *v119; // rax
  __int64 v120; // r13
  __int64 v121; // rcx
  __int64 v122; // rcx
  __int64 v123; // rax
  __int64 v124; // rax
  unsigned __int8 *v125; // rax
  __int64 v126; // r13
  __int64 v127; // rcx
  __int64 v128; // rcx
  __int64 v129; // rax
  __int64 v130; // rax
  bool v131; // r13
  unsigned __int64 v132; // rax
  __int64 v133; // rcx
  __int64 v134; // r9
  __int64 v135; // r8
  int v136; // edx
  unsigned __int64 v137; // rax
  __int64 v138; // rcx
  __int64 v139; // rdx
  _BYTE *v140; // rax
  __int64 v141; // rdx
  _BYTE *v142; // rax
  unsigned __int8 *v143; // [rsp+8h] [rbp-D8h]
  __int64 v144; // [rsp+10h] [rbp-D0h]
  __int64 v145; // [rsp+18h] [rbp-C8h]
  __int64 v146; // [rsp+18h] [rbp-C8h]
  __int64 v147; // [rsp+18h] [rbp-C8h]
  int v148; // [rsp+18h] [rbp-C8h]
  __int64 v149; // [rsp+20h] [rbp-C0h]
  _BYTE *v150; // [rsp+30h] [rbp-B0h]
  __int64 v151; // [rsp+30h] [rbp-B0h]
  __int64 v152; // [rsp+30h] [rbp-B0h]
  __int64 v153; // [rsp+30h] [rbp-B0h]
  __int64 v154; // [rsp+30h] [rbp-B0h]
  __int64 v155; // [rsp+30h] [rbp-B0h]
  __int64 v156; // [rsp+30h] [rbp-B0h]
  unsigned __int8 v157; // [rsp+38h] [rbp-A8h]
  bool v158; // [rsp+38h] [rbp-A8h]
  __int64 v159; // [rsp+38h] [rbp-A8h]
  __int64 v160; // [rsp+38h] [rbp-A8h]
  __int64 v161; // [rsp+38h] [rbp-A8h]
  unsigned int v162; // [rsp+40h] [rbp-A0h]
  char v163; // [rsp+46h] [rbp-9Ah]
  unsigned __int8 v164; // [rsp+47h] [rbp-99h]
  __int64 v165; // [rsp+58h] [rbp-88h] BYREF
  __m128i v166; // [rsp+60h] [rbp-80h] BYREF
  __m128i v167; // [rsp+70h] [rbp-70h]
  unsigned __int64 v168; // [rsp+80h] [rbp-60h]
  unsigned __int8 *v169; // [rsp+88h] [rbp-58h]
  __m128i v170; // [rsp+90h] [rbp-50h]
  __int64 v171; // [rsp+A0h] [rbp-40h]

  v157 = 0;
  v164 = *a2;
  v162 = *a2 - 29;
  while ( 1 )
  {
    if ( !sub_B46D50(a2) )
      goto LABEL_7;
    v3 = sub_F13260(*((unsigned __int8 **)a2 - 8));
    if ( v3 < (unsigned int)sub_F13260(*((unsigned __int8 **)a2 - 4)) )
      v157 = sub_B506C0(a2) ^ 1;
    v4 = sub_B46D50(a2);
    v5 = *((_QWORD *)a2 - 4);
    v6 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( v4 )
    {
      sub_F0D460((__int64)&v166, (__int64)a1, *((unsigned __int8 **)a2 - 8), *((char **)a2 - 4));
      v7 = v167.m128i_i8[0];
      if ( v167.m128i_i8[0] )
      {
        v41 = v166.m128i_i64[0];
        if ( (a2[7] & 0x40) != 0 )
          v42 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v42 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v43 = *(_QWORD *)v42;
        if ( *(_QWORD *)v42 )
        {
          v44 = *((_QWORD *)v42 + 1);
          **((_QWORD **)v42 + 2) = v44;
          if ( v44 )
            *(_QWORD *)(v44 + 16) = *((_QWORD *)v42 + 2);
        }
        *(_QWORD *)v42 = v41;
        if ( v41 )
        {
          v45 = *(_QWORD *)(v41 + 16);
          *((_QWORD *)v42 + 1) = v45;
          if ( v45 )
            *(_QWORD *)(v45 + 16) = v42 + 8;
          *((_QWORD *)v42 + 2) = v41 + 16;
          *(_QWORD *)(v41 + 16) = v42;
        }
        if ( *(_BYTE *)v43 > 0x1Cu )
        {
          v46 = a1[2].m128i_i64[1];
          v165 = v43;
          v47 = v46 + 2096;
          sub_F200C0(v46 + 2096, &v165);
          v48 = *(_QWORD *)(v43 + 16);
          if ( v48 )
          {
            if ( !*(_QWORD *)(v48 + 8) )
            {
              v165 = *(_QWORD *)(v48 + 24);
              sub_F200C0(v47, &v165);
            }
          }
        }
        v49 = v166.m128i_i64[1];
        if ( (a2[7] & 0x40) != 0 )
          v50 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v50 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v51 = *((_QWORD *)v50 + 4);
        if ( v51 )
        {
          v52 = *((_QWORD *)v50 + 5);
          **((_QWORD **)v50 + 6) = v52;
          if ( v52 )
            *(_QWORD *)(v52 + 16) = *((_QWORD *)v50 + 6);
        }
        *((_QWORD *)v50 + 4) = v49;
        if ( v49 )
        {
          v53 = *(_QWORD *)(v49 + 16);
          *((_QWORD *)v50 + 5) = v53;
          if ( v53 )
            *(_QWORD *)(v53 + 16) = v50 + 40;
          *((_QWORD *)v50 + 6) = v49 + 16;
          *(_QWORD *)(v49 + 16) = v50 + 32;
        }
        if ( *(_BYTE *)v51 > 0x1Cu )
        {
          v54 = a1[2].m128i_i64[1];
          v165 = v51;
          v55 = v54 + 2096;
          sub_F200C0(v54 + 2096, &v165);
          v56 = *(_QWORD *)(v51 + 16);
          if ( v56 )
          {
            if ( !*(_QWORD *)(v56 + 8) )
            {
              v165 = *(_QWORD *)(v56 + 24);
              sub_F200C0(v55, &v165);
            }
          }
        }
        v157 = v7;
      }
LABEL_7:
      v6 = (_BYTE *)*((_QWORD *)a2 - 8);
      v5 = *((_QWORD *)a2 - 4);
    }
    v8 = 0;
    v9 = 0;
    v150 = v6;
    if ( (unsigned __int8)(*v6 - 42) < 0x12u )
      v8 = v6;
    if ( (unsigned __int8)(*(_BYTE *)v5 - 42) <= 0x11u )
      v9 = v5;
    v163 = sub_B46CC0(a2);
    if ( !v163 )
      return v157;
    v11 = v150;
    if ( !v8 || v164 != *v8 )
    {
      if ( !v9 || v164 != *(_BYTE *)v9 )
        goto LABEL_18;
      goto LABEL_142;
    }
    v99 = *((_QWORD *)v8 - 4);
    v100 = _mm_loadu_si128(a1 + 7);
    v101 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v102 = a1[10].m128i_i64[0];
    v154 = *((_QWORD *)v8 - 8);
    v103 = _mm_loadu_si128(a1 + 9);
    v166 = _mm_loadu_si128(a1 + 6);
    v168 = v101;
    v171 = v102;
    v169 = a2;
    v167 = v100;
    v170 = v103;
    v104 = sub_101E7C0(v162, v99, v5, &v166);
    if ( v104 )
    {
      if ( (a2[7] & 0x40) != 0 )
        v119 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v119 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v120 = *(_QWORD *)v119;
      if ( *(_QWORD *)v119 )
      {
        v121 = *((_QWORD *)v119 + 1);
        **((_QWORD **)v119 + 2) = v121;
        if ( v121 )
          *(_QWORD *)(v121 + 16) = *((_QWORD *)v119 + 2);
      }
      *(_QWORD *)v119 = v154;
      if ( v154 )
      {
        v122 = *(_QWORD *)(v154 + 16);
        *((_QWORD *)v119 + 1) = v122;
        if ( v122 )
          *(_QWORD *)(v122 + 16) = v119 + 8;
        *((_QWORD *)v119 + 2) = v154 + 16;
        *(_QWORD *)(v154 + 16) = v119;
      }
      if ( *(_BYTE *)v120 > 0x1Cu )
      {
        v123 = a1[2].m128i_i64[1];
        v156 = v104;
        v166.m128i_i64[0] = v120;
        v159 = v123 + 2096;
        sub_F200C0(v123 + 2096, v166.m128i_i64);
        v124 = *(_QWORD *)(v120 + 16);
        v104 = v156;
        if ( v124 )
        {
          if ( !*(_QWORD *)(v124 + 8) )
          {
            v166.m128i_i64[0] = *(_QWORD *)(v124 + 24);
            sub_F200C0(v159, v166.m128i_i64);
            v104 = v156;
          }
        }
      }
      if ( (a2[7] & 0x40) != 0 )
        v125 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v125 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v126 = *((_QWORD *)v125 + 4);
      if ( v126 )
      {
        v127 = *((_QWORD *)v125 + 5);
        **((_QWORD **)v125 + 6) = v127;
        if ( v127 )
          *(_QWORD *)(v127 + 16) = *((_QWORD *)v125 + 6);
      }
      *((_QWORD *)v125 + 4) = v104;
      v128 = *(_QWORD *)(v104 + 16);
      *((_QWORD *)v125 + 5) = v128;
      if ( v128 )
        *(_QWORD *)(v128 + 16) = v125 + 40;
      *((_QWORD *)v125 + 6) = v104 + 16;
      *(_QWORD *)(v104 + 16) = v125 + 32;
      if ( *(_BYTE *)v126 > 0x1Cu )
      {
        v129 = a1[2].m128i_i64[1];
        v166.m128i_i64[0] = v126;
        v160 = v129 + 2096;
        sub_F200C0(v129 + 2096, v166.m128i_i64);
        v130 = *(_QWORD *)(v126 + 16);
        if ( v130 )
        {
          if ( !*(_QWORD *)(v130 + 8) )
          {
            v166.m128i_i64[0] = *(_QWORD *)(v130 + 24);
            sub_F200C0(v160, v166.m128i_i64);
          }
        }
      }
      v131 = sub_F06FC0(a2);
      if ( v131 )
        v131 = sub_F06FC0(v8);
      v132 = *a2;
      if ( (unsigned __int8)v132 <= 0x36u )
      {
        v133 = 0x40540000000000LL;
        if ( _bittest64(&v133, v132) )
        {
          if ( (a2[1] & 4) != 0 )
          {
            if ( *(_BYTE *)v99 == 17 )
            {
              v134 = v99 + 24;
              goto LABEL_197;
            }
            v139 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v99 + 8) + 8LL) - 17;
            if ( (unsigned int)v139 <= 1 && *(_BYTE *)v99 <= 0x15u )
            {
              v140 = sub_AD7630(v99, 0, v139);
              if ( v140 )
              {
                if ( *v140 == 17 )
                {
                  v134 = (__int64)(v140 + 24);
LABEL_197:
                  v135 = v5 + 24;
                  if ( *(_BYTE *)v5 == 17 )
                  {
LABEL_198:
                    v136 = *a2;
                    LOBYTE(v165) = 0;
                    switch ( v136 )
                    {
                      case ',':
                        sub_C46BD0((__int64)&v166, v134, v135, &v165);
                        if ( v166.m128i_i32[2] > 0x40u )
                          goto LABEL_202;
                        goto LABEL_204;
                      case '.':
                        sub_C4A7C0((__int64)&v166, v134, v135, (bool *)&v165);
                        if ( v166.m128i_i32[2] > 0x40u )
                          goto LABEL_202;
                        goto LABEL_204;
                      case '*':
                        sub_C45F70((__int64)&v166, v134, v135, &v165);
                        if ( v166.m128i_i32[2] > 0x40u )
                        {
LABEL_202:
                          if ( v166.m128i_i64[0] )
                            j_j___libc_free_0_0(v166.m128i_i64[0]);
                        }
LABEL_204:
                        if ( !(_BYTE)v165 )
                        {
                          v137 = *v8;
                          if ( (unsigned __int8)v137 <= 0x36u )
                          {
                            v138 = 0x40540000000000LL;
                            if ( _bittest64(&v138, v137) )
                            {
                              if ( (v8[1] & 4) != 0 )
                              {
                                sub_F08660(a2);
                                if ( v131 )
                                  sub_B447F0(a2, 1);
                                sub_B44850(a2, 1);
                                goto LABEL_137;
                              }
                            }
                          }
                        }
                        break;
                    }
                  }
                  else
                  {
                    v161 = v134;
                    v141 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
                    if ( (unsigned int)v141 <= 1 && *(_BYTE *)v5 <= 0x15u )
                    {
                      v142 = sub_AD7630(v5, 0, v141);
                      if ( v142 )
                      {
                        if ( *v142 == 17 )
                        {
                          v134 = v161;
                          v135 = (__int64)(v142 + 24);
                          goto LABEL_198;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      sub_F08660(a2);
      if ( v131 )
        goto LABEL_67;
      goto LABEL_137;
    }
    if ( v9 && v164 == *(_BYTE *)v9 )
    {
      v11 = (_BYTE *)*((_QWORD *)a2 - 8);
LABEL_142:
      v105 = *(_QWORD *)(v9 - 64);
      v106 = _mm_loadu_si128(a1 + 7);
      v152 = *(_QWORD *)(v9 - 32);
      v107 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v166 = _mm_loadu_si128(a1 + 6);
      v108 = _mm_loadu_si128(a1 + 9);
      v109 = a1[10].m128i_i64[0];
      v167 = v106;
      v168 = v107;
      v171 = v109;
      v169 = a2;
      v170 = v108;
      v63 = sub_101E7C0(v162, v11, v105, &v166);
      if ( v63 )
        goto LABEL_93;
    }
    if ( !sub_B46CC0(a2) )
      return v157;
LABEL_18:
    if ( !sub_B46D50(a2) )
      return v157;
    v12 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v13 = *v12;
    if ( (unsigned __int8)v13 > 0x1Cu && (unsigned int)(v13 - 67) <= 0xC )
    {
      v14 = *((_QWORD *)v12 + 2);
      if ( v14 )
      {
        if ( !*(_QWORD *)(v14 + 8) && v13 == 68 )
        {
          v110 = *a2;
          if ( (unsigned int)(v110 - 57) <= 2 )
          {
            v111 = *((_QWORD *)v12 - 4);
            if ( (unsigned __int8)(*(_BYTE *)v111 - 42) <= 0x11u )
            {
              v112 = *(_QWORD *)(v111 + 16);
              if ( v112 )
              {
                if ( !*(_QWORD *)(v112 + 8) && (_BYTE)v110 == *(_BYTE *)v111 )
                {
                  v113 = *((_QWORD *)a2 - 4);
                  v143 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
                  if ( *(_BYTE *)v113 <= 0x15u )
                  {
                    v114 = *(_BYTE **)(v111 - 32);
                    v144 = *((_QWORD *)v12 - 4);
                    if ( *v114 <= 0x15u )
                    {
                      v147 = *((_QWORD *)a2 - 4);
                      v155 = a1[5].m128i_i64[1];
                      v115 = (_BYTE *)sub_96F480(0x27u, (__int64)v114, *(_QWORD *)(v113 + 8), v155);
                      if ( v115 )
                      {
                        v116 = sub_96E6C0(v110 - 29, v147, v115, v155);
                        if ( v116 )
                        {
                          sub_F20660((__int64)a1, (__int64)v143, 0, *(_QWORD *)(v144 - 64));
                          sub_F20660((__int64)a1, (__int64)a2, 1u, v116);
                          sub_B44F30(a2);
                          sub_B44F30(v143);
                          goto LABEL_137;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if ( v8 )
    {
      if ( v164 == *v8 )
      {
        v57 = *((_QWORD *)v8 - 8);
        v58 = _mm_loadu_si128(a1 + 7);
        v152 = *((_QWORD *)v8 - 4);
        v59 = *((_QWORD *)a2 - 4);
        v60 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v61 = a1[10].m128i_i64[0];
        v166 = _mm_loadu_si128(a1 + 6);
        v62 = _mm_loadu_si128(a1 + 9);
        v167 = v58;
        v168 = v60;
        v171 = v61;
        v169 = a2;
        v170 = v62;
        v63 = sub_101E7C0(v162, v59, v57, &v166);
        if ( v63 )
        {
LABEL_93:
          if ( (a2[7] & 0x40) != 0 )
            v64 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v64 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v65 = *(_QWORD *)v64;
          if ( *(_QWORD *)v64 )
          {
            v66 = *((_QWORD *)v64 + 1);
            **((_QWORD **)v64 + 2) = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = *((_QWORD *)v64 + 2);
          }
          *(_QWORD *)v64 = v63;
          v67 = *(_QWORD *)(v63 + 16);
          *((_QWORD *)v64 + 1) = v67;
          if ( v67 )
            *(_QWORD *)(v67 + 16) = v64 + 8;
          *((_QWORD *)v64 + 2) = v63 + 16;
          *(_QWORD *)(v63 + 16) = v64;
          if ( *(_BYTE *)v65 > 0x1Cu )
          {
            v68 = a1[2].m128i_i64[1];
            v166.m128i_i64[0] = v65;
            v69 = v68 + 2096;
            sub_F200C0(v68 + 2096, v166.m128i_i64);
            v70 = *(_QWORD *)(v65 + 16);
            if ( v70 )
            {
              if ( !*(_QWORD *)(v70 + 8) )
              {
                v166.m128i_i64[0] = *(_QWORD *)(v70 + 24);
                sub_F200C0(v69, v166.m128i_i64);
              }
            }
          }
          if ( (a2[7] & 0x40) != 0 )
            v71 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v71 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v72 = *((_QWORD *)v71 + 4);
          if ( v72 )
          {
            v73 = *((_QWORD *)v71 + 5);
            **((_QWORD **)v71 + 6) = v73;
            if ( v73 )
              *(_QWORD *)(v73 + 16) = *((_QWORD *)v71 + 6);
          }
          *((_QWORD *)v71 + 4) = v152;
          if ( v152 )
          {
            v74 = *(_QWORD *)(v152 + 16);
            *((_QWORD *)v71 + 5) = v74;
            if ( v74 )
              *(_QWORD *)(v74 + 16) = v71 + 40;
            *((_QWORD *)v71 + 6) = v152 + 16;
            *(_QWORD *)(v152 + 16) = v71 + 32;
          }
          if ( *(_BYTE *)v72 > 0x1Cu )
          {
            v75 = a1[2].m128i_i64[1];
            v166.m128i_i64[0] = v72;
            v76 = v75 + 2096;
            sub_F200C0(v75 + 2096, v166.m128i_i64);
            v77 = *(_QWORD *)(v72 + 16);
            if ( v77 )
            {
              if ( !*(_QWORD *)(v77 + 8) )
              {
                v166.m128i_i64[0] = *(_QWORD *)(v77 + 24);
                sub_F200C0(v76, v166.m128i_i64);
              }
            }
          }
LABEL_136:
          sub_F08660(a2);
          goto LABEL_137;
        }
      }
    }
    if ( !v9 )
      return v157;
    if ( v164 == *(_BYTE *)v9 )
    {
      v78 = *(_QWORD *)(v9 - 32);
      v79 = _mm_loadu_si128(a1 + 7);
      v80 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v81 = *((_QWORD *)a2 - 8);
      v153 = *(_QWORD *)(v9 - 64);
      v82 = _mm_loadu_si128(a1 + 9);
      v166 = _mm_loadu_si128(a1 + 6);
      v83 = a1[10].m128i_i64[0];
      v168 = v80;
      v167 = v79;
      v171 = v83;
      v169 = a2;
      v170 = v82;
      v84 = sub_101E7C0(v162, v78, v81, &v166);
      if ( v84 )
      {
        if ( (a2[7] & 0x40) != 0 )
          v85 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v85 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v86 = *(_QWORD *)v85;
        if ( *(_QWORD *)v85 )
        {
          v87 = *((_QWORD *)v85 + 1);
          **((_QWORD **)v85 + 2) = v87;
          if ( v87 )
            *(_QWORD *)(v87 + 16) = *((_QWORD *)v85 + 2);
        }
        *(_QWORD *)v85 = v153;
        if ( v153 )
        {
          v88 = *(_QWORD *)(v153 + 16);
          *((_QWORD *)v85 + 1) = v88;
          if ( v88 )
            *(_QWORD *)(v88 + 16) = v85 + 8;
          *((_QWORD *)v85 + 2) = v153 + 16;
          *(_QWORD *)(v153 + 16) = v85;
        }
        if ( *(_BYTE *)v86 > 0x1Cu )
        {
          v89 = a1[2].m128i_i64[1];
          v166.m128i_i64[0] = v86;
          v90 = v89 + 2096;
          sub_F200C0(v89 + 2096, v166.m128i_i64);
          v91 = *(_QWORD *)(v86 + 16);
          if ( v91 )
          {
            if ( !*(_QWORD *)(v91 + 8) )
            {
              v166.m128i_i64[0] = *(_QWORD *)(v91 + 24);
              sub_F200C0(v90, v166.m128i_i64);
            }
          }
        }
        if ( (a2[7] & 0x40) != 0 )
          v92 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v92 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v93 = *((_QWORD *)v92 + 4);
        if ( v93 )
        {
          v94 = *((_QWORD *)v92 + 5);
          **((_QWORD **)v92 + 6) = v94;
          if ( v94 )
            *(_QWORD *)(v94 + 16) = *((_QWORD *)v92 + 6);
        }
        *((_QWORD *)v92 + 4) = v84;
        v95 = *(_QWORD *)(v84 + 16);
        *((_QWORD *)v92 + 5) = v95;
        if ( v95 )
          *(_QWORD *)(v95 + 16) = v92 + 40;
        *((_QWORD *)v92 + 6) = v84 + 16;
        *(_QWORD *)(v84 + 16) = v92 + 32;
        if ( *(_BYTE *)v93 > 0x1Cu )
        {
          v96 = a1[2].m128i_i64[1];
          v166.m128i_i64[0] = v93;
          v97 = v96 + 2096;
          sub_F200C0(v96 + 2096, v166.m128i_i64);
          v98 = *(_QWORD *)(v93 + 16);
          if ( v98 )
          {
            if ( !*(_QWORD *)(v98 + 8) )
            {
              v166.m128i_i64[0] = *(_QWORD *)(v98 + 24);
              sub_F200C0(v97, v166.m128i_i64);
            }
          }
        }
        goto LABEL_136;
      }
    }
    if ( !v8 )
      return v157;
    if ( v164 != *v8 )
      return v157;
    if ( v164 != *(_BYTE *)v9 )
      return v157;
    v15 = *((_QWORD *)v8 + 2);
    if ( !v15 )
      return v157;
    if ( *(_QWORD *)(v15 + 8) )
      return v157;
    v16 = *((_QWORD *)v8 - 8);
    if ( !v16 )
      return v157;
    v17 = (_BYTE *)*((_QWORD *)v8 - 4);
    if ( *v17 > 0x15u )
      return v157;
    v18 = *(_QWORD *)(v9 + 16);
    if ( !v18 )
      return v157;
    if ( *(_QWORD *)(v18 + 8) )
      return v157;
    v145 = *(_QWORD *)(v9 - 64);
    if ( !v145 )
      return v157;
    v19 = *(_BYTE **)(v9 - 32);
    if ( *v19 > 0x15u )
      return v157;
    v151 = sub_96E6C0(v162, (__int64)v17, v19, a1[5].m128i_i64[1]);
    if ( !v151 )
      return v157;
    v158 = sub_F06FC0(a2);
    if ( v158 && (v158 = sub_F06FC0(v8)) && (v158 = sub_F06FC0((unsigned __int8 *)v9)) && v162 == 13 )
    {
      LOWORD(v168) = 257;
      v20 = sub_B504D0(13, v16, v145, (__int64)&v166, 0, 0);
      sub_B447F0((unsigned __int8 *)v20, 1);
      if ( !(unsigned __int8)sub_920620(v20) )
        goto LABEL_43;
    }
    else
    {
      LOWORD(v168) = 257;
      v20 = sub_B504D0(v162, v16, v145, (__int64)&v166, 0, 0);
      if ( !(unsigned __int8)sub_920620(v20) )
        goto LABEL_43;
    }
    v148 = sub_B45210(v9);
    v117 = sub_B45210((__int64)v8);
    v118 = sub_B45210((__int64)a2);
    sub_B45150(v20, v148 & v117 & v118);
LABEL_43:
    v21 = *((_QWORD *)a2 + 6);
    v166.m128i_i64[0] = v21;
    if ( v21 )
    {
      sub_B96E90((__int64)&v166, v21, 1);
      v22 = *(_QWORD *)(v20 + 48);
      v23 = v20 + 48;
      if ( !v22 )
        goto LABEL_46;
    }
    else
    {
      v22 = *(_QWORD *)(v20 + 48);
      v23 = v20 + 48;
      if ( !v22 )
        goto LABEL_48;
    }
    v146 = v23;
    sub_B91220(v23, v22);
    v23 = v146;
LABEL_46:
    v24 = (unsigned __int8 *)v166.m128i_i64[0];
    *(_QWORD *)(v20 + 48) = v166.m128i_i64[0];
    if ( v24 )
      sub_B976B0((__int64)&v166, v24, v23);
LABEL_48:
    v25 = v149;
    LOWORD(v25) = 0;
    sub_B44220((_QWORD *)v20, (__int64)(a2 + 24), v25);
    v26 = a1[2].m128i_i64[1];
    v166.m128i_i64[0] = v20;
    sub_F200C0(v26 + 2096, v166.m128i_i64);
    sub_BD6B90((unsigned __int8 *)v20, (unsigned __int8 *)v9);
    if ( (a2[7] & 0x40) != 0 )
      v27 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v27 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v28 = *(_QWORD *)v27;
    if ( *(_QWORD *)v27 )
    {
      v29 = *((_QWORD *)v27 + 1);
      **((_QWORD **)v27 + 2) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = *((_QWORD *)v27 + 2);
    }
    *(_QWORD *)v27 = v20;
    v30 = *(_QWORD *)(v20 + 16);
    *((_QWORD *)v27 + 1) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = v27 + 8;
    *((_QWORD *)v27 + 2) = v20 + 16;
    *(_QWORD *)(v20 + 16) = v27;
    if ( *(_BYTE *)v28 > 0x1Cu )
    {
      v31 = a1[2].m128i_i64[1];
      v166.m128i_i64[0] = v28;
      v32 = v31 + 2096;
      sub_F200C0(v31 + 2096, v166.m128i_i64);
      v33 = *(_QWORD *)(v28 + 16);
      if ( v33 )
      {
        if ( !*(_QWORD *)(v33 + 8) )
        {
          v166.m128i_i64[0] = *(_QWORD *)(v33 + 24);
          sub_F200C0(v32, v166.m128i_i64);
        }
      }
    }
    if ( (a2[7] & 0x40) != 0 )
      v34 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v34 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v35 = *((_QWORD *)v34 + 4);
    if ( v35 )
    {
      v36 = *((_QWORD *)v34 + 5);
      **((_QWORD **)v34 + 6) = v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = *((_QWORD *)v34 + 6);
    }
    *((_QWORD *)v34 + 4) = v151;
    v37 = *(_QWORD *)(v151 + 16);
    *((_QWORD *)v34 + 5) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = v34 + 40;
    *((_QWORD *)v34 + 6) = v151 + 16;
    *(_QWORD *)(v151 + 16) = v34 + 32;
    if ( *(_BYTE *)v35 > 0x1Cu )
    {
      v38 = a1[2].m128i_i64[1];
      v166.m128i_i64[0] = v35;
      v39 = v38 + 2096;
      sub_F200C0(v38 + 2096, v166.m128i_i64);
      v40 = *(_QWORD *)(v35 + 16);
      if ( v40 )
      {
        if ( !*(_QWORD *)(v40 + 8) )
        {
          v166.m128i_i64[0] = *(_QWORD *)(v40 + 24);
          sub_F200C0(v39, v166.m128i_i64);
        }
      }
    }
    sub_F08660(a2);
    if ( v158 )
LABEL_67:
      sub_B447F0(a2, 1);
LABEL_137:
    v157 = v163;
  }
}
