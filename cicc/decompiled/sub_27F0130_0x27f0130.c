// Function: sub_27F0130
// Address: 0x27f0130
//
__int64 __fastcall sub_27F0130(
        unsigned __int8 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        _BYTE *a7,
        __int64 a8)
{
  __int64 v11; // r15
  unsigned __int8 v12; // al
  unsigned __int16 v13; // ax
  _QWORD *v15; // rax
  char v16; // al
  char v17; // r9
  __int64 v18; // r11
  unsigned int v19; // r10d
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 *v25; // r11
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rdi
  unsigned int v29; // ebx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned int v33; // r8d
  unsigned __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  unsigned int v37; // eax
  unsigned __int16 v38; // ax
  int v39; // ecx
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  unsigned __int8 **v43; // rax
  unsigned __int8 *v44; // rdi
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rax
  char v48; // r10
  __int64 v49; // rsi
  _QWORD *v50; // rax
  _QWORD *i; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned int v55; // eax
  unsigned int v56; // ebx
  int v57; // eax
  int v58; // edx
  unsigned int v59; // eax
  int v60; // eax
  int v61; // edx
  __int64 v62; // rcx
  unsigned int v63; // esi
  unsigned __int8 **v64; // rax
  unsigned __int8 *v65; // rdi
  unsigned __int8 *v66; // r13
  unsigned int v67; // r9d
  __int64 v68; // rax
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // rcx
  int v72; // ebx
  __int64 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rdi
  __int64 v76; // rsi
  unsigned int v77; // ecx
  __int64 *v78; // rax
  __int64 v79; // r10
  __int64 v80; // rbx
  __int64 v81; // r13
  char v82; // al
  __int64 v83; // rax
  __int64 v84; // rsi
  _QWORD *v85; // rax
  _QWORD *v86; // rdx
  char v87; // al
  __m128i v88; // xmm3
  __m128i v89; // xmm4
  __m128i v90; // xmm5
  __int64 v91; // r13
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 *v94; // r11
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 *v98; // r11
  __int64 v99; // r13
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rax
  __int64 v103; // rdx
  unsigned __int8 *v104; // r13
  unsigned __int8 *v105; // rbx
  int v106; // esi
  __int64 v107; // r9
  int v108; // esi
  unsigned int v109; // r8d
  unsigned __int8 **v110; // rax
  unsigned __int8 *v111; // r11
  unsigned __int8 *v112; // rsi
  __int64 *v113; // rcx
  __int64 *v114; // r9
  __int64 v115; // r10
  __int64 v116; // rdi
  int v117; // r11d
  __int64 v118; // rsi
  unsigned int v119; // edx
  __int64 *v120; // rax
  __int64 v121; // rbx
  int v122; // eax
  int v123; // r12d
  int v124; // eax
  int v125; // r8d
  int v126; // eax
  int v127; // r9d
  int v128; // eax
  int v129; // r10d
  __int64 v130; // rax
  __int64 v131; // rax
  char v132; // al
  int v133; // eax
  int v134; // r8d
  __int64 *v135; // [rsp+28h] [rbp-4F8h]
  char v136; // [rsp+48h] [rbp-4D8h]
  __int64 *v137; // [rsp+48h] [rbp-4D8h]
  unsigned __int8 *v138; // [rsp+48h] [rbp-4D8h]
  __int64 v140; // [rsp+50h] [rbp-4D0h]
  __int64 *v142; // [rsp+50h] [rbp-4D0h]
  __int64 *v143; // [rsp+50h] [rbp-4D0h]
  __int64 *v144; // [rsp+50h] [rbp-4D0h]
  int v145; // [rsp+50h] [rbp-4D0h]
  unsigned int v146; // [rsp+50h] [rbp-4D0h]
  char v147; // [rsp+58h] [rbp-4C8h]
  unsigned __int8 v148; // [rsp+58h] [rbp-4C8h]
  unsigned __int8 v149; // [rsp+58h] [rbp-4C8h]
  unsigned __int8 *v150; // [rsp+58h] [rbp-4C8h]
  __int64 *v151; // [rsp+58h] [rbp-4C8h]
  __int64 *v152; // [rsp+58h] [rbp-4C8h]
  __m128i v153; // [rsp+60h] [rbp-4C0h] BYREF
  __m128i v154; // [rsp+70h] [rbp-4B0h] BYREF
  __m128i v155; // [rsp+80h] [rbp-4A0h] BYREF
  __m128i v156; // [rsp+90h] [rbp-490h] BYREF
  __m256i v157; // [rsp+A0h] [rbp-480h]
  __m128i v158; // [rsp+C0h] [rbp-460h]
  __m128i v159; // [rsp+D0h] [rbp-450h]
  _QWORD v160[2]; // [rsp+E0h] [rbp-440h] BYREF
  _BYTE v161[324]; // [rsp+F0h] [rbp-430h] BYREF
  int v162; // [rsp+234h] [rbp-2ECh]
  __int64 v163; // [rsp+238h] [rbp-2E8h]
  _QWORD *v164; // [rsp+240h] [rbp-2E0h] BYREF
  __int64 v165; // [rsp+248h] [rbp-2D8h] BYREF
  __int64 v166; // [rsp+250h] [rbp-2D0h]
  __m128i v167; // [rsp+258h] [rbp-2C8h] BYREF
  __int64 v168; // [rsp+268h] [rbp-2B8h]
  __m128i v169; // [rsp+270h] [rbp-2B0h] BYREF
  __m128i v170; // [rsp+280h] [rbp-2A0h] BYREF
  char v171[8]; // [rsp+290h] [rbp-290h] BYREF
  unsigned int v172; // [rsp+298h] [rbp-288h]
  _QWORD v173[2]; // [rsp+3A0h] [rbp-180h] BYREF
  char v174; // [rsp+3B0h] [rbp-170h]
  _BYTE *v175; // [rsp+3B8h] [rbp-168h]
  __int64 v176; // [rsp+3C0h] [rbp-160h]
  _BYTE v177[28]; // [rsp+3C8h] [rbp-158h] BYREF
  int v178; // [rsp+3E4h] [rbp-13Ch]
  __int64 v179; // [rsp+3E8h] [rbp-138h]
  __int16 v180; // [rsp+448h] [rbp-D8h]
  _QWORD v181[2]; // [rsp+450h] [rbp-D0h] BYREF
  __int64 v182; // [rsp+460h] [rbp-C0h]
  __int64 v183; // [rsp+468h] [rbp-B8h] BYREF
  unsigned int v184; // [rsp+470h] [rbp-B0h]
  char v185; // [rsp+4E8h] [rbp-38h] BYREF

  v11 = *(_QWORD *)a5;
  v12 = *a1;
  if ( *a1 == 61 )
  {
    v13 = *((_WORD *)a1 + 1);
    if ( ((v13 >> 7) & 6) != 0 || (v13 & 1) != 0 )
      return 0;
    v15 = (_QWORD *)*((_QWORD *)a1 - 4);
    v165 = -1;
    v166 = 0;
    v164 = v15;
    v167 = 0u;
    v168 = 0;
    v16 = sub_CF5020((__int64)a2, (__int64)&v164, 0);
    v17 = a6;
    v18 = a8;
    if ( (v16 & 2) != 0 )
    {
      if ( (a1[7] & 0x20) == 0 || (v21 = sub_B91C10((__int64)a1, 6), v17 = a6, v18 = a8, !v21) )
      {
        v140 = v18;
        v136 = v17;
        if ( !sub_B46500(a1) || v136 )
        {
          v22 = *((_QWORD *)a1 - 4);
          v23 = sub_B43CC0((__int64)a1);
          v24 = (_QWORD *)sub_9208B0(v23, *((_QWORD *)a1 + 1));
          v25 = (__int64 *)v140;
          v165 = v26;
          v164 = v24;
          if ( !(_BYTE)v26 && *(_BYTE *)v22 > 0x15u )
          {
            v27 = *(_QWORD *)(v22 + 16);
            if ( v27 )
            {
              v28 = a3;
              v29 = 0;
              do
              {
                ++v29;
                v30 = *(_QWORD *)(v27 + 24);
                if ( v29 > (unsigned int)qword_4FFE408 )
                  break;
                if ( *(_BYTE *)v30 == 85 )
                {
                  v31 = *(_QWORD *)(v30 - 32);
                  if ( v31 )
                  {
                    if ( !*(_BYTE *)v31
                      && *(_QWORD *)(v31 + 24) == *(_QWORD *)(v30 + 80)
                      && (*(_BYTE *)(v31 + 33) & 0x20) != 0
                      && *(_DWORD *)(v31 + 36) == 205
                      && !*(_QWORD *)(v30 + 16) )
                    {
                      v32 = *(_QWORD *)(v30 - 32LL * (*(_DWORD *)(v30 + 4) & 0x7FFFFFF));
                      v33 = *(_DWORD *)(v32 + 32);
                      v34 = *(_QWORD *)(v32 + 24);
                      v35 = 1LL << ((unsigned __int8)v33 - 1);
                      if ( v33 > 0x40 )
                      {
                        if ( (*(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6)) & v35) != 0 )
                          goto LABEL_21;
                        v36 = 8LL * *(_QWORD *)v34;
                      }
                      else
                      {
                        v36 = v34 & v35;
                        if ( v36 )
                          goto LABEL_21;
                        if ( v33 )
                          v36 = 8 * ((__int64)(v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33));
                      }
                      if ( v36 >= (unsigned __int64)v164 )
                      {
                        v137 = v25;
                        sub_B196A0(v28, *(_QWORD *)(v30 + 40), **(_QWORD **)(a4 + 32));
                        v25 = v137;
                        v19 = v37;
                        if ( (_BYTE)v37 )
                          return v19;
                      }
                    }
                  }
                }
LABEL_21:
                v27 = *(_QWORD *)(v27 + 8);
              }
              while ( v27 );
            }
          }
          v60 = *(_DWORD *)(v11 + 56);
          if ( v60 )
          {
            v61 = v60 - 1;
            v62 = *(_QWORD *)(v11 + 40);
            v63 = (v60 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v64 = (unsigned __int8 **)(v62 + 16LL * v63);
            v65 = *v64;
            if ( a1 == *v64 )
            {
LABEL_80:
              v66 = v64[1];
LABEL_81:
              v67 = 0;
              if ( (a1[7] & 0x20) != 0 )
              {
                v142 = v25;
                v68 = sub_B91C10((__int64)a1, 16);
                v25 = v142;
                v67 = v68 != 0;
              }
              v143 = v25;
              v69 = sub_27ED270((__int64 *)v11, v66, a4, (__int64)a1, a7, v67);
              v72 = v69;
              if ( v143 )
              {
                if ( (_BYTE)v69 )
                {
                  if ( (unsigned __int8)sub_D48480(a4, *((_QWORD *)a1 - 4), v70, v71) )
                  {
                    v91 = *v143;
                    v92 = sub_B2BE50(*v143);
                    v93 = sub_B6EA50(v92);
                    v94 = v143;
                    if ( v93
                      || (v130 = sub_B2BE50(v91),
                          v131 = sub_B6F970(v130),
                          v132 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v131 + 48LL))(v131),
                          v94 = v143,
                          v132) )
                    {
                      v151 = v94;
                      sub_B176B0(
                        (__int64)&v164,
                        (__int64)"licm",
                        (__int64)"LoadWithLoopInvariantAddressInvalidated",
                        39,
                        (__int64)a1);
                      sub_B18290(
                        (__int64)&v164,
                        "failed to move load with loop-invariant address because the loop may invalidate its value",
                        0x59u);
                      v156.m128i_i32[2] = v165;
                      *(__m128i *)&v157.m256i_u64[1] = _mm_loadu_si128(&v167);
                      v156.m128i_i8[12] = BYTE4(v165);
                      v98 = v151;
                      v158 = _mm_loadu_si128(&v169);
                      v157.m256i_i64[0] = v166;
                      v156.m128i_i64[0] = (__int64)&unk_49D9D40;
                      v159 = _mm_loadu_si128(&v170);
                      v157.m256i_i64[3] = v168;
                      v160[0] = v161;
                      v160[1] = 0x400000000LL;
                      if ( v172 )
                      {
                        sub_27EFAF0((__int64)v160, (__int64)v171, v172, v95, v96, v97);
                        v98 = v151;
                      }
                      v152 = v98;
                      v164 = &unk_49D9D40;
                      v161[320] = v177[24];
                      v162 = v178;
                      v163 = v179;
                      v156.m128i_i64[0] = (__int64)&unk_49D9DB0;
                      sub_23FD590((__int64)v171);
                      sub_1049740(v152, (__int64)&v156);
                      v156.m128i_i64[0] = (__int64)&unk_49D9D40;
                      sub_23FD590((__int64)v160);
                    }
                  }
                }
              }
              return v72 ^ 1u;
            }
            v124 = 1;
            while ( v65 != (unsigned __int8 *)-4096LL )
            {
              v125 = v124 + 1;
              v63 = v61 & (v124 + v63);
              v64 = (unsigned __int8 **)(v62 + 16LL * v63);
              v65 = *v64;
              if ( a1 == *v64 )
                goto LABEL_80;
              v124 = v125;
            }
          }
          v66 = 0;
          goto LABEL_81;
        }
        return 0;
      }
    }
    return 1;
  }
  if ( v12 == 85 )
  {
    v52 = *((_QWORD *)a1 - 4);
    if ( v52
      && !*(_BYTE *)v52
      && *(_QWORD *)(v52 + 24) == *((_QWORD *)a1 + 10)
      && (*(_BYTE *)(v52 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v52 + 36) - 68) <= 3 )
    {
      return 0;
    }
    if ( (unsigned __int8)sub_B46790(a1, 0) )
      return 0;
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 6) )
      return 0;
    if ( (unsigned __int8)sub_B49560((__int64)a1, 6) )
      return 0;
    v53 = sub_B43CB0((__int64)a1);
    if ( (unsigned __int8)sub_B2D610(v53, 49) )
      return 0;
    v54 = *((_QWORD *)a1 - 4);
    if ( v54 && !*(_BYTE *)v54 && *(_QWORD *)(v54 + 24) == *((_QWORD *)a1 + 10) && *(_DWORD *)(v54 + 36) == 11 )
      return 1;
    v55 = sub_CF5CA0((__int64)a2, (__int64)a1);
    if ( !v55 )
      return 1;
    if ( (((unsigned __int8)(v55 >> 6) | (unsigned __int8)((v55 >> 4) | v55 | (v55 >> 2))) & 2) != 0 )
      return 0;
    v56 = v55 & 0xFFFFFFFC;
    if ( (v55 & 0xFFFFFFFC) != 0 )
    {
      v113 = *(__int64 **)(a4 + 32);
      v114 = *(__int64 **)(a4 + 40);
      if ( v113 != v114 )
      {
        v115 = *(_QWORD *)(*(_QWORD *)a5 + 104LL);
        v116 = *(unsigned int *)(*(_QWORD *)a5 + 120LL);
        v117 = v116 - 1;
        while ( 1 )
        {
          v118 = *v113;
          if ( (_DWORD)v116 )
          {
            v119 = v117 & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
            v120 = (__int64 *)(v115 + 16LL * v119);
            v121 = *v120;
            if ( v118 == *v120 )
            {
LABEL_143:
              if ( (__int64 *)(v115 + 16 * v116) != v120 && v120[1] )
                return 0;
            }
            else
            {
              v122 = 1;
              while ( v121 != -4096 )
              {
                v123 = v122 + 1;
                v119 = v117 & (v122 + v119);
                v120 = (__int64 *)(v115 + 16LL * v119);
                v121 = *v120;
                if ( v118 == *v120 )
                  goto LABEL_143;
                v122 = v123;
              }
            }
          }
          if ( v114 == ++v113 )
            return 1;
        }
      }
      return 1;
    }
    v57 = *a1;
    v58 = v57 - 29;
    if ( v57 == 40 )
    {
      v59 = sub_B491D0((__int64)a1);
    }
    else
    {
      v59 = 0;
      if ( v58 != 56 )
      {
        v59 = 2;
        if ( v58 != 5 )
LABEL_174:
          BUG();
      }
    }
    v99 = -32 - 32LL * v59;
    if ( (a1[7] & 0x80u) != 0 )
    {
      v100 = sub_BD2BC0((__int64)a1);
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)((v100 + v101) >> 4) )
          goto LABEL_174;
      }
      else if ( (unsigned int)((v100 + v101 - sub_BD2BC0((__int64)a1)) >> 4) )
      {
        if ( (a1[7] & 0x80u) == 0 )
          goto LABEL_174;
        v145 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v102 = sub_BD2BC0((__int64)a1);
        v56 = *(_DWORD *)(v102 + v103 - 4) - v145;
      }
    }
    v104 = &a1[v99 - 32LL * v56];
    v105 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    if ( v105 == v104 )
      return 1;
    v146 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
    while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v105 + 8LL) + 8LL) != 14 )
    {
LABEL_134:
      v105 += 32;
      if ( v104 == v105 )
        return 1;
    }
    v106 = *(_DWORD *)(v11 + 56);
    v107 = *(_QWORD *)(v11 + 40);
    if ( v106 )
    {
      v108 = v106 - 1;
      v109 = v108 & v146;
      v110 = (unsigned __int8 **)(v107 + 16LL * (v108 & v146));
      v111 = *v110;
      if ( a1 == *v110 )
      {
LABEL_138:
        v112 = v110[1];
        goto LABEL_139;
      }
      v128 = 1;
      while ( v111 != (unsigned __int8 *)-4096LL )
      {
        v129 = v128 + 1;
        v109 = v108 & (v128 + v109);
        v110 = (unsigned __int8 **)(v107 + 16LL * v109);
        v111 = *v110;
        if ( a1 == *v110 )
          goto LABEL_138;
        v128 = v129;
      }
    }
    v112 = 0;
LABEL_139:
    if ( (unsigned __int8)sub_27ED270((__int64 *)v11, v112, a4, (__int64)a1, a7, 0) )
      return 0;
    goto LABEL_134;
  }
  if ( v12 == 64 )
    return sub_27EC2D0((__int64)a1, a4, a5);
  if ( v12 != 62 )
    return 1;
  v38 = *((_WORD *)a1 + 1);
  if ( ((v38 >> 7) & 6) != 0 || (v38 & 1) != 0 )
    return 0;
  v19 = sub_27EC2D0((__int64)a1, a4, a5);
  if ( (_BYTE)v19 )
    return v19;
  if ( *a7 )
    return 0;
  v39 = *(_DWORD *)(v11 + 56);
  v40 = *(_QWORD *)(v11 + 40);
  if ( v39 )
  {
    v41 = v39 - 1;
    v42 = v41 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v43 = (unsigned __int8 **)(v40 + 16LL * v42);
    v44 = *v43;
    if ( a1 == *v43 )
    {
LABEL_44:
      v138 = v43[1];
      goto LABEL_45;
    }
    v133 = 1;
    while ( v44 != (unsigned __int8 *)-4096LL )
    {
      v134 = v133 + 1;
      v42 = v41 & (v133 + v42);
      v43 = (unsigned __int8 **)(v40 + 16LL * v42);
      v44 = *v43;
      if ( a1 == *v43 )
        goto LABEL_44;
      v133 = v134;
    }
  }
  v138 = 0;
LABEL_45:
  v164 = a2;
  v45 = &v167.m128i_i64[1];
  v165 = (__int64)a2;
  v166 = 0;
  v167.m128i_i64[0] = 1;
  do
  {
    *v45 = -4;
    v45 += 5;
    *(v45 - 4) = -3;
    *(v45 - 3) = -4;
    *(v45 - 2) = -3;
  }
  while ( v45 != v173 );
  v173[1] = 0;
  v173[0] = v181;
  v175 = v177;
  v176 = 0x400000000LL;
  v180 = 256;
  v174 = 0;
  v181[1] = 0;
  v182 = 1;
  v181[0] = &unk_49DDBE8;
  v46 = &v183;
  do
  {
    *v46 = -4096;
    v46 += 2;
  }
  while ( v46 != (__int64 *)&v185 );
  v47 = sub_27EB760((_QWORD *)v11, (__int64)&v164, (__int64)a7, v138);
  v48 = 0;
  if ( v47 != *(_QWORD *)(v11 + 128) )
  {
    v49 = *(_QWORD *)(v47 + 64);
    if ( *(_BYTE *)(a4 + 84) )
    {
      v50 = *(_QWORD **)(a4 + 64);
      for ( i = &v50[*(unsigned int *)(a4 + 76)]; i != v50; ++v50 )
      {
        if ( v49 == *v50 )
          goto LABEL_55;
      }
    }
    else
    {
      v73 = sub_C8CA60(a4 + 56, v49);
      v48 = 0;
      if ( v73 )
        goto LABEL_55;
    }
  }
  v135 = *(__int64 **)(a4 + 40);
  if ( *(__int64 **)(a4 + 32) == v135 )
  {
LABEL_113:
    v48 = 1;
    goto LABEL_55;
  }
  v144 = *(__int64 **)(a4 + 32);
  while ( 1 )
  {
    v74 = *(unsigned int *)(v11 + 88);
    v75 = *(_QWORD *)(v11 + 72);
    v76 = *v144;
    if ( (_DWORD)v74 )
    {
      v77 = (v74 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v78 = (__int64 *)(v75 + 16LL * v77);
      v79 = *v78;
      if ( v76 != *v78 )
      {
        v126 = 1;
        while ( v79 != -4096 )
        {
          v127 = v126 + 1;
          v77 = (v74 - 1) & (v126 + v77);
          v78 = (__int64 *)(v75 + 16LL * v77);
          v79 = *v78;
          if ( v76 == *v78 )
            goto LABEL_97;
          v126 = v127;
        }
        goto LABEL_112;
      }
LABEL_97:
      if ( v78 != (__int64 *)(v75 + 16 * v74) )
      {
        v80 = v78[1];
        if ( v80 )
        {
          v81 = *(_QWORD *)(v80 + 8);
          if ( v81 != v80 )
            break;
        }
      }
    }
LABEL_112:
    if ( v135 == ++v144 )
      goto LABEL_113;
  }
  while ( 1 )
  {
    if ( !v81 )
      BUG();
    v82 = *(_BYTE *)(v81 - 32);
    if ( v82 != 26 )
    {
      if ( v82 == 27 )
      {
        v87 = **(_BYTE **)(v81 + 40);
        v150 = *(unsigned __int8 **)(v81 + 40);
        if ( v87 == 61 )
          goto LABEL_108;
        if ( v87 == 85 )
        {
          sub_D66630(&v153, (__int64)a1);
          v88 = _mm_loadu_si128(&v153);
          v89 = _mm_loadu_si128(&v154);
          v90 = _mm_loadu_si128(&v155);
          v158.m128i_i8[0] = 1;
          v156 = v88;
          *(__m128i *)v157.m256i_i8 = v89;
          *(__m128i *)&v157.m256i_u64[2] = v90;
          if ( (unsigned __int8)sub_CF63E0(v164, v150, &v156, (__int64)&v165) )
            goto LABEL_108;
        }
      }
      goto LABEL_111;
    }
    v83 = sub_27EB760((_QWORD *)v11, (__int64)&v164, (__int64)a7, (_BYTE *)(v81 - 32));
    if ( v83 == *(_QWORD *)(v11 + 128) )
      goto LABEL_110;
    v84 = *(_QWORD *)(v83 + 64);
    if ( *(_BYTE *)(a4 + 84) )
      break;
    if ( sub_C8CA60(a4 + 56, v84) )
      goto LABEL_108;
LABEL_110:
    if ( !a7[16] && !sub_1041420(v11, (__int64)v138, v81 - 32) )
      goto LABEL_108;
LABEL_111:
    v81 = *(_QWORD *)(v81 + 8);
    if ( v80 == v81 )
      goto LABEL_112;
  }
  v85 = *(_QWORD **)(a4 + 64);
  v86 = &v85[*(unsigned int *)(a4 + 76)];
  if ( v85 == v86 )
    goto LABEL_110;
  while ( v84 != *v85 )
  {
    if ( v86 == ++v85 )
      goto LABEL_110;
  }
LABEL_108:
  v48 = 0;
LABEL_55:
  v181[0] = &unk_49DDBE8;
  if ( (v182 & 1) == 0 )
  {
    v147 = v48;
    sub_C7D6A0(v183, 16LL * v184, 8);
    v48 = v147;
  }
  v148 = v48;
  nullsub_184();
  v19 = v148;
  if ( v175 != v177 )
  {
    _libc_free((unsigned __int64)v175);
    v19 = v148;
  }
  if ( (v167.m128i_i8[0] & 1) == 0 )
  {
    v149 = v19;
    sub_C7D6A0(v167.m128i_i64[1], 40LL * (unsigned int)v168, 8);
    return v149;
  }
  return v19;
}
