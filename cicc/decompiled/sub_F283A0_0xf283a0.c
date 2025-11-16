// Function: sub_F283A0
// Address: 0xf283a0
//
unsigned __int8 *__fastcall sub_F283A0(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rdx
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm3
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r11
  unsigned __int8 *v22; // r8
  _BYTE *v23; // rax
  __int64 v24; // rdx
  int v25; // r9d
  __int64 v26; // r10
  __int64 v27; // r11
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned __int8 *v33; // rax
  unsigned int v34; // ecx
  __int64 v35; // r14
  __int64 v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // r8
  __int64 *v43; // r9
  unsigned __int8 v44; // al
  unsigned int *v45; // rdi
  char *v46; // r10
  unsigned int *v47; // rcx
  char *v48; // rbx
  bool v49; // al
  unsigned int *v50; // r11
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rax
  unsigned __int8 *v57; // rax
  __int64 v58; // r13
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  unsigned int **v63; // rdi
  __int64 v64; // r13
  __int64 v65; // r14
  _QWORD *v66; // rax
  __int64 v67; // r12
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  unsigned int *v74; // rdx
  __int64 v75; // rax
  unsigned int *v76; // rbx
  unsigned int *v77; // rax
  __int64 v78; // r13
  unsigned int *v79; // r12
  __int64 v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r8
  __int64 v84; // rdx
  unsigned __int64 v85; // r9
  unsigned int **v86; // rdi
  __int64 v87; // rax
  __int64 *v88; // rbx
  __int64 v89; // r10
  __int64 v90; // rax
  unsigned __int8 v91; // al
  _QWORD *v92; // rax
  __int64 v93; // r9
  __int64 v94; // r15
  __int64 v95; // rcx
  __int64 v96; // rbx
  __int64 v97; // r13
  __int64 v98; // r15
  __int64 v99; // rt0
  __int64 v100; // r12
  __int64 v101; // rdx
  unsigned int v102; // esi
  __int64 v103; // rt1
  __int64 v104; // rdx
  __int64 *v105; // rax
  __int64 v106; // rdx
  int v107; // r8d
  __int64 *v108; // rax
  _BYTE *v109; // rax
  __int64 v110; // [rsp+8h] [rbp-158h]
  __int64 v111; // [rsp+8h] [rbp-158h]
  __int64 v112; // [rsp+10h] [rbp-150h]
  __int64 v113; // [rsp+10h] [rbp-150h]
  __int64 v114; // [rsp+18h] [rbp-148h]
  _DWORD *v115; // [rsp+18h] [rbp-148h]
  __int64 v116; // [rsp+20h] [rbp-140h]
  int v117; // [rsp+28h] [rbp-138h]
  unsigned __int8 *v118; // [rsp+28h] [rbp-138h]
  _DWORD *v119; // [rsp+28h] [rbp-138h]
  char v120; // [rsp+37h] [rbp-129h]
  unsigned __int8 *v121; // [rsp+38h] [rbp-128h]
  __int64 v122; // [rsp+38h] [rbp-128h]
  __int64 *v123; // [rsp+40h] [rbp-120h]
  __int64 v124; // [rsp+48h] [rbp-118h]
  __int64 v125; // [rsp+48h] [rbp-118h]
  unsigned int v126; // [rsp+48h] [rbp-118h]
  unsigned __int8 *v127; // [rsp+48h] [rbp-118h]
  __int64 v128; // [rsp+48h] [rbp-118h]
  __int64 v129; // [rsp+50h] [rbp-110h]
  int v130; // [rsp+50h] [rbp-110h]
  __int64 v131; // [rsp+50h] [rbp-110h]
  __int64 v132; // [rsp+50h] [rbp-110h]
  __int64 v133; // [rsp+50h] [rbp-110h]
  int v134; // [rsp+50h] [rbp-110h]
  __int64 v135; // [rsp+50h] [rbp-110h]
  __int64 v136; // [rsp+60h] [rbp-100h]
  char *v137; // [rsp+60h] [rbp-100h]
  __int64 v138; // [rsp+60h] [rbp-100h]
  __int64 v139; // [rsp+60h] [rbp-100h]
  __int64 v140; // [rsp+60h] [rbp-100h]
  int v141; // [rsp+68h] [rbp-F8h]
  __int64 v142; // [rsp+68h] [rbp-F8h]
  __int64 v143; // [rsp+68h] [rbp-F8h]
  __int64 v144; // [rsp+68h] [rbp-F8h]
  __int64 v145; // [rsp+68h] [rbp-F8h]
  unsigned int *v146; // [rsp+68h] [rbp-F8h]
  int v147; // [rsp+68h] [rbp-F8h]
  int v148; // [rsp+7Ch] [rbp-E4h] BYREF
  __int64 v149[4]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v150; // [rsp+A0h] [rbp-C0h]
  __int64 v151[4]; // [rsp+B0h] [rbp-B0h] BYREF
  __int16 v152; // [rsp+D0h] [rbp-90h]
  __m128i v153; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v154; // [rsp+F0h] [rbp-70h] BYREF
  unsigned __int64 v155; // [rsp+100h] [rbp-60h]
  __int64 v156; // [rsp+108h] [rbp-58h]
  __m128i v157; // [rsp+110h] [rbp-50h]
  __int64 v158; // [rsp+120h] [rbp-40h]

  v2 = a2;
  v3 = (__int64)a1;
  v4 = *(_QWORD *)(a2 - 32);
  v5 = *(unsigned int *)(a2 + 80);
  v6 = _mm_loadu_si128(a1 + 6);
  v7 = _mm_loadu_si128(a1 + 7);
  v8 = _mm_loadu_si128(a1 + 9);
  v9 = a1[10].m128i_i64[0];
  v155 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v156 = a2;
  v10 = *(_QWORD *)(a2 + 72);
  v158 = v9;
  v153 = v6;
  v154 = v7;
  v157 = v8;
  v11 = sub_1002A30(v4, v10, v5, &v153);
  if ( v11 )
    return sub_F162A0(v3, v2, v11);
  if ( *(_DWORD *)(v2 + 80) != 1 )
    goto LABEL_44;
  v13 = **(_DWORD **)(v2 + 72);
  if ( v13 )
    goto LABEL_44;
  v14 = *(_QWORD *)(v2 - 32);
  if ( *(_BYTE *)v14 != 85 )
    goto LABEL_44;
  v15 = *(_QWORD *)(v14 - 32);
  if ( !v15 )
    goto LABEL_44;
  if ( *(_BYTE *)v15 )
    goto LABEL_44;
  if ( *(_QWORD *)(v15 + 24) != *(_QWORD *)(v14 + 80) )
    goto LABEL_44;
  if ( *(_DWORD *)(v15 + 36) != 179 )
    goto LABEL_44;
  v16 = *(_QWORD *)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v16 != 86 )
    goto LABEL_44;
  if ( (*(_BYTE *)(v16 + 7) & 0x40) != 0 )
  {
    v17 = *(_QWORD **)(v16 - 8);
    if ( !*v17 )
      goto LABEL_44;
  }
  else
  {
    v17 = (_QWORD *)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
    if ( !*v17 )
      goto LABEL_44;
  }
  if ( v17[4] )
  {
    if ( v17[8] )
    {
      v18 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
      v19 = *(_QWORD *)(v18 + 16);
      if ( v19 )
      {
        if ( !*(_QWORD *)(v19 + 8) )
        {
          v20 = *(_QWORD *)(v4 + 16);
          if ( v20 )
          {
            v21 = *(_QWORD *)(v20 + 8);
            if ( !v21 )
            {
              v22 = *(unsigned __int8 **)(v18 - 32);
              v136 = a1[2].m128i_i64[0];
              v116 = *(_QWORD *)(v18 - 96);
              v23 = *(_BYTE **)(v18 - 64);
              v24 = (unsigned __int8)*v23;
              v121 = v23;
              if ( (_BYTE)v24 == 18
                || (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v23 + 1) + 8LL) - 17 <= 1
                && (unsigned __int8)v24 <= 0x15u
                && (v127 = *(unsigned __int8 **)(v18 - 32),
                    v23 = sub_AD7630((__int64)v23, 0, v24),
                    v13 = 0,
                    v21 = 0,
                    v22 = v127,
                    v23)
                && *v23 == 18 )
              {
                v120 = 1;
                v123 = (__int64 *)(v23 + 24);
                goto LABEL_21;
              }
              v104 = *v22;
              if ( (_BYTE)v104 == 18 )
              {
                v105 = (__int64 *)(v22 + 24);
                v120 = 0;
                v22 = v121;
                v123 = v105;
                goto LABEL_21;
              }
              v135 = v21;
              v147 = v13;
              if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v22 + 1) + 8LL) - 17 <= 1
                && (unsigned __int8)v104 <= 0x15u )
              {
                v109 = sub_AD7630((__int64)v22, 0, v104);
                if ( v109 )
                {
                  if ( *v109 == 18 )
                  {
                    v120 = 0;
                    v22 = v121;
                    v123 = (__int64 *)(v109 + 24);
                    v21 = v135;
                    v13 = v147;
LABEL_21:
                    v124 = (__int64)v22;
                    v141 = v13;
                    v129 = v21;
                    sub_D5F1F0(v136, v2);
                    v151[0] = (__int64)"frexp";
                    v25 = v141;
                    v152 = 259;
                    v149[0] = v124;
                    v26 = *(_QWORD *)(v4 - 32);
                    if ( v26 )
                    {
                      v27 = v129;
                      if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v4 + 80) )
                        v27 = *(_QWORD *)(v4 + 80);
                      else
                        v26 = 0;
                    }
                    else
                    {
                      v27 = 0;
                    }
                    LOWORD(v155) = 257;
                    v28 = *(_QWORD *)(v136 + 120);
                    v29 = *(_QWORD *)(v136 + 112);
                    v30 = v29 + 56 * v28;
                    if ( v29 != v30 )
                    {
                      v31 = *(_QWORD *)(v136 + 112);
                      do
                      {
                        v32 = *(_QWORD *)(v31 + 40) - *(_QWORD *)(v31 + 32);
                        v31 += 56;
                        v25 += v32 >> 3;
                      }
                      while ( v30 != v31 );
                    }
                    v110 = v26;
                    v130 = v25 + 2;
                    LOBYTE(v117) = 16 * (_DWORD)v28 != 0;
                    v112 = v27;
                    v114 = *(_QWORD *)(v136 + 112);
                    v125 = *(_QWORD *)(v136 + 120);
                    v33 = (unsigned __int8 *)sub_BD2CC0(
                                               88,
                                               ((unsigned __int64)(unsigned int)(16 * v28) << 32)
                                             | (unsigned int)(v25 + 2));
                    v142 = (__int64)v33;
                    if ( v33 )
                    {
                      v34 = v130 & 0x7FFFFFF | (v117 << 28);
                      v118 = v33;
                      sub_B44260((__int64)v33, **(_QWORD **)(v112 + 16), 56, v34, 0, 0);
                      *(_QWORD *)(v142 + 72) = 0;
                      sub_B4A290(v142, v112, v110, v149, 1, (__int64)&v153, v114, v125);
                    }
                    else
                    {
                      v118 = 0;
                    }
                    if ( *(_BYTE *)(v136 + 108) )
                    {
                      v108 = (__int64 *)sub_BD5C60((__int64)v118);
                      *(_QWORD *)(v142 + 72) = sub_A7A090((__int64 *)(v142 + 72), v108, -1, 72);
                    }
                    if ( (unsigned __int8)sub_920620((__int64)v118) )
                    {
                      v106 = *(_QWORD *)(v136 + 96);
                      v107 = *(_DWORD *)(v136 + 104);
                      if ( v106 )
                      {
                        v134 = *(_DWORD *)(v136 + 104);
                        sub_B99FD0(v142, 3u, v106);
                        v107 = v134;
                      }
                      sub_B45150(v142, v107);
                    }
                    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v136 + 88) + 16LL))(
                      *(_QWORD *)(v136 + 88),
                      v142,
                      v151,
                      *(_QWORD *)(v136 + 56),
                      *(_QWORD *)(v136 + 64));
                    if ( *(_QWORD *)v136 != *(_QWORD *)v136 + 16LL * *(unsigned int *)(v136 + 8) )
                    {
                      v113 = v4;
                      v35 = *(_QWORD *)v136 + 16LL * *(unsigned int *)(v136 + 8);
                      v111 = v3;
                      v36 = *(_QWORD *)v136;
                      do
                      {
                        v37 = *(_QWORD *)(v36 + 8);
                        v38 = *(_DWORD *)v36;
                        v36 += 16;
                        sub_B99FD0(v142, v38, v37);
                      }
                      while ( v35 != v36 );
                      v4 = v113;
                      v3 = v111;
                    }
                    sub_B45260(v118, v4, 1);
                    v153.m128i_i64[0] = (__int64)"mantissa";
                    LOWORD(v155) = 259;
                    LODWORD(v151[0]) = 0;
                    v143 = sub_94D3D0((unsigned int **)v136, v142, (__int64)v151, 1, (__int64)&v153);
                    v119 = (_DWORD *)*v123;
                    v115 = sub_C33340();
                    if ( v119 == v115 )
                    {
                      sub_C41050(v151, (__int64)v123, &v148, 1u);
                      sub_C3C840(&v153, v151);
                      sub_C3C840(v149, &v153);
                      sub_969EE0((__int64)&v153);
                      sub_969EE0((__int64)v151);
                    }
                    else
                    {
                      sub_C3C390(v151, v123, &v148, 1);
                      sub_C338E0((__int64)&v153, (__int64)v151);
                      sub_C407B0(v149, v153.m128i_i64, v119);
                      sub_C338F0((__int64)&v153);
                      sub_C338F0((__int64)v151);
                    }
                    v131 = sub_AD8F10(*((_QWORD *)v121 + 1), v149);
                    v153.m128i_i64[0] = (__int64)"select.frexp";
                    LOWORD(v155) = 259;
                    v39 = sub_B45210(v18);
                    BYTE4(v151[0]) = 1;
                    LODWORD(v151[0]) = v39;
                    v40 = v131;
                    if ( !v120 )
                    {
                      v40 = v143;
                      v143 = v131;
                    }
                    v144 = sub_B36280((unsigned int **)v136, v116, v40, v143, v151[0], (__int64)&v153, 0);
                    if ( v115 == (_DWORD *)v149[0] )
                      sub_969EE0((__int64)v149);
                    else
                      sub_C338F0((__int64)v149);
                    v11 = v144;
                    if ( v144 )
                      return sub_F162A0(v3, v2, v11);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
LABEL_44:
  if ( *(_BYTE *)v4 == 94 )
  {
    v45 = *(unsigned int **)(v2 + 72);
    v46 = *(char **)(v4 + 72);
    v47 = &v45[*(unsigned int *)(v2 + 80)];
    v132 = *(unsigned int *)(v2 + 80);
    v145 = (__int64)v45;
    v48 = &v46[4 * *(unsigned int *)(v4 + 80)];
    v49 = v48 == v46;
    if ( v47 != v45 && v48 != v46 )
    {
      v50 = *(unsigned int **)(v2 + 72);
      while ( *(_DWORD *)v46 == *v50 )
      {
        ++v50;
        v46 += 4;
        v49 = v48 == v46;
        if ( v50 == v47 )
          goto LABEL_65;
        if ( v48 == v46 )
          goto LABEL_75;
      }
      LOWORD(v155) = 257;
      v51 = *(_QWORD *)(v4 - 64);
      v41 = (__int64)sub_BD2C40(104, unk_3F10A14);
      if ( v41 )
      {
        v52 = sub_B501B0(*(_QWORD *)(v51 + 8), v45, v132);
        sub_B44260(v41, v52, 64, 1u, 0, 0);
        if ( *(_QWORD *)(v41 - 32) )
        {
          v53 = *(_QWORD *)(v41 - 24);
          **(_QWORD **)(v41 - 16) = v53;
          if ( v53 )
            *(_QWORD *)(v53 + 16) = *(_QWORD *)(v41 - 16);
        }
        *(_QWORD *)(v41 - 32) = v51;
        v54 = *(_QWORD *)(v51 + 16);
        *(_QWORD *)(v41 - 24) = v54;
        if ( v54 )
          *(_QWORD *)(v54 + 16) = v41 - 24;
        *(_QWORD *)(v41 - 16) = v51 + 16;
        *(_QWORD *)(v51 + 16) = v41 - 32;
        *(_QWORD *)(v41 + 72) = v41 + 88;
        *(_QWORD *)(v41 + 80) = 0x400000000LL;
        sub_B50030(v41, v45, v132, (__int64)&v153);
      }
      return (unsigned __int8 *)v41;
    }
    v50 = *(unsigned int **)(v2 + 72);
    if ( v47 == v45 )
    {
LABEL_65:
      if ( v49 )
      {
        v11 = *(_QWORD *)(v4 - 32);
        return sub_F162A0(v3, v2, v11);
      }
    }
LABEL_75:
    if ( v47 == v50 )
    {
      v63 = *(unsigned int ***)(v3 + 32);
      LOWORD(v155) = 257;
      v137 = v46;
      v64 = sub_94D3D0(v63, *(_QWORD *)(v4 - 64), v145, v132, (__int64)&v153);
      LOWORD(v155) = 257;
      v65 = *(_QWORD *)(v4 - 32);
      v66 = sub_BD2C40(104, unk_3F148BC);
      v67 = (v48 - v137) >> 2;
      v41 = (__int64)v66;
      if ( v66 )
      {
        sub_B44260((__int64)v66, *(_QWORD *)(v64 + 8), 65, 2u, 0, 0);
        *(_QWORD *)(v41 + 72) = v41 + 88;
        *(_QWORD *)(v41 + 80) = 0x400000000LL;
        sub_B4FD20(v41, v64, v65, v137, v67, (__int64)&v153);
      }
      return (unsigned __int8 *)v41;
    }
    if ( v48 == v46 )
    {
      LOWORD(v155) = 257;
      v58 = *(_QWORD *)(v4 - 32);
      v146 = v50;
      v59 = v47 - v50;
      v41 = (__int64)sub_BD2C40(104, unk_3F10A14);
      if ( v41 )
      {
        v60 = sub_B501B0(*(_QWORD *)(v58 + 8), v146, v59);
        sub_B44260(v41, v60, 64, 1u, 0, 0);
        if ( *(_QWORD *)(v41 - 32) )
        {
          v61 = *(_QWORD *)(v41 - 24);
          **(_QWORD **)(v41 - 16) = v61;
          if ( v61 )
            *(_QWORD *)(v61 + 16) = *(_QWORD *)(v41 - 16);
        }
        *(_QWORD *)(v41 - 32) = v58;
        v62 = *(_QWORD *)(v58 + 16);
        *(_QWORD *)(v41 - 24) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = v41 - 24;
        *(_QWORD *)(v41 - 16) = v58 + 16;
        *(_QWORD *)(v58 + 16) = v41 - 32;
        *(_QWORD *)(v41 + 72) = v41 + 88;
        *(_QWORD *)(v41 + 80) = 0x400000000LL;
        sub_B50030(v41, v146, v59, (__int64)&v153);
      }
      return (unsigned __int8 *)v41;
    }
  }
  v41 = sub_F23930(v3, v2);
  if ( v41 )
    return (unsigned __int8 *)v41;
  v44 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 != 61 )
    goto LABEL_48;
  v55 = *(_QWORD *)(v4 + 8);
  if ( *(_BYTE *)(v55 + 8) == 15 )
  {
    if ( sub_BCEA30(v55) )
      return (unsigned __int8 *)v41;
    if ( sub_B46500((unsigned __int8 *)v4) )
      goto LABEL_72;
  }
  else if ( sub_B46500((unsigned __int8 *)v4) )
  {
    return (unsigned __int8 *)v41;
  }
  if ( (*(_BYTE *)(v4 + 2) & 1) == 0 )
  {
    v56 = *(_QWORD *)(v4 + 16);
    if ( v56 )
    {
      if ( !*(_QWORD *)(v56 + 8) )
      {
        v153.m128i_i64[0] = (__int64)&v154;
        v153.m128i_i64[1] = 0x400000000LL;
        v68 = sub_BCB2D0(*(_QWORD **)(*(_QWORD *)(v3 + 32) + 72LL));
        v69 = sub_ACD640(v68, 0, 0);
        v72 = v153.m128i_u32[2];
        v73 = v153.m128i_u32[2] + 1LL;
        if ( v73 > v153.m128i_u32[3] )
        {
          sub_C8D5F0((__int64)&v153, &v154, v73, 8u, v70, v71);
          v72 = v153.m128i_u32[2];
        }
        *(_QWORD *)(v153.m128i_i64[0] + 8 * v72) = v69;
        v74 = *(unsigned int **)(v2 + 72);
        v75 = *(unsigned int *)(v2 + 80);
        ++v153.m128i_i32[2];
        v76 = v74;
        v77 = &v74[v75];
        if ( v74 != v77 )
        {
          v133 = v4;
          v138 = v2;
          v78 = v3;
          v79 = v77;
          do
          {
            v80 = *v76;
            v81 = sub_BCB2D0(*(_QWORD **)(*(_QWORD *)(v78 + 32) + 72LL));
            v82 = sub_ACD640(v81, v80, 0);
            v84 = v153.m128i_u32[2];
            v85 = v153.m128i_u32[2] + 1LL;
            if ( v85 > v153.m128i_u32[3] )
            {
              v128 = v82;
              sub_C8D5F0((__int64)&v153, &v154, v153.m128i_u32[2] + 1LL, 8u, v83, v85);
              v84 = v153.m128i_u32[2];
              v82 = v128;
            }
            ++v76;
            *(_QWORD *)(v153.m128i_i64[0] + 8 * v84) = v82;
            ++v153.m128i_i32[2];
          }
          while ( v79 != v76 );
          v3 = v78;
          v4 = v133;
          v2 = v138;
        }
        sub_D5F1F0(*(_QWORD *)(v3 + 32), v4);
        v86 = *(unsigned int ***)(v3 + 32);
        v152 = 257;
        v87 = sub_921130(
                v86,
                *(_QWORD *)(v4 + 8),
                *(_QWORD *)(v4 - 32),
                v153.m128i_i64[0],
                v153.m128i_u32[2],
                (__int64)v151,
                3u);
        v88 = *(__int64 **)(v3 + 32);
        v89 = *(_QWORD *)(v2 + 8);
        v150 = 257;
        v122 = v87;
        v139 = v89;
        v90 = sub_AA4E30(v88[6]);
        v91 = sub_AE5020(v90, v139);
        v152 = 257;
        v126 = v91;
        v92 = sub_BD2C40(80, unk_3F10A14);
        v93 = v126;
        v94 = (__int64)v92;
        if ( v92 )
          sub_B4D190((__int64)v92, v139, v122, (__int64)v151, 0, v126, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64, __int64))(*(_QWORD *)v88[11] + 16LL))(
          v88[11],
          v94,
          v149,
          v88[7],
          v88[8],
          v93);
        v95 = *v88 + 16LL * *((unsigned int *)v88 + 2);
        v96 = *v88;
        if ( v96 != v95 )
        {
          v140 = v3;
          v99 = v94;
          v98 = v2;
          v97 = v99;
          v100 = v95;
          do
          {
            v101 = *(_QWORD *)(v96 + 8);
            v102 = *(_DWORD *)v96;
            v96 += 16;
            sub_B99FD0(v97, v102, v101);
          }
          while ( v100 != v96 );
          v3 = v140;
          v103 = v97;
          v2 = v98;
          v94 = v103;
        }
        sub_B91FC0(v151, v4);
        sub_B9A100(v94, v151);
        v41 = (__int64)sub_F162A0(v3, v2, v94);
        if ( (__m128i *)v153.m128i_i64[0] != &v154 )
          _libc_free(v153.m128i_i64[0], v2);
        return (unsigned __int8 *)v41;
      }
    }
  }
LABEL_72:
  v44 = *(_BYTE *)v4;
LABEL_48:
  if ( v44 == 84 )
  {
    v57 = sub_F27020(v3, v2, v4, 0, v42, v43);
    if ( v57 )
      return v57;
    v44 = *(_BYTE *)v4;
  }
  if ( v44 != 86 )
    return (unsigned __int8 *)v41;
  return sub_F26350(v3, (_BYTE *)v2, v4, 1);
}
