// Function: sub_32ABE50
// Address: 0x32abe50
//
__int64 __fastcall sub_32ABE50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int16 *v6; // rax
  unsigned __int16 v7; // bx
  __int64 v8; // r13
  __int64 result; // rax
  const __m128i *v10; // rax
  __int64 v11; // r14
  __int64 v12; // r12
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r10d
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, unsigned int); // rdx
  char (__fastcall *v23)(__int64, unsigned int); // rax
  const __m128i *v24; // roff
  __int64 *v25; // rcx
  __int64 v26; // rax
  __int16 v27; // dx
  __int64 v28; // rsi
  __int64 v29; // rax
  __int16 v30; // di
  __int64 v31; // rax
  __int64 v32; // r12
  int v33; // eax
  __int16 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  bool v37; // cc
  _QWORD *v38; // rax
  __int64 v39; // rbx
  __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r8
  __int64 v45; // r12
  __int64 v46; // rdx
  __int64 v47; // r13
  unsigned __int64 v48; // rdi
  __int64 v49; // rsi
  __int64 v50; // rbx
  int v51; // r9d
  __int64 v52; // r12
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // rsi
  __int64 v56; // r14
  char v57; // al
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  char v61; // al
  char v62; // al
  char v63; // al
  unsigned int v64; // eax
  __int64 v65; // r8
  __int64 v66; // r9
  unsigned __int16 v67; // r12
  __int64 v68; // rbx
  signed int v69; // r15d
  __int64 v70; // rdx
  int v71; // eax
  unsigned __int16 v72; // ax
  __int64 v73; // rdx
  bool v74; // al
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // rdx
  char v78; // al
  __int64 v79; // rbx
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  unsigned int v84; // eax
  __int64 v85; // rdx
  int v86; // edx
  int v87; // ebx
  int v88; // r8d
  __int64 v89; // r9
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // rax
  int v94; // edx
  __int64 v95; // rdx
  _QWORD *v96; // rax
  char v97; // al
  signed int v98; // r10d
  __int64 *v99; // rbx
  __int64 v100; // rdx
  int v101; // r9d
  int v102; // eax
  __int64 v103; // r15
  int v104; // edx
  int v105; // r13d
  int v106; // r12d
  __int64 v107; // rax
  __int64 v108; // rdx
  __int128 v109; // [rsp-30h] [rbp-150h]
  __int128 v110; // [rsp-20h] [rbp-140h]
  __int128 v111; // [rsp-20h] [rbp-140h]
  __int128 v112; // [rsp-10h] [rbp-130h]
  __int64 v113; // [rsp+0h] [rbp-120h]
  unsigned int v114; // [rsp+0h] [rbp-120h]
  __int64 v115; // [rsp+8h] [rbp-118h]
  __int64 v116; // [rsp+10h] [rbp-110h]
  __int64 v117; // [rsp+18h] [rbp-108h]
  __int64 v118; // [rsp+18h] [rbp-108h]
  __int64 v119; // [rsp+20h] [rbp-100h]
  __int64 *v120; // [rsp+20h] [rbp-100h]
  __int64 *v121; // [rsp+28h] [rbp-F8h]
  __int64 v122; // [rsp+28h] [rbp-F8h]
  unsigned int v123; // [rsp+30h] [rbp-F0h]
  __int64 v124; // [rsp+30h] [rbp-F0h]
  __int64 v125; // [rsp+30h] [rbp-F0h]
  __int16 v126; // [rsp+30h] [rbp-F0h]
  int v127; // [rsp+38h] [rbp-E8h]
  __int64 v128; // [rsp+38h] [rbp-E8h]
  int v129; // [rsp+38h] [rbp-E8h]
  signed int v130; // [rsp+38h] [rbp-E8h]
  __int64 v131; // [rsp+38h] [rbp-E8h]
  __int64 v132; // [rsp+38h] [rbp-E8h]
  __int64 v133; // [rsp+40h] [rbp-E0h]
  int v134; // [rsp+40h] [rbp-E0h]
  unsigned int v135; // [rsp+40h] [rbp-E0h]
  int v136; // [rsp+40h] [rbp-E0h]
  int v137; // [rsp+40h] [rbp-E0h]
  __int128 v138; // [rsp+40h] [rbp-E0h]
  __int128 v139; // [rsp+50h] [rbp-D0h]
  unsigned int v140; // [rsp+50h] [rbp-D0h]
  __int64 v142; // [rsp+50h] [rbp-D0h]
  int v143; // [rsp+50h] [rbp-D0h]
  __int64 v144; // [rsp+50h] [rbp-D0h]
  signed int v145; // [rsp+50h] [rbp-D0h]
  __int64 v146; // [rsp+50h] [rbp-D0h]
  __int64 v148; // [rsp+60h] [rbp-C0h]
  int v149; // [rsp+60h] [rbp-C0h]
  __int64 v150; // [rsp+60h] [rbp-C0h]
  __int64 v151; // [rsp+68h] [rbp-B8h]
  __int64 v152; // [rsp+78h] [rbp-A8h]
  __int64 v153; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v154; // [rsp+88h] [rbp-98h]
  __int64 v155; // [rsp+90h] [rbp-90h] BYREF
  __int64 v156; // [rsp+98h] [rbp-88h]
  __m128i v157; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v158; // [rsp+B0h] [rbp-70h]
  __int64 v159; // [rsp+B8h] [rbp-68h]
  _BYTE *v160; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v161; // [rsp+C8h] [rbp-58h]
  _BYTE v162[80]; // [rsp+D0h] [rbp-50h] BYREF

  v5 = a2;
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v153) = v7;
  v154 = v8;
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0x9Eu )
      return 0;
    v8 = 0;
    v24 = *(const __m128i **)(a2 + 40);
    v11 = v24->m128i_i64[0];
    v12 = v24->m128i_u32[2];
    v16 = *(_DWORD *)(v24->m128i_i64[0] + 24);
    v139 = (__int128)_mm_loadu_si128(v24);
    v7 = word_4456580[v7 - 1];
  }
  else
  {
    if ( !sub_30070D0((__int64)&v153) )
      return 0;
    v10 = *(const __m128i **)(a2 + 40);
    v11 = v10->m128i_i64[0];
    v12 = v10->m128i_u32[2];
    v139 = (__int128)_mm_loadu_si128(v10);
    v127 = *(_DWORD *)(v10->m128i_i64[0] + 24);
    v13 = sub_30070B0((__int64)&v153);
    v16 = v127;
    if ( v13 )
    {
      v72 = sub_3009970((__int64)&v153, a2, v14, v15, a5);
      v16 = v127;
      v7 = v72;
      v8 = v73;
    }
  }
  v17 = *(_QWORD *)(v11 + 56);
  v18 = 1;
  v19 = v7;
  if ( !v17 )
    goto LABEL_30;
  do
  {
    while ( *(_DWORD *)(v17 + 8) != (_DWORD)v12 )
    {
      v17 = *(_QWORD *)(v17 + 32);
      if ( !v17 )
        goto LABEL_16;
    }
    if ( !v18 )
      goto LABEL_30;
    v20 = *(_QWORD *)(v17 + 32);
    if ( !v20 )
      goto LABEL_17;
    if ( (_DWORD)v12 == *(_DWORD *)(v20 + 8) )
      goto LABEL_30;
    v17 = *(_QWORD *)(v20 + 32);
    v18 = 0;
  }
  while ( v17 );
LABEL_16:
  if ( v18 == 1 )
    goto LABEL_30;
LABEL_17:
  if ( *(_DWORD *)(v11 + 68) != 1 )
    goto LABEL_30;
  v21 = *(_QWORD *)(a1 + 8);
  v22 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v21 + 1368LL);
  if ( v22 != sub_2FE4300 )
  {
    v134 = v16;
    v57 = v22(v21, v16);
    v16 = v134;
    v19 = v7;
    if ( !v57 )
      goto LABEL_30;
    goto LABEL_78;
  }
  v23 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v21 + 1360LL);
  if ( v23 == sub_2FE3400 )
  {
    if ( v16 <= 98 )
    {
      if ( v16 > 55 )
      {
        switch ( v16 )
        {
          case '8':
          case ':':
          case '?':
          case '@':
          case 'D':
          case 'F':
          case 'L':
          case 'M':
          case 'R':
          case 'S':
          case '`':
          case 'b':
            goto LABEL_78;
          default:
            break;
        }
      }
LABEL_22:
      if ( v16 > 56 )
      {
LABEL_23:
        switch ( v16 )
        {
          case '9':
          case ';':
          case '<':
          case '=':
          case '>':
          case 'T':
          case 'U':
          case 'a':
          case 'c':
          case 'd':
            goto LABEL_78;
          default:
            goto LABEL_30;
        }
      }
      goto LABEL_30;
    }
    if ( v16 > 188 )
    {
      if ( (unsigned int)(v16 - 279) > 7 )
        goto LABEL_29;
    }
    else if ( v16 <= 185 && (unsigned int)(v16 - 172) > 0xB )
    {
      if ( v16 <= 100 )
        goto LABEL_23;
LABEL_29:
      if ( (unsigned int)(v16 - 190) > 4 )
        goto LABEL_30;
    }
  }
  else
  {
    v137 = v16;
    v78 = v23(v21, v16);
    v16 = v137;
    v19 = v7;
    if ( !v78 )
    {
      if ( v137 <= 100 )
        goto LABEL_22;
      goto LABEL_29;
    }
  }
LABEL_78:
  v58 = *(_QWORD *)(v11 + 48) + 16LL * (unsigned int)v12;
  if ( *(_WORD *)v58 == v7 && (*(_QWORD *)(v58 + 8) == v8 || v7) )
  {
    v25 = *(__int64 **)(v11 + 40);
    v128 = *((unsigned int *)v25 + 2);
    v59 = *(_QWORD *)(*v25 + 48) + 16 * v128;
    v133 = *v25;
    if ( *(_WORD *)v59 == v7 )
    {
      v28 = *(_QWORD *)(v59 + 8);
      if ( v8 != v28 && !v7 )
        goto LABEL_60;
      v60 = *(_QWORD *)(v25[5] + 48) + 16LL * *((unsigned int *)v25 + 12);
      if ( *(_WORD *)v60 != v7 )
        goto LABEL_30;
      if ( *(_QWORD *)(v60 + 8) != v8 && !v7 )
      {
LABEL_60:
        if ( v16 != 158 )
          return 0;
        LOWORD(v160) = 0;
        v161 = v28;
LABEL_62:
        v119 = v19;
        v121 = v25;
        if ( !sub_30070D0((__int64)&v160) )
          return 0;
        v19 = v119;
        v25 = v121;
        v27 = 0;
        goto LABEL_33;
      }
      v124 = v19;
      v130 = v16;
      v61 = sub_33CF8D0(v11, v133);
      v16 = v130;
      v19 = v124;
      if ( v61 )
      {
        v62 = sub_33CF8D0(v11, *(_QWORD *)(*(_QWORD *)(v11 + 40) + 40LL));
        v16 = v130;
        v19 = v124;
        if ( v62 )
        {
          if ( v130 > 62 )
          {
            if ( (unsigned int)(v130 - 65) <= 1 )
              goto LABEL_30;
          }
          else if ( v130 > 58 )
          {
            goto LABEL_30;
          }
          v63 = sub_328A020(*(_QWORD *)(a1 + 8), v130, v153, v154, *(unsigned __int8 *)(a1 + 33));
          v16 = v130;
          v19 = v124;
          if ( v63 )
          {
            v64 = sub_3281500(&v153, (unsigned int)v130);
            v160 = v162;
            v161 = 0x800000000LL;
            sub_11B1960((__int64)&v160, v64, -1, 0x800000000LL, v65, v66);
            v19 = v124;
            v114 = v12;
            v67 = v7;
            v152 = 0x100000000LL;
            v68 = 0;
            v125 = v5;
            v69 = v130;
            while ( 1 )
            {
              v70 = *(_QWORD *)(v11 + 40);
              v135 = *(_DWORD *)((char *)&v152 + v68);
              v122 = *(_QWORD *)(v70 + 40LL * (v135 == 0));
              v71 = *(_DWORD *)(v122 + 24);
              if ( v71 == 11 || v71 == 35 )
              {
                v90 = *(_QWORD *)(v70 + 40LL * v135);
                v132 = v90;
                if ( *(_DWORD *)(v90 + 24) == 158 )
                {
                  v91 = *(_QWORD *)(v90 + 40);
                  v92 = *(_QWORD *)(*(_QWORD *)v91 + 48LL) + 16LL * *(unsigned int *)(v91 + 8);
                  if ( (_WORD)v153 == *(_WORD *)v92 && (v154 == *(_QWORD *)(v92 + 8) || *(_WORD *)v92) )
                  {
                    v93 = *(_QWORD *)(v91 + 40);
                    v94 = *(_DWORD *)(v93 + 24);
                    if ( v94 == 11 || v94 == 35 )
                    {
                      v95 = *(_QWORD *)(v93 + 96);
                      v96 = *(_QWORD **)(v95 + 24);
                      if ( *(_DWORD *)(v95 + 32) > 0x40u )
                        v96 = (_QWORD *)*v96;
                      v116 = v19;
                      *(_DWORD *)v160 = (_DWORD)v96;
                      v97 = (*(__int64 (__fastcall **)(_QWORD, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8)
                                                                                                + 624LL))(
                              *(_QWORD *)(a1 + 8),
                              v160,
                              (unsigned int)v161,
                              (unsigned int)v153,
                              v154);
                      v19 = v116;
                      if ( v97 )
                        break;
                    }
                  }
                }
              }
              v68 += 4;
              if ( v68 == 8 )
              {
                v7 = v67;
                v16 = v69;
                v12 = v114;
                v5 = v125;
                if ( v160 != v162 )
                {
                  v131 = v19;
                  v136 = v16;
                  _libc_free((unsigned __int64)v160);
                  v16 = v136;
                  v19 = v131;
                }
                goto LABEL_30;
              }
            }
            v98 = v69;
            v155 = *(_QWORD *)(v125 + 80);
            if ( v155 )
            {
              sub_325F5D0(&v155);
              v98 = v69;
            }
            v145 = v98;
            LODWORD(v156) = *(_DWORD *)(v125 + 72);
            v157 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v132 + 40));
            v99 = (__int64 *)a1;
            v158 = sub_34007B0(*(_QWORD *)a1, *(_DWORD *)(v122 + 96) + 24, (unsigned int)&v155, v153, v154, 0, 0);
            v159 = v100;
            *((_QWORD *)&v111 + 1) = v157.m128i_i64[2 * (int)(1 - v135) + 1];
            *(_QWORD *)&v111 = v157.m128i_i64[2 * (int)(1 - v135)];
            *((_QWORD *)&v109 + 1) = v157.m128i_i64[2 * (int)v135 + 1];
            *(_QWORD *)&v109 = v157.m128i_i64[2 * (int)v135];
            v102 = sub_3406EB0(*(_QWORD *)a1, v145, (unsigned int)&v155, v153, v154, v101, v109, v111);
            v103 = *(_QWORD *)a1;
            v105 = v104;
            v106 = v102;
            v150 = (__int64)v160;
            v151 = (unsigned int)v161;
            v107 = sub_3288990(*v99, (unsigned int)v153, v154);
            v146 = sub_33FCE10(v103, v153, v154, (unsigned int)&v155, v106, v105, v107, v108, v150, v151);
            sub_9C6650(&v155);
            v48 = (unsigned __int64)v160;
            result = v146;
            if ( v160 != v162 )
              goto LABEL_59;
            return result;
          }
        }
      }
    }
  }
LABEL_30:
  if ( v16 != 158 )
    return 0;
  v25 = *(__int64 **)(v11 + 40);
  v128 = *((unsigned int *)v25 + 2);
  v26 = *(_QWORD *)(*v25 + 48) + 16 * v128;
  v27 = *(_WORD *)v26;
  v133 = *v25;
  v28 = *(_QWORD *)(v26 + 8);
  LOWORD(v160) = v27;
  v161 = v28;
  if ( !v27 )
    goto LABEL_62;
  if ( (unsigned __int16)(v27 - 17) > 0x9Eu )
    return 0;
LABEL_33:
  v29 = *(_QWORD *)(v11 + 48) + 16 * v12;
  v30 = *(_WORD *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  if ( v30 == v7 )
  {
    if ( v7 || v8 == v31 )
      goto LABEL_35;
    v161 = v31;
    LOWORD(v160) = 0;
  }
  else
  {
    LOWORD(v160) = v30;
    v161 = v31;
    if ( v30 )
    {
      if ( (unsigned __int16)(v30 - 2) > 7u )
        goto LABEL_35;
      goto LABEL_66;
    }
  }
  v118 = v19;
  v120 = v25;
  v126 = v27;
  v74 = sub_30070A0((__int64)&v160);
  v27 = v126;
  v25 = v120;
  v19 = v118;
  if ( !v74 )
    goto LABEL_35;
LABEL_66:
  if ( !*(_BYTE *)(a1 + 34) || v7 && *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * v7 + 112) )
  {
    v49 = *(_QWORD *)(v11 + 80);
    *(_QWORD *)&v139 = v11;
    v50 = *(_QWORD *)a1;
    v160 = (_BYTE *)v49;
    *((_QWORD *)&v139 + 1) = v12 | *((_QWORD *)&v139 + 1) & 0xFFFFFFFF00000000LL;
    if ( v49 )
    {
      v129 = v19;
      sub_B96E90((__int64)&v160, v49, 1);
      LODWORD(v19) = v129;
    }
    LODWORD(v161) = *(_DWORD *)(v11 + 72);
    v52 = sub_33FAF80(v50, 216, (unsigned int)&v160, v19, v8, v19, v139);
    v54 = v53;
    if ( v160 )
      sub_B91220((__int64)&v160, (__int64)v160);
    v55 = *(_QWORD *)(v5 + 80);
    v56 = *(_QWORD *)a1;
    v160 = (_BYTE *)v55;
    if ( v55 )
      sub_B96E90((__int64)&v160, v55, 1);
    *((_QWORD *)&v112 + 1) = v54;
    *(_QWORD *)&v112 = v52;
    LODWORD(v161) = *(_DWORD *)(v5 + 72);
    result = sub_33FAF80(v56, 167, (unsigned int)&v160, v153, v154, v51, v112);
    if ( v160 )
    {
      v142 = result;
      sub_B91220((__int64)&v160, (__int64)v160);
      return v142;
    }
    return result;
  }
LABEL_35:
  v32 = v25[5];
  v33 = *(_DWORD *)(v32 + 24);
  if ( v33 != 35 && v33 != 11 )
    return 0;
  LOWORD(v155) = v27;
  v156 = v28;
  if ( v27 )
    goto LABEL_38;
  if ( sub_3007100((__int64)&v155) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v155 )
    {
      if ( (unsigned __int16)(v155 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_38:
      v123 = word_4456340[(unsigned __int16)v155 - 1];
      goto LABEL_39;
    }
  }
  v123 = sub_3007130((__int64)&v155, v28);
LABEL_39:
  if ( (_WORD)v153 )
  {
    if ( (unsigned __int16)(v153 - 176) > 0x34u )
    {
LABEL_41:
      v140 = word_4456340[(unsigned __int16)v153 - 1];
      goto LABEL_42;
    }
  }
  else if ( !sub_3007100((__int64)&v153) )
  {
    goto LABEL_106;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v153 )
  {
    if ( (unsigned __int16)(v153 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_41;
  }
LABEL_106:
  v140 = sub_3007130((__int64)&v153, v28);
LABEL_42:
  v34 = v155;
  if ( (_WORD)v155 )
  {
    if ( (unsigned __int16)(v155 - 17) <= 0xD3u )
    {
      v34 = word_4456580[(unsigned __int16)v155 - 1];
      v35 = 0;
      goto LABEL_45;
    }
  }
  else if ( sub_30070B0((__int64)&v155) )
  {
    v34 = sub_3009970((__int64)&v155, v28, v75, v76, a5);
    v35 = v77;
    goto LABEL_45;
  }
  v35 = v156;
LABEL_45:
  if ( v7 != v34 || !v7 && v8 != v35 || v140 > v123 )
    return 0;
  v160 = v162;
  v161 = 0x800000000LL;
  sub_11B1960((__int64)&v160, v123, -1, v123, a5, v19);
  v36 = *(_QWORD *)(v32 + 96);
  v37 = *(_DWORD *)(v36 + 32) <= 0x40u;
  v38 = *(_QWORD **)(v36 + 24);
  if ( !v37 )
    v38 = (_QWORD *)*v38;
  *(_DWORD *)v160 = (_DWORD)v38;
  v39 = *(_QWORD *)a1;
  v40 = (__int64)v160;
  v41 = (unsigned int)v161;
  v117 = *(_QWORD *)(a1 + 8);
  v42 = sub_3288990(*(_QWORD *)a1, (unsigned int)v155, v156);
  v157.m128i_i64[0] = *(_QWORD *)(v5 + 80);
  if ( v157.m128i_i64[0] )
  {
    v113 = v42;
    v115 = v43;
    sub_325F5D0(v157.m128i_i64);
    v42 = v113;
    v43 = v115;
  }
  v157.m128i_i32[2] = *(_DWORD *)(v5 + 72);
  v45 = sub_3449A00(v117, v155, v156, (unsigned int)&v157, v133, v128, v42, v43, v40, v41, v39);
  v47 = v46;
  if ( v157.m128i_i64[0] )
    sub_B91220((__int64)&v157, v157.m128i_i64[0]);
  result = 0;
  if ( v45 )
  {
    if ( (_WORD)v153 == (_WORD)v155 && ((_WORD)v153 || v156 == v154) )
    {
      result = v45;
    }
    else
    {
      if ( v140 == v123 )
      {
        if ( v160 != v162 )
          _libc_free((unsigned __int64)v160);
        return 0;
      }
      v79 = *(_QWORD *)a1;
      v157.m128i_i64[0] = *(_QWORD *)(v5 + 80);
      if ( v157.m128i_i64[0] )
        sub_325F5D0(v157.m128i_i64);
      v157.m128i_i32[2] = *(_DWORD *)(v5 + 72);
      *(_QWORD *)&v138 = sub_3400EE0(v79, 0, &v157, 0, v44);
      *((_QWORD *)&v138 + 1) = v80;
      sub_9C6650(&v157);
      v84 = sub_3281170(&v155, 0, v81, v82, v83);
      v87 = sub_327FCF0(*(__int64 **)(*(_QWORD *)a1 + 64LL), v84, v85, v140, 0);
      v88 = v86;
      v89 = *(_QWORD *)a1;
      v157.m128i_i64[0] = *(_QWORD *)(v5 + 80);
      if ( v157.m128i_i64[0] )
      {
        v149 = v89;
        v143 = v86;
        sub_325F5D0(v157.m128i_i64);
        v88 = v143;
        LODWORD(v89) = v149;
      }
      *((_QWORD *)&v110 + 1) = v47;
      *(_QWORD *)&v110 = v45;
      v157.m128i_i32[2] = *(_DWORD *)(v5 + 72);
      v144 = sub_3406EB0(v89, 161, (unsigned int)&v157, v87, v88, v89, v110, v138);
      sub_9C6650(&v157);
      result = v144;
    }
  }
  v48 = (unsigned __int64)v160;
  if ( v160 != v162 )
  {
LABEL_59:
    v148 = result;
    _libc_free(v48);
    return v148;
  }
  return result;
}
