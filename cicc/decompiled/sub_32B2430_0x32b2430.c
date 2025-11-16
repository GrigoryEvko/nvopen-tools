// Function: sub_32B2430
// Address: 0x32b2430
//
unsigned int *__fastcall sub_32B2430(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int16 *v8; // rdx
  __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 *v13; // rsi
  __m128i v14; // xmm1
  __int64 v15; // rax
  unsigned int *result; // rax
  __int64 v17; // r15
  int v18; // eax
  __int64 v19; // rdx
  unsigned __int16 *v20; // rax
  unsigned __int16 v21; // r13
  __int64 v22; // rax
  int v23; // eax
  unsigned int *v24; // r14
  __int64 v25; // r8
  __int64 v26; // rax
  __int32 v27; // edx
  __int64 v28; // rcx
  __int64 v29; // r8
  __m128i *v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __m128i v38; // xmm0
  unsigned __int64 v39; // rdx
  unsigned __int16 v40; // r13
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned __int16 *v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rdx
  int v55; // esi
  __int64 v56; // rdx
  __int64 v57; // r13
  unsigned int *v58; // rbx
  __int64 v59; // rax
  __int16 v60; // r13
  __int64 v61; // r15
  unsigned int *v62; // r8
  unsigned __int64 v63; // rcx
  __int64 v64; // rsi
  unsigned int v65; // r14d
  unsigned int *v66; // r13
  __int64 (*v67)(); // r9
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rdi
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rdx
  bool v80; // al
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  bool v84; // al
  bool v85; // al
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int16 v89; // ax
  __int64 v90; // rdx
  char v91; // r13
  __int64 v92; // r13
  int v93; // edx
  int v94; // r14d
  int v95; // edx
  char v96; // al
  __int128 v97; // [rsp-10h] [rbp-320h]
  __int16 v98; // [rsp+Ah] [rbp-306h]
  __int16 v99; // [rsp+12h] [rbp-2FEh]
  __int64 v100; // [rsp+18h] [rbp-2F8h]
  bool v101; // [rsp+2Fh] [rbp-2E1h]
  __int64 v102; // [rsp+38h] [rbp-2D8h]
  unsigned int *v103; // [rsp+50h] [rbp-2C0h]
  __int64 v104; // [rsp+50h] [rbp-2C0h]
  __int32 v105; // [rsp+58h] [rbp-2B8h]
  __int64 v106; // [rsp+58h] [rbp-2B8h]
  __int64 v107; // [rsp+58h] [rbp-2B8h]
  __m128i v108; // [rsp+60h] [rbp-2B0h] BYREF
  unsigned __int64 v109; // [rsp+70h] [rbp-2A0h]
  unsigned int *v110; // [rsp+78h] [rbp-298h]
  __int64 v111; // [rsp+80h] [rbp-290h]
  __int64 v112; // [rsp+88h] [rbp-288h]
  __int64 v113; // [rsp+90h] [rbp-280h]
  __int64 v114; // [rsp+98h] [rbp-278h]
  __int64 v115; // [rsp+A0h] [rbp-270h]
  __int64 v116; // [rsp+A8h] [rbp-268h]
  __int64 v117; // [rsp+B0h] [rbp-260h] BYREF
  __int64 v118; // [rsp+B8h] [rbp-258h]
  _QWORD v119[2]; // [rsp+C0h] [rbp-250h] BYREF
  __m128i v120; // [rsp+D0h] [rbp-240h] BYREF
  __m128i v121; // [rsp+E0h] [rbp-230h] BYREF
  unsigned __int64 *v122; // [rsp+F0h] [rbp-220h] BYREF
  __int64 v123; // [rsp+F8h] [rbp-218h]
  char v124; // [rsp+100h] [rbp-210h]
  unsigned int *v125; // [rsp+110h] [rbp-200h] BYREF
  __int64 v126; // [rsp+118h] [rbp-1F8h]
  _BYTE v127[128]; // [rsp+120h] [rbp-1F0h] BYREF
  unsigned __int64 v128[2]; // [rsp+1A0h] [rbp-170h] BYREF
  _BYTE v129[264]; // [rsp+1B0h] [rbp-160h] BYREF
  int v130; // [rsp+2B8h] [rbp-58h] BYREF
  unsigned __int64 v131; // [rsp+2C0h] [rbp-50h]
  int *v132; // [rsp+2C8h] [rbp-48h]
  int *v133; // [rsp+2D0h] [rbp-40h]
  __int64 v134; // [rsp+2D8h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  v109 = a3;
  v8 = *(__int16 **)(a1 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v117) = v9;
  v118 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 176) > 0x34u )
    {
LABEL_3:
      LODWORD(v110) = word_4456340[(unsigned __int16)v117 - 1];
      goto LABEL_6;
    }
  }
  else if ( !sub_3007100((__int64)&v117) )
  {
    goto LABEL_5;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v117 )
  {
    if ( (unsigned __int16)(v117 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_3;
  }
LABEL_5:
  LODWORD(v110) = sub_3007130((__int64)&v117, a2);
LABEL_6:
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_QWORD *)v11;
  v13 = *(unsigned __int64 **)(v11 + 8);
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v15 = *(_QWORD *)(*(_QWORD *)v11 + 56LL);
  v119[0] = v12;
  v119[1] = v13;
  v120 = v14;
  if ( !v15 )
    return 0;
  v102 = *(_QWORD *)(v15 + 32);
  if ( v102 )
    return 0;
  v17 = v120.m128i_i64[0];
  v18 = *(_DWORD *)(v120.m128i_i64[0] + 24);
  if ( v18 == 51 )
  {
LABEL_17:
    v101 = 0;
    goto LABEL_18;
  }
  v19 = *(_QWORD *)(v120.m128i_i64[0] + 56);
  if ( !v19 || *(_QWORD *)(v19 + 32) )
    return 0;
  if ( *(_DWORD *)(v12 + 24) == 156 )
  {
    v91 = sub_326A930(v12, (unsigned int)v13, 0);
    if ( v91 )
    {
      v13 = (unsigned __int64 *)v120.m128i_i64[1];
      if ( *(_DWORD *)(v17 + 24) != 156 )
        goto LABEL_113;
      if ( (unsigned __int8)sub_326A930(v17, v120.m128i_u32[2], 0) || (unsigned __int8)sub_33CA720(v17) )
        goto LABEL_114;
    }
    else
    {
      v96 = sub_33CA720(v12);
      v13 = (unsigned __int64 *)v120.m128i_i64[1];
      v91 = v96;
      if ( *(_DWORD *)(v17 + 24) == 156
        && ((unsigned __int8)sub_326A930(v17, v120.m128i_u32[2], 0) || (unsigned __int8)sub_33CA720(v17)) )
      {
        if ( v91 )
          goto LABEL_114;
LABEL_120:
        if ( (unsigned __int8)sub_33D1E40(v17, v13) )
          goto LABEL_114;
        return 0;
      }
    }
    if ( !v91 )
      goto LABEL_114;
LABEL_113:
    if ( (unsigned __int8)sub_33D1E40(v12, v13) )
      goto LABEL_114;
    return 0;
  }
  if ( v18 != 156 )
    goto LABEL_17;
  v13 = (unsigned __int64 *)v120.m128i_i64[1];
  if ( (unsigned __int8)sub_326A930(v120.m128i_i64[0], v120.m128i_u32[2], 0) || (unsigned __int8)sub_33CA720(v17) )
    goto LABEL_120;
LABEL_114:
  if ( *(_DWORD *)(v12 + 24) != 156 || *(_DWORD *)(v17 + 24) != 156 )
    goto LABEL_17;
  v13 = 0;
  v101 = 0;
  v92 = sub_33D2250(v12, 0);
  v94 = v93;
  if ( v92 )
  {
    v13 = 0;
    if ( v92 == sub_33D2250(v17, 0) )
      v101 = v94 == v95;
  }
LABEL_18:
  v130 = 0;
  v125 = (unsigned int *)v127;
  v126 = 0x800000000LL;
  v128[0] = (unsigned __int64)v129;
  v128[1] = 0x1000000000LL;
  v132 = &v130;
  v133 = &v130;
  v20 = *(unsigned __int16 **)(v7 + 48);
  v131 = 0;
  v134 = 0;
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  LOWORD(v122) = v21;
  v123 = v22;
  if ( v21 )
  {
    if ( (unsigned __int16)(v21 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v23 = word_4456340[v21 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v122) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v23 = sub_3007130((__int64)&v122, (__int64)v13);
  }
  v24 = *(unsigned int **)(v7 + 96);
  v103 = &v24[v23];
  if ( v103 != v24 )
  {
    v100 = v7;
    while ( 1 )
    {
      v40 = v117;
      v31 = *v24;
      if ( (_WORD)v117 )
      {
        if ( (unsigned __int16)(v117 - 17) <= 0xD3u )
        {
          v25 = 0;
          v40 = word_4456580[(unsigned __int16)v117 - 1];
          goto LABEL_26;
        }
      }
      else if ( sub_30070B0((__int64)&v117) )
      {
        v40 = sub_3009970((__int64)&v117, (__int64)v13, v41, v42, v43);
        v25 = v44;
        goto LABEL_26;
      }
      v25 = v118;
LABEL_26:
      v122 = 0;
      LODWORD(v123) = 0;
      v26 = sub_33F17F0(v6, 51, &v122, v40, v25);
      v13 = v122;
      if ( v122 )
      {
        v105 = v27;
        v108.m128i_i64[0] = v26;
        sub_B91220((__int64)&v122, (__int64)v122);
        v27 = v105;
        v26 = v108.m128i_i64[0];
      }
      v121.m128i_i64[0] = v26;
      v121.m128i_i32[2] = v27;
      if ( (int)v31 >= 0 )
      {
        v28 = (unsigned int)v110;
        v30 = (__m128i *)v119;
        if ( (int)v110 <= (int)v31 )
        {
          v31 = (unsigned int)(v31 - (_DWORD)v110);
          v30 = &v120;
        }
        v32 = v30->m128i_i64[0];
        v33 = *(_DWORD *)(v32 + 24);
        if ( v33 == 156 )
        {
          v29 = *(_QWORD *)(v32 + 40);
          v46 = v29 + 40 * v31;
          v26 = *(_QWORD *)v46;
          v121.m128i_i64[0] = *(_QWORD *)v46;
          v121.m128i_i32[2] = *(_DWORD *)(v46 + 8);
        }
        else
        {
          if ( v33 != 167 )
            goto LABEL_53;
          v34 = *(_QWORD *)(v32 + 40);
          v26 = *(_QWORD *)v34;
          v35 = *(unsigned int *)(v34 + 8);
          if ( (_DWORD)v31 )
          {
            v47 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16 * v35);
            v48 = *((_QWORD *)v47 + 1);
            v49 = *v47;
            v122 = 0;
            LODWORD(v123) = 0;
            v50 = sub_33F17F0(v6, 51, &v122, v49, v48);
            v13 = v122;
            v51 = v50;
            if ( v122 )
            {
              v108.m128i_i64[0] = v35;
              sub_B91220((__int64)&v122, (__int64)v122);
              LODWORD(v35) = v108.m128i_i32[0];
            }
            v26 = v51;
          }
          v121.m128i_i64[0] = v26;
          v121.m128i_i32[2] = v35;
        }
      }
      v36 = *(unsigned int *)(v26 + 24);
      if ( (unsigned int)v36 > 0x33 || (v45 = 0x8001800001800LL, !_bittest64(&v45, v36)) )
      {
        if ( !v101 )
        {
          v13 = v128;
          sub_32B20F0((__int64)&v122, (__int64)v128, &v121, v28, v29, a6);
          if ( !v124 )
          {
LABEL_53:
            v52 = 0;
            v53 = 0;
            goto LABEL_54;
          }
        }
      }
      v37 = (unsigned int)v126;
      v38 = _mm_load_si128(&v121);
      v39 = (unsigned int)v126 + 1LL;
      if ( v39 > HIDWORD(v126) )
      {
        v13 = (unsigned __int64 *)v127;
        v108 = v38;
        sub_C8D5F0((__int64)&v125, v127, v39, 0x10u, v29, a6);
        v37 = (unsigned int)v126;
        v38 = _mm_load_si128(&v108);
      }
      ++v24;
      *(__m128i *)&v125[4 * v37] = v38;
      LODWORD(v126) = v126 + 1;
      if ( v103 == v24 )
      {
        v7 = v100;
        break;
      }
    }
  }
  v55 = (unsigned __int16)v117;
  if ( (_WORD)v117 )
  {
    if ( (unsigned __int16)(v117 - 17) > 0xD3u )
    {
LABEL_61:
      v108.m128i_i16[0] = v55;
      v56 = v118;
      goto LABEL_62;
    }
    v108.m128i_i16[0] = v117;
    v56 = 0;
    v55 = (unsigned __int16)word_4456580[(unsigned __int16)v117 - 1];
  }
  else
  {
    LODWORD(v110) = 0;
    v80 = sub_30070B0((__int64)&v117);
    v55 = (int)v110;
    if ( !v80 )
      goto LABEL_61;
    v55 = sub_3009970((__int64)&v117, (unsigned int)v110, v81, v82, v83);
    v108.m128i_i16[0] = v117;
  }
LABEL_62:
  v121.m128i_i16[0] = v55;
  v121.m128i_i64[1] = v56;
  if ( (_WORD)v55 )
  {
    if ( (unsigned __int16)(v55 - 2) > 7u
      && (unsigned __int16)(v55 - 17) > 0x6Cu
      && (unsigned __int16)(v55 - 176) > 0x1Fu )
    {
      goto LABEL_72;
    }
  }
  else
  {
    LODWORD(v110) = v55;
    v84 = sub_3007070((__int64)&v121);
    v55 = (int)v110;
    if ( !v84 )
      goto LABEL_72;
  }
  v57 = 4LL * (unsigned int)v126;
  v110 = &v125[v57];
  if ( &v125[v57] != v125 )
  {
    v104 = v6;
    WORD1(v6) = v98;
    v106 = v7;
    v58 = v125;
    do
    {
      v59 = *(_QWORD *)(*(_QWORD *)v58 + 48LL) + 16LL * v58[2];
      v60 = *(_WORD *)v59;
      v61 = *(_QWORD *)(v59 + 8);
      LOWORD(v6) = *(_WORD *)v59;
      if ( sub_3280B30((__int64)&v121, (unsigned int)v6, v61) )
      {
        v121.m128i_i16[0] = v60;
        v121.m128i_i64[1] = v61;
      }
      v58 += 4;
    }
    while ( v110 != v58 );
    v7 = v106;
    v6 = v104;
    v55 = v121.m128i_u16[0];
  }
LABEL_72:
  if ( v108.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v108.m128i_i16[0] - 17) <= 0xD3u )
    {
      v108.m128i_i16[0] = word_4456580[v108.m128i_u16[0] - 1];
      goto LABEL_75;
    }
  }
  else
  {
    LODWORD(v110) = v55;
    v85 = sub_30070B0((__int64)&v117);
    LOWORD(v55) = (_WORD)v110;
    if ( v85 )
    {
      v89 = sub_3009970((__int64)&v117, (unsigned int)v110, v86, v87, v88);
      LOWORD(v55) = (_WORD)v110;
      v108.m128i_i16[0] = v89;
      v102 = v90;
      goto LABEL_75;
    }
  }
  v102 = v118;
LABEL_75:
  v62 = (unsigned int *)(unsigned int)v126;
  v63 = (unsigned __int64)v125;
  if ( ((_WORD)v55 != v108.m128i_i16[0] || !(_WORD)v55 && v121.m128i_i64[1] != v102)
    && &v125[4 * (unsigned int)v126] != v125 )
  {
    v110 = &v125[4 * (unsigned int)v126];
    HIWORD(v65) = v99;
    v66 = v125;
    while ( 1 )
    {
      if ( *(_DWORD *)(*(_QWORD *)v66 + 24LL) == 51 )
      {
        v122 = 0;
        LODWORD(v123) = 0;
        v73 = sub_33F17F0(v6, 51, &v122, v121.m128i_i64[0], v121.m128i_i64[1]);
        v75 = v74;
        v76 = v73;
        if ( v122 )
        {
          v107 = v74;
          v108.m128i_i64[0] = v73;
          sub_B91220((__int64)&v122, (__int64)v122);
          v75 = v107;
          v76 = v108.m128i_i64[0];
        }
        v116 = v75;
        v115 = v76;
        *(_QWORD *)v66 = v76;
        v66[2] = v116;
      }
      else
      {
        v67 = *(__int64 (**)())(*(_QWORD *)v109 + 1432LL);
        v68 = *(_QWORD *)(*(_QWORD *)v66 + 48LL) + 16LL * v66[2];
        if ( v67 == sub_2FE34A0
          || (LOWORD(v65) = *(_WORD *)v68,
              !((unsigned __int8 (__fastcall *)(unsigned __int64, _QWORD, _QWORD, _QWORD, __int64))v67)(
                 v109,
                 v65,
                 *(_QWORD *)(v68 + 8),
                 v121.m128i_u32[0],
                 v121.m128i_i64[1])) )
        {
          v69 = *(_QWORD *)(v7 + 80);
          v122 = (unsigned __int64 *)v69;
          if ( v69 )
            sub_B96E90((__int64)&v122, v69, 1);
          LODWORD(v123) = *(_DWORD *)(v7 + 72);
          v70 = sub_33FB160(v6, *(_QWORD *)v66, *((_QWORD *)v66 + 1), &v122, v121.m128i_u32[0], v121.m128i_i64[1]);
          v112 = v71;
          v111 = v70;
          *(_QWORD *)v66 = v70;
          v66[2] = v112;
          v72 = (__int64)v122;
          if ( !v122 )
            goto LABEL_92;
        }
        else
        {
          v77 = *(_QWORD *)(v7 + 80);
          v122 = (unsigned __int64 *)v77;
          if ( v77 )
            sub_B96E90((__int64)&v122, v77, 1);
          LODWORD(v123) = *(_DWORD *)(v7 + 72);
          v78 = sub_33FB310(v6, *(_QWORD *)v66, *((_QWORD *)v66 + 1), &v122, v121.m128i_u32[0], v121.m128i_i64[1]);
          v114 = v79;
          v113 = v78;
          *(_QWORD *)v66 = v78;
          v66[2] = v114;
          v72 = (__int64)v122;
          if ( !v122 )
            goto LABEL_92;
        }
        sub_B91220((__int64)&v122, v72);
      }
LABEL_92:
      v66 += 4;
      if ( v110 == v66 )
      {
        v63 = (unsigned __int64)v125;
        v62 = (unsigned int *)(unsigned int)v126;
        break;
      }
    }
  }
  v64 = *(_QWORD *)(v7 + 80);
  v122 = (unsigned __int64 *)v64;
  if ( v64 )
  {
    v109 = v63;
    v110 = v62;
    sub_B96E90((__int64)&v122, v64, 1);
    v63 = v109;
    v62 = v110;
  }
  *((_QWORD *)&v97 + 1) = v62;
  LODWORD(v123) = *(_DWORD *)(v7 + 72);
  *(_QWORD *)&v97 = v63;
  v52 = sub_33FC220(v6, 156, (unsigned int)&v122, v117, v118, a6, v97);
  if ( v122 )
  {
    v109 = v53;
    v110 = (unsigned int *)v52;
    sub_B91220((__int64)&v122, (__int64)v122);
    v53 = v109;
    v52 = (__int64)v110;
  }
LABEL_54:
  v109 = v53;
  v110 = (unsigned int *)v52;
  sub_325FE10(v131);
  result = v110;
  v54 = v109;
  if ( (_BYTE *)v128[0] != v129 )
  {
    _libc_free(v128[0]);
    v54 = v109;
    result = v110;
  }
  if ( v125 != (unsigned int *)v127 )
  {
    v109 = v54;
    v110 = result;
    _libc_free((unsigned __int64)v125);
    return v110;
  }
  return result;
}
