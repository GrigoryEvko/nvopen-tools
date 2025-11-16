// Function: sub_1FBA730
// Address: 0x1fba730
//
__int64 __fastcall sub_1FBA730(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rbx
  __m128i v14; // xmm1
  __int64 v15; // rcx
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  const void **v18; // rax
  int v19; // ecx
  int v20; // r8d
  int v21; // r9d
  __int64 result; // rax
  unsigned int v23; // ecx
  int v24; // eax
  char v25; // cl
  __int64 v26; // r9
  __int64 v27; // rdi
  unsigned int v28; // edx
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // r13
  __int64 v35; // rsi
  _QWORD *v36; // rdi
  _QWORD *v37; // r12
  __int64 v38; // rcx
  __int64 v39; // rax
  bool v40; // cc
  _QWORD *v41; // rax
  unsigned __int64 v42; // rdx
  __int16 v43; // ax
  __int64 v44; // rsi
  unsigned __int8 *v45; // rax
  unsigned int v46; // r11d
  char v47; // al
  unsigned int v48; // r11d
  char v49; // cl
  __int64 *v50; // r12
  __int128 v51; // rax
  __int128 v52; // rax
  __int64 *v53; // r13
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // rdx
  _QWORD *v57; // rax
  unsigned int v58; // esi
  unsigned __int8 v59; // cl
  unsigned int v60; // eax
  __int64 v61; // rax
  _QWORD *v62; // r14
  __int64 v63; // rdx
  _QWORD *v64; // rax
  int v65; // r14d
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rdx
  bool v69; // al
  __int64 v70; // rax
  bool v71; // al
  __int64 v72; // rdx
  int v73; // r9d
  _DWORD *v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rax
  _QWORD *v82; // r14
  unsigned __int8 *v83; // rax
  __int64 v84; // rdx
  const void **v85; // rax
  int v86; // eax
  __int64 v87; // r12
  unsigned __int8 *v88; // rax
  const void **v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rax
  _QWORD *v92; // rax
  __int128 v93; // rax
  __int128 v94; // rax
  __int64 v95; // rsi
  __int64 *v96; // r13
  int v97; // eax
  unsigned int v98; // r11d
  __int128 v99; // rax
  unsigned int v100; // eax
  const void **v101; // rdx
  bool v102; // al
  unsigned int v103; // eax
  unsigned int v104; // r14d
  unsigned __int8 v105; // al
  unsigned int v106; // eax
  __int64 v107; // r9
  unsigned int v108; // edi
  __int64 v109; // r11
  unsigned __int8 *v110; // rax
  __int64 v111; // rax
  const void **v112; // rdx
  __int128 v113; // rax
  __int128 v114; // rax
  __int128 v115; // rax
  __int64 v116; // rax
  __int64 *v117; // r13
  __int128 v118; // rax
  __int64 v119; // rbx
  __int128 v120; // kr00_16
  unsigned int v121; // eax
  const void **v122; // rdx
  unsigned int v123; // eax
  unsigned __int64 v124; // rdx
  __int64 v125; // [rsp+0h] [rbp-E0h]
  unsigned int v126; // [rsp+8h] [rbp-D8h]
  unsigned int v127; // [rsp+10h] [rbp-D0h]
  unsigned int v128; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v129; // [rsp+18h] [rbp-C8h]
  unsigned int v130; // [rsp+18h] [rbp-C8h]
  _QWORD *v131; // [rsp+18h] [rbp-C8h]
  unsigned int v132; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v133; // [rsp+18h] [rbp-C8h]
  char v134; // [rsp+20h] [rbp-C0h]
  unsigned int v135; // [rsp+20h] [rbp-C0h]
  const void **v136; // [rsp+20h] [rbp-C0h]
  unsigned int v137; // [rsp+20h] [rbp-C0h]
  __int64 v138; // [rsp+20h] [rbp-C0h]
  __int64 v139; // [rsp+20h] [rbp-C0h]
  __int64 v140; // [rsp+28h] [rbp-B8h]
  unsigned int v141; // [rsp+30h] [rbp-B0h]
  unsigned int v142; // [rsp+30h] [rbp-B0h]
  char v143; // [rsp+30h] [rbp-B0h]
  char v144; // [rsp+30h] [rbp-B0h]
  const void **v145; // [rsp+30h] [rbp-B0h]
  unsigned int v146; // [rsp+30h] [rbp-B0h]
  unsigned int v147; // [rsp+30h] [rbp-B0h]
  unsigned __int8 v148; // [rsp+30h] [rbp-B0h]
  __int64 v149; // [rsp+38h] [rbp-A8h]
  unsigned __int8 v150; // [rsp+38h] [rbp-A8h]
  __int64 v151; // [rsp+38h] [rbp-A8h]
  __int64 v152; // [rsp+40h] [rbp-A0h]
  unsigned __int8 v153; // [rsp+4Bh] [rbp-95h]
  int v154; // [rsp+4Ch] [rbp-94h]
  __int128 v155; // [rsp+50h] [rbp-90h]
  __int64 v156; // [rsp+50h] [rbp-90h]
  __int64 v157; // [rsp+50h] [rbp-90h]
  __int64 v158; // [rsp+60h] [rbp-80h]
  __int64 v159; // [rsp+60h] [rbp-80h]
  __int64 v160; // [rsp+60h] [rbp-80h]
  __int128 v161; // [rsp+60h] [rbp-80h]
  __int64 v162; // [rsp+70h] [rbp-70h] BYREF
  const void **v163; // [rsp+78h] [rbp-68h]
  __int64 v164; // [rsp+80h] [rbp-60h] BYREF
  const void **v165; // [rsp+88h] [rbp-58h]
  __int64 v166; // [rsp+90h] [rbp-50h] BYREF
  int v167; // [rsp+98h] [rbp-48h]
  __int64 (__fastcall *v168)(__int64 *, __int64 *, int); // [rsp+A0h] [rbp-40h]
  void *v169; // [rsp+A8h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 32);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = *(_QWORD *)v11;
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v149 = *(_QWORD *)(v11 + 40);
  v15 = *(unsigned int *)(v11 + 48);
  v141 = *(_DWORD *)(v11 + 48);
  v140 = *(unsigned int *)(v11 + 8);
  v16 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v11 + 40LL) + 16 * v140);
  v17 = *v16;
  v18 = (const void **)*((_QWORD *)v16 + 1);
  LOBYTE(v162) = v17;
  v163 = v18;
  v154 = sub_1D159C0((__int64)&v162, a2, v17, v15, a8, a9);
  if ( (unsigned int)sub_1D23330(*(_QWORD *)a1, v12.m128i_i64[0], v12.m128i_i64[1], 0) == v154 )
    return v12.m128i_i64[0];
  if ( (_BYTE)v162 )
  {
    if ( (unsigned __int8)(v162 - 14) > 0x5Fu )
      goto LABEL_7;
  }
  else if ( !sub_1F58D20((__int64)&v162) )
  {
    goto LABEL_7;
  }
  result = (__int64)sub_1FA8C50(a1, a2, *(double *)v12.m128i_i64, *(double *)v14.m128i_i64, a5);
  if ( result )
    return result;
LABEL_7:
  v152 = sub_1D1ADA0(v14.m128i_i64[0], v14.m128i_u32[2], v14.m128i_i64[1], v19, v20, v21);
  v24 = *(unsigned __int16 *)(v13 + 24);
  if ( (v24 == 32 || v24 == 10) && (*(_BYTE *)(v13 + 26) & 8) == 0 && v152 && (*(_BYTE *)(v152 + 26) & 8) == 0 )
  {
    v33 = *(_QWORD *)(a2 + 72);
    v34 = *(_QWORD *)a1;
    v166 = v33;
    if ( v33 )
      sub_1623A60((__int64)&v166, v33, 2);
    v167 = *(_DWORD *)(a2 + 64);
    result = sub_1D392A0(v34, 123, (__int64)&v166, v162, v163, v13, v12, *(double *)v14.m128i_i64, a5, v152);
    v35 = v166;
    if ( v166 )
      goto LABEL_30;
    return result;
  }
  LODWORD(v166) = v154;
  v169 = sub_1F6DC00;
  v168 = (__int64 (__fastcall *)(__int64 *, __int64 *, int))sub_1F6C150;
  v25 = sub_1D169E0(v14.m128i_i64[0], (_QWORD *)v14.m128i_i64[1], (__int64)&v166, v23);
  if ( v168 )
  {
    v134 = v25;
    v168(&v166, &v166, 3);
    v25 = v134;
  }
  if ( v25 )
  {
    v36 = *(_QWORD **)a1;
    v166 = 0;
    v167 = 0;
    v37 = sub_1D2B300(v36, 0x30u, (__int64)&v166, v162, (__int64)v163, v26);
    if ( v166 )
      sub_161E7C0((__int64)&v166, v166);
    return (__int64)v37;
  }
  if ( !v152 )
  {
    result = (__int64)sub_1F77C50((__int64 **)a1, a2, *(double *)v12.m128i_i64, *(double *)v14.m128i_i64, a5);
    if ( result )
      return result;
    v32 = *(unsigned __int16 *)(v13 + 24);
LABEL_17:
    if ( (_WORD)v32 != 123 )
    {
      LOBYTE(v30) = v152 != 0;
      goto LABEL_19;
    }
LABEL_46:
    v44 = *(_QWORD *)(a2 + 72);
    v164 = v44;
    if ( v44 )
      sub_1623A60((__int64)&v164, v44, 2);
    LODWORD(v165) = *(_DWORD *)(a2 + 64);
    v45 = (unsigned __int8 *)(*(_QWORD *)(v149 + 40) + 16LL * v141);
    v46 = *v45;
    v136 = (const void **)*((_QWORD *)v45 + 1);
    LODWORD(v166) = v154;
    v169 = sub_1F6E8F0;
    v168 = (__int64 (__fastcall *)(__int64 *, __int64 *, int))sub_1F6C180;
    v142 = v46;
    v47 = sub_1D16BF0(
            v14.m128i_i64[0],
            v14.m128i_u32[2],
            *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
            (__int64)&v166);
    v48 = v142;
    v49 = v47;
    if ( v168 )
    {
      v130 = v142;
      v143 = v47;
      v168(&v166, &v166, 3);
      v48 = v130;
      v49 = v143;
    }
    if ( v49 )
    {
      v50 = *(__int64 **)a1;
      *(_QWORD *)&v51 = sub_1D38BB0(
                          *(_QWORD *)a1,
                          (unsigned int)(v154 - 1),
                          (__int64)&v164,
                          v48,
                          v136,
                          0,
                          v12,
                          *(double *)v14.m128i_i64,
                          a5,
                          0);
      result = (__int64)sub_1D332F0(
                          v50,
                          123,
                          (__int64)&v164,
                          (unsigned int)v162,
                          v163,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          **(_QWORD **)(v13 + 32),
                          *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                          v51);
      goto LABEL_52;
    }
    v147 = v48;
    LODWORD(v166) = v154;
    v169 = sub_1F6E220;
    v168 = (__int64 (__fastcall *)(__int64 *, __int64 *, int))sub_1F6C1B0;
    LOBYTE(v97) = sub_1D16BF0(
                    v14.m128i_i64[0],
                    v14.m128i_u32[2],
                    *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
                    *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
                    (__int64)&v166);
    v98 = v147;
    LODWORD(v38) = v97;
    if ( v168 )
    {
      v132 = v147;
      v148 = v97;
      v168(&v166, &v166, 3);
      v98 = v132;
      LODWORD(v38) = v148;
    }
    if ( (_BYTE)v38 )
    {
      *(_QWORD *)&v99 = sub_1D332F0(
                          *(__int64 **)a1,
                          52,
                          (__int64)&v164,
                          v98,
                          v136,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          v14.m128i_i64[0],
                          v14.m128i_u64[1],
                          *(_OWORD *)(*(_QWORD *)(v13 + 32) + 40LL));
      result = (__int64)sub_1D332F0(
                          *(__int64 **)a1,
                          123,
                          (__int64)&v164,
                          (unsigned int)v162,
                          v163,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          **(_QWORD **)(v13 + 32),
                          *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                          v99);
LABEL_52:
      if ( v164 )
      {
        v159 = result;
        sub_161E7C0((__int64)&v164, v164);
        return v159;
      }
      return result;
    }
    if ( v164 )
      sub_161E7C0((__int64)&v164, v164);
    v43 = *(_WORD *)(v13 + 24);
    goto LABEL_113;
  }
  v27 = *(_QWORD *)(v152 + 88);
  v28 = *(_DWORD *)(v27 + 32);
  if ( v28 > 0x40 )
  {
    if ( v28 != (unsigned int)sub_16A57B0(v27 + 24) )
      goto LABEL_15;
    return v12.m128i_i64[0];
  }
  if ( !*(_QWORD *)(v27 + 24) )
    return v12.m128i_i64[0];
LABEL_15:
  result = (__int64)sub_1F77C50((__int64 **)a1, a2, *(double *)v12.m128i_i64, *(double *)v14.m128i_i64, a5);
  if ( result )
    return result;
  LOWORD(v32) = *(_WORD *)(v13 + 24);
  if ( (_WORD)v32 != 122 )
    goto LABEL_17;
  v54 = *(_QWORD *)(v13 + 32);
  LODWORD(v38) = v149;
  if ( v149 != *(_QWORD *)(v54 + 40) || (LODWORD(v38) = v141, *(_DWORD *)(v54 + 48) != v141) )
  {
LABEL_65:
    v55 = sub_1D1ADA0(*(_QWORD *)(v54 + 40), *(_QWORD *)(v54 + 48), v29, v38, v30, v31);
    if ( !v55 )
      goto LABEL_83;
    v131 = *(_QWORD **)(*(_QWORD *)a1 + 48LL);
    v56 = *(_QWORD *)(v152 + 88);
    v57 = *(_QWORD **)(v56 + 24);
    if ( *(_DWORD *)(v56 + 32) > 0x40u )
      v57 = (_QWORD *)*v57;
    v58 = v154 - (_DWORD)v57;
    if ( v154 - (_DWORD)v57 == 32 )
    {
      v59 = 5;
    }
    else if ( v58 > 0x20 )
    {
      if ( v58 == 64 )
      {
        v59 = 6;
      }
      else
      {
        if ( v58 != 128 )
        {
LABEL_106:
          v138 = v55;
          v100 = sub_1F58CC0(v131, v58);
          v55 = v138;
          v127 = v100;
          v59 = v100;
          v145 = v101;
LABEL_73:
          v60 = v127;
          LOBYTE(v60) = v59;
          v128 = v60;
          v137 = v60;
          if ( (_BYTE)v162 )
          {
            if ( (unsigned __int8)(v162 - 14) > 0x5Fu )
              goto LABEL_75;
            v104 = word_42FA680[(unsigned __int8)(v162 - 14)];
          }
          else
          {
            v153 = v59;
            v125 = v55;
            v102 = sub_1F58D20((__int64)&v162);
            v55 = v125;
            v59 = v153;
            if ( !v102 )
              goto LABEL_75;
            v103 = sub_1F58D30((__int64)&v162);
            v55 = v125;
            v104 = v103;
          }
          v139 = v55;
          v105 = sub_1D15020(v128, v104);
          v55 = v139;
          v59 = v105;
          if ( v105 )
          {
            v145 = 0;
          }
          else
          {
            v121 = sub_1F593D0(v131, v128, (__int64)v145, v104);
            v55 = v139;
            v126 = v121;
            v59 = v121;
            v145 = v122;
          }
          v106 = v126;
          LOBYTE(v106) = v59;
          v137 = v106;
LABEL_75:
          v61 = *(_QWORD *)(v152 + 88);
          v62 = *(_QWORD **)(v61 + 24);
          if ( *(_DWORD *)(v61 + 32) > 0x40u )
            v62 = (_QWORD *)*v62;
          v63 = *(_QWORD *)(v55 + 88);
          v64 = *(_QWORD **)(v63 + 24);
          if ( *(_DWORD *)(v63 + 32) > 0x40u )
            v64 = (_QWORD *)*v64;
          v65 = (_DWORD)v62 - (_DWORD)v64;
          if ( v65 > 0 )
          {
            v66 = *(_QWORD *)(a1 + 8);
            v67 = 1;
            if ( v59 == 1 || v59 && (v67 = v59, *(_QWORD *)(v66 + 8LL * v59 + 120)) )
            {
              v133 = v59;
              if ( (*(_BYTE *)(v66 + 259 * v67 + 2564) & 0xFB) == 0 && sub_1F6C880(v66, 0x91u, v162) )
              {
                v108 = v137;
                LOBYTE(v108) = v133;
                if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, const void **, _QWORD, const void **))(*(_QWORD *)v107 + 800LL))(
                       v107,
                       (unsigned int)v162,
                       v163,
                       v108,
                       v145) )
                {
                  v166 = *(_QWORD *)(a2 + 72);
                  if ( v166 )
                    sub_1F6CA20(&v166);
                  v109 = *(_QWORD *)a1;
                  v167 = *(_DWORD *)(a2 + 64);
                  v160 = v109;
                  v110 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v13 + 32) + 40LL)
                                           + 16LL * *(unsigned int *)(*(_QWORD *)(v13 + 32) + 8LL));
                  v111 = sub_1F6BF40(a1, *v110, *((_QWORD *)v110 + 1));
                  *(_QWORD *)&v113 = sub_1D38BB0(
                                       v160,
                                       v65,
                                       (__int64)&v166,
                                       v111,
                                       v112,
                                       0,
                                       v12,
                                       *(double *)v14.m128i_i64,
                                       a5,
                                       0);
                  *(_QWORD *)&v114 = sub_1D332F0(
                                       *(__int64 **)a1,
                                       124,
                                       (__int64)&v166,
                                       (unsigned int)v162,
                                       v163,
                                       0,
                                       *(double *)v12.m128i_i64,
                                       *(double *)v14.m128i_i64,
                                       a5,
                                       **(_QWORD **)(v13 + 32),
                                       *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                                       v113);
                  *(_QWORD *)&v115 = sub_1D309E0(
                                       *(__int64 **)a1,
                                       145,
                                       (__int64)&v166,
                                       v108,
                                       v145,
                                       0,
                                       *(double *)v12.m128i_i64,
                                       *(double *)v14.m128i_i64,
                                       *(double *)a5.m128i_i64,
                                       v114);
                  v116 = sub_1D309E0(
                           *(__int64 **)a1,
                           142,
                           (__int64)&v166,
                           **(unsigned __int8 **)(a2 + 40),
                           *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
                           0,
                           *(double *)v12.m128i_i64,
                           *(double *)v14.m128i_i64,
                           *(double *)a5.m128i_i64,
                           v115);
LABEL_130:
                  v157 = v116;
                  sub_17CD270(&v166);
                  return v157;
                }
              }
            }
          }
LABEL_83:
          LOBYTE(v30) = 1;
          goto LABEL_19;
        }
        v59 = 7;
      }
    }
    else if ( v58 == 8 )
    {
      v59 = 3;
    }
    else
    {
      v59 = 4;
      if ( v58 != 16 )
      {
        v59 = 2;
        if ( v58 != 1 )
          goto LABEL_106;
      }
    }
    v145 = 0;
    goto LABEL_73;
  }
  v39 = *(_QWORD *)(v152 + 88);
  v40 = *(_DWORD *)(v39 + 32) <= 0x40u;
  v41 = *(_QWORD **)(v39 + 24);
  if ( !v40 )
    v41 = (_QWORD *)*v41;
  v135 = sub_1F7DE30(*(_QWORD **)(*(_QWORD *)a1 + 48LL), v154 - (int)v41);
  v129 = v42;
  if ( sub_1F7E0D0((__int64)&v162) )
  {
    v123 = sub_1D15970(&v162);
    v135 = sub_1F7DEB0(*(_QWORD **)(*(_QWORD *)a1 + 48LL), v135, v129, v123, 0);
    v129 = v124;
  }
  if ( !*(_BYTE *)(a1 + 24) || sub_1F6C830(*(_QWORD *)(a1 + 8), 0x94u, v135) )
  {
    v117 = *(__int64 **)a1;
    *(_QWORD *)&v118 = sub_1D2EF30(v117, v135, v129, v38, v30, v31);
    v119 = *(_QWORD *)(v13 + 32);
    v120 = v118;
    v166 = *(_QWORD *)(a2 + 72);
    if ( v166 )
    {
      v161 = v118;
      sub_1F6CA20(&v166);
      v120 = v161;
    }
    v167 = *(_DWORD *)(a2 + 64);
    v116 = (__int64)sub_1D332F0(
                      v117,
                      148,
                      (__int64)&v166,
                      (unsigned int)v162,
                      v163,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)v14.m128i_i64,
                      a5,
                      *(_QWORD *)v119,
                      *(_QWORD *)(v119 + 8),
                      v120);
    goto LABEL_130;
  }
  v43 = *(_WORD *)(v13 + 24);
  if ( v43 == 123 )
    goto LABEL_46;
LABEL_113:
  LOBYTE(v30) = v152 != 0;
  if ( v43 == 122 && v152 )
  {
    v54 = *(_QWORD *)(v13 + 32);
    goto LABEL_65;
  }
LABEL_19:
  if ( *(_WORD *)(v149 + 24) == 145 && *(_WORD *)(**(_QWORD **)(v149 + 32) + 24LL) == 118 )
  {
    v144 = v30;
    *(_QWORD *)&v52 = sub_1F87630((__int64 **)a1, v149, *(double *)v12.m128i_i64, *(double *)v14.m128i_i64, a5);
    LOBYTE(v30) = v144;
    if ( (_QWORD)v52 )
    {
      v53 = *(__int64 **)a1;
      v166 = *(_QWORD *)(a2 + 72);
      if ( v166 )
      {
        v155 = v52;
        sub_1F6CA20(&v166);
        v52 = v155;
      }
      v167 = *(_DWORD *)(a2 + 64);
      result = (__int64)sub_1D332F0(
                          v53,
                          123,
                          (__int64)&v166,
                          (unsigned int)v162,
                          v163,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          v12.m128i_i64[0],
                          v12.m128i_u64[1],
                          v52);
      goto LABEL_59;
    }
  }
  if ( *(_WORD *)(v13 + 24) != 145
    || (v68 = *(_QWORD *)(v13 + 32), (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v68 + 24LL) - 123 > 1)
    || (v150 = v30, v69 = sub_1D18C00(*(_QWORD *)v68, 1, *(_DWORD *)(v68 + 8)), LOBYTE(v30) = v150, !v69)
    || (v70 = *(_QWORD *)(**(_QWORD **)(v13 + 32) + 32LL),
        v71 = sub_1D18C00(*(_QWORD *)(v70 + 40), 1, *(_DWORD *)(v70 + 48)),
        LODWORD(v30) = v150,
        !v71) )
  {
    if ( (_BYTE)v30 )
      goto LABEL_22;
LABEL_97:
    if ( !(unsigned __int8)sub_1D1F9F0(*(_QWORD *)a1, v12.m128i_i64[0], v12.m128i_i64[1], 0) )
      return 0;
    goto LABEL_98;
  }
  if ( !v150 )
    goto LABEL_97;
  v74 = *(_DWORD **)(v13 + 32);
  v75 = *(_QWORD *)v74;
  v146 = v74[2];
  v76 = *(_QWORD *)(*(_QWORD *)v74 + 32LL);
  v151 = v75;
  v77 = *(_QWORD *)(v76 + 48);
  v78 = sub_1D1ADA0(*(_QWORD *)(v76 + 40), v77, v72, v75, v30, v73);
  if ( v78 )
  {
    v81 = *(_QWORD *)(v78 + 88);
    v82 = *(_QWORD **)(v81 + 24);
    if ( *(_DWORD *)(v81 + 32) > 0x40u )
      v82 = (_QWORD *)*v82;
    v83 = (unsigned __int8 *)(*(_QWORD *)(v151 + 40) + 16LL * v146);
    v84 = *v83;
    v85 = (const void **)*((_QWORD *)v83 + 1);
    LOBYTE(v164) = v84;
    v165 = v85;
    if ( (unsigned int)sub_1D159C0((__int64)&v164, v77, v84, v151, v79, v80) - v154 == (_DWORD)v82 )
    {
      v166 = *(_QWORD *)(a2 + 72);
      if ( v166 )
        sub_1F6CA20(&v166);
      v86 = *(_DWORD *)(a2 + 64);
      v87 = *(_QWORD *)a1;
      v167 = v86;
      v88 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v151 + 32) + 40LL)
                              + 16LL * *(unsigned int *)(*(_QWORD *)(v151 + 32) + 8LL));
      v90 = sub_1F6BF40(a1, *v88, *((_QWORD *)v88 + 1));
      v91 = *(_QWORD *)(v152 + 88);
      v40 = *(_DWORD *)(v91 + 32) <= 0x40u;
      v92 = *(_QWORD **)(v91 + 24);
      if ( !v40 )
        v92 = (_QWORD *)*v92;
      *(_QWORD *)&v93 = sub_1D38BB0(
                          v87,
                          (__int64)v92 + (unsigned int)v82,
                          (__int64)&v166,
                          v90,
                          v89,
                          0,
                          v12,
                          *(double *)v14.m128i_i64,
                          a5,
                          0);
      *(_QWORD *)&v94 = sub_1D332F0(
                          *(__int64 **)a1,
                          123,
                          (__int64)&v166,
                          (unsigned int)v164,
                          v165,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          **(_QWORD **)(v151 + 32),
                          *(_QWORD *)(*(_QWORD *)(v151 + 32) + 8LL),
                          v93);
      v156 = sub_1D309E0(
               *(__int64 **)a1,
               145,
               (__int64)&v166,
               (unsigned int)v162,
               v163,
               0,
               *(double *)v12.m128i_i64,
               *(double *)v14.m128i_i64,
               *(double *)a5.m128i_i64,
               v94);
      sub_17CD270(&v166);
      return v156;
    }
  }
LABEL_22:
  if ( (unsigned __int8)sub_1FB1D70(a1, a2, 0) )
    return a2;
  if ( !(unsigned __int8)sub_1D1F9F0(*(_QWORD *)a1, v12.m128i_i64[0], v12.m128i_i64[1], 0) )
  {
    if ( (*(_BYTE *)(v152 + 26) & 8) == 0 )
    {
      result = (__int64)sub_1F77880((__int64 **)a1, a2, *(double *)v12.m128i_i64, *(double *)v14.m128i_i64, a5);
      if ( result )
        return result;
    }
    return 0;
  }
LABEL_98:
  v95 = *(_QWORD *)(a2 + 72);
  v96 = *(__int64 **)a1;
  v166 = v95;
  if ( v95 )
    sub_1623A60((__int64)&v166, v95, 2);
  v167 = *(_DWORD *)(a2 + 64);
  result = (__int64)sub_1D332F0(
                      v96,
                      124,
                      (__int64)&v166,
                      (unsigned int)v162,
                      v163,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)v14.m128i_i64,
                      a5,
                      v13,
                      v140 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                      *(_OWORD *)&v14);
LABEL_59:
  v35 = v166;
  if ( v166 )
  {
LABEL_30:
    v158 = result;
    sub_161E7C0((__int64)&v166, v35);
    return v158;
  }
  return result;
}
