// Function: sub_3878E40
// Address: 0x3878e40
//
__int64 __fastcall sub_3878E40(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 **a5,
        __int64 *a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        _BYTE *a15)
{
  __int64 v15; // r15
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r9d
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // di
  unsigned int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r13
  char v37; // al
  _QWORD *v38; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  void *v45; // rdi
  unsigned int v46; // eax
  __int64 v47; // rdx
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 ***v58; // rax
  __int64 v59; // r13
  __int64 v60; // r13
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rsi
  unsigned __int8 *v65; // rsi
  __int64 v66; // r14
  int v67; // r13d
  int v68; // r14d
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 *v71; // r14
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  _QWORD *v78; // rax
  __int64 v79; // rcx
  __int64 *v80; // rax
  __int64 v81; // rsi
  unsigned __int64 v82; // rcx
  __int64 v83; // rcx
  __int64 v84; // rdx
  __int64 v85; // rax
  int v86; // esi
  __int64 v87; // r14
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  unsigned __int64 v91; // rsi
  double v92; // xmm4_8
  double v93; // xmm5_8
  __int64 v94; // rax
  void *v95; // rcx
  __int64 v96; // rsi
  __int64 v97; // rdx
  unsigned __int64 v98; // rax
  __int64 v99; // rdi
  int v100; // eax
  __int64 v101; // rax
  int v102; // ecx
  char v103; // al
  char v104; // al
  __int64 v105; // rsi
  double v106; // xmm4_8
  double v107; // xmm5_8
  __int64 v108; // rax
  __int64 v109; // rcx
  unsigned int v110; // r14d
  _QWORD **v111; // rax
  __int64 v112; // r14
  __int64 v113; // rbx
  __int64 v114; // rax
  __int64 v115; // rbx
  __int64 v116; // rax
  unsigned __int64 v117; // rax
  char v118; // al
  _QWORD *v119; // rdx
  unsigned int v120; // esi
  int v121; // eax
  int v122; // eax
  unsigned int v123; // esi
  int v124; // eax
  int v125; // eax
  __int64 v126; // r10
  unsigned int v127; // r14d
  _QWORD **v128; // rax
  __int64 v129; // r14
  __int64 v130; // rbx
  __int64 v131; // rax
  __int64 v132; // rbx
  __int64 v133; // rax
  __int64 v134; // rax
  void *v135; // [rsp-10h] [rbp-170h]
  __int64 v136; // [rsp-8h] [rbp-168h]
  __int64 ***v137; // [rsp+10h] [rbp-150h]
  __int64 v138; // [rsp+18h] [rbp-148h]
  __int64 v139; // [rsp+18h] [rbp-148h]
  __int64 v140; // [rsp+18h] [rbp-148h]
  __int64 v141; // [rsp+20h] [rbp-140h]
  __int64 v142; // [rsp+28h] [rbp-138h]
  __int64 v143; // [rsp+30h] [rbp-130h]
  __int64 v144; // [rsp+30h] [rbp-130h]
  __int64 v145; // [rsp+38h] [rbp-128h]
  bool v146; // [rsp+38h] [rbp-128h]
  __int64 v147; // [rsp+40h] [rbp-120h]
  bool v148; // [rsp+40h] [rbp-120h]
  char v150; // [rsp+48h] [rbp-118h]
  __int64 v151; // [rsp+50h] [rbp-110h]
  _QWORD *v152; // [rsp+50h] [rbp-110h]
  __int64 ***v153; // [rsp+50h] [rbp-110h]
  __int64 v154; // [rsp+50h] [rbp-110h]
  bool v155; // [rsp+58h] [rbp-108h]
  __int64 v156; // [rsp+58h] [rbp-108h]
  __int64 v157; // [rsp+58h] [rbp-108h]
  __int64 v158; // [rsp+58h] [rbp-108h]
  __int64 v159; // [rsp+60h] [rbp-100h]
  __int64 *v160; // [rsp+60h] [rbp-100h]
  __int64 v164[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v165; // [rsp+90h] [rbp-D0h]
  __int64 v166[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v167; // [rsp+B0h] [rbp-B0h]
  __int64 *v168[6]; // [rsp+C0h] [rbp-A0h] BYREF
  _QWORD v169[2]; // [rsp+F0h] [rbp-70h] BYREF
  unsigned __int64 v170; // [rsp+100h] [rbp-60h]
  char v171[72]; // [rsp+118h] [rbp-48h] BYREF

  v15 = a3;
  v18 = sub_13FCB50(a3);
  if ( v18 )
  {
    v22 = v18;
    *a6 = 0;
    v155 = 0;
    *a15 = 0;
    v23 = a1[26];
    if ( v23 )
      v155 = sub_15CC890(*(_QWORD *)(*a1 + 56LL), v22, **(_QWORD **)(v23 + 32));
    v25 = sub_157F280(**(_QWORD **)(v15 + 32));
    if ( v24 != v25 )
    {
      v145 = 0;
      v159 = v15;
      v147 = 0;
      v26 = v24;
      while ( 1 )
      {
        if ( !sub_1456C80(*a1, *(_QWORD *)v25) )
          goto LABEL_26;
        v27 = sub_146F1B0(*a1, v25);
        v20 = v27;
        if ( *(_WORD *)(v27 + 24) != 7 || v27 != a2 && !v155 )
          goto LABEL_26;
        v28 = 0x17FFFFFFE8LL;
        v29 = *(_BYTE *)(v25 + 23) & 0x40;
        v30 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
        if ( v30 )
        {
          v31 = 24LL * *(unsigned int *)(v25 + 56) + 8;
          v32 = 0;
          do
          {
            v19 = v25 - 24LL * v30;
            if ( v29 )
              v19 = *(_QWORD *)(v25 - 8);
            if ( v22 == *(_QWORD *)(v19 + v31) )
            {
              v28 = 24 * v32;
              goto LABEL_17;
            }
            ++v32;
            v31 += 8;
          }
          while ( v30 != (_DWORD)v32 );
          v28 = 0x17FFFFFFE8LL;
        }
LABEL_17:
        if ( v29 )
        {
          v33 = *(_QWORD *)(v25 - 8);
        }
        else
        {
          v19 = 24LL * v30;
          v33 = v25 - v19;
        }
        v34 = *(_QWORD *)(v33 + v28);
        if ( *(_BYTE *)(v34 + 16) <= 0x17u )
          goto LABEL_26;
        v151 = v20;
        if ( *((_BYTE *)a1 + 257) )
        {
          if ( !(unsigned __int8)sub_3870860((__int64)a1, v25, (__int64 *)v34, v159) )
            goto LABEL_26;
          v20 = v151;
          if ( a1[26] == v159 )
          {
            v104 = sub_3871F10(a1, v34, a1[27]);
            v20 = v151;
            if ( !v104 )
              goto LABEL_26;
          }
        }
        else
        {
          v103 = sub_3870430(a1, v25, v34, v159);
          v20 = v151;
          if ( !v103 )
            goto LABEL_26;
        }
        if ( v20 == a2 )
        {
          v126 = v34;
          v36 = v25;
          v15 = v159;
          v145 = v126;
          *a6 = 0;
          *a15 = 0;
          goto LABEL_31;
        }
        if ( *a6 && !*a15 )
          goto LABEL_26;
        v142 = v20;
        v152 = (_QWORD *)*a1;
        v40 = sub_1456040(**(_QWORD **)(v20 + 32));
        v143 = sub_1456E10((__int64)v152, v40);
        v41 = sub_1456040(**(_QWORD **)(a2 + 32));
        v42 = sub_1456E10((__int64)v152, v41);
        v19 = v143;
        if ( *(_DWORD *)(v42 + 8) >> 8 > *(_DWORD *)(v143 + 8) >> 8 )
          goto LABEL_26;
        v43 = sub_1483C80(v152, v142, v42, a7, a8);
        if ( *(_WORD *)(v43 + 24) != 7 )
          goto LABEL_26;
        if ( a2 == v43 )
          break;
        v144 = v43;
        v44 = sub_1480620((__int64)v152, a2, 0);
        if ( v144 == sub_13A5B00((__int64)v152, **(_QWORD **)(a2 + 32), v44, 0, 0) )
        {
          *a15 = 1;
LABEL_141:
          v154 = *a1;
          v134 = sub_1456040(**(_QWORD **)(a2 + 32));
          v145 = v34;
          v147 = v25;
          *a6 = sub_1456E10(v154, v134);
        }
LABEL_26:
        v35 = *(_QWORD *)(v25 + 32);
        if ( !v35 )
          BUG();
        v25 = 0;
        if ( *(_BYTE *)(v35 - 8) == 77 )
          v25 = v35 - 24;
        if ( v26 == v25 )
        {
          v36 = v147;
          v15 = v159;
          if ( !v147 )
            goto LABEL_41;
LABEL_31:
          if ( a1[26] == v15 )
            sub_38708D0((__int64)a1, *(_QWORD *)(*a1 + 56LL), v145, a1[27], v36);
          v168[0] = (__int64 *)v36;
          v37 = sub_3872C70((__int64)(a1 + 7), (__int64 *)v168, v169);
          v38 = (_QWORD *)v169[0];
          if ( v37 )
          {
LABEL_34:
            sub_38740E0((__int64)a1, v145);
            return v36;
          }
          v123 = *((_DWORD *)a1 + 20);
          v124 = *((_DWORD *)a1 + 18);
          ++a1[7];
          v125 = v124 + 1;
          if ( 4 * v125 >= 3 * v123 )
          {
            v123 *= 2;
          }
          else if ( v123 - *((_DWORD *)a1 + 19) - v125 > v123 >> 3 )
          {
LABEL_134:
            *((_DWORD *)a1 + 18) = v125;
            if ( *v38 != -8 )
              --*((_DWORD *)a1 + 19);
            *v38 = v168[0];
            goto LABEL_34;
          }
          sub_3873F30((__int64)(a1 + 7), v123);
          sub_3872C70((__int64)(a1 + 7), (__int64 *)v168, v169);
          v38 = (_QWORD *)v169[0];
          v125 = *((_DWORD *)a1 + 18) + 1;
          goto LABEL_134;
        }
      }
      *a15 = 0;
      goto LABEL_141;
    }
  }
LABEL_41:
  v160 = a1 + 33;
  sub_38701C0(v168, a1 + 33, (__int64)a1, v19, v20, v21);
  v141 = (__int64)(a1 + 19);
  sub_16CCCB0(v169, (__int64)v171, (__int64)(a1 + 19));
  v45 = (void *)a1[21];
  ++a1[19];
  if ( v45 != (void *)a1[20] )
  {
    v46 = 4 * (*((_DWORD *)a1 + 45) - *((_DWORD *)a1 + 46));
    v47 = *((unsigned int *)a1 + 44);
    if ( v46 < 0x20 )
      v46 = 32;
    if ( (unsigned int)v47 > v46 )
    {
      sub_16CC920(v141);
      goto LABEL_47;
    }
    memset(v45, -1, 8 * v47);
  }
  *(_QWORD *)((char *)a1 + 180) = 0;
LABEL_47:
  v48 = sub_13FC520(v15);
  v49 = sub_157EBA0(v48);
  v137 = sub_38767A0(
           a1,
           **(_QWORD **)(a2 + 32),
           (__int64 **)a4,
           v49,
           (__m128)a7,
           *(double *)a8.m128i_i64,
           a9,
           a10,
           v50,
           v51,
           a13,
           a14);
  v52 = sub_13A5BC0((_QWORD *)a2, *a1);
  v55 = v52;
  if ( *(_BYTE *)(a4 + 8) != 15 && (v150 = sub_1456260(v52)) != 0 )
  {
    v105 = sub_1480620(*a1, v55, 0);
    v108 = *(_QWORD *)(**(_QWORD **)(v15 + 32) + 48LL);
    v109 = v108 - 24;
    if ( !v108 )
      v109 = 0;
    v146 = 0;
    v153 = sub_38767A0(a1, v105, a5, v109, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v106, v107, a13, a14);
    v148 = 0;
  }
  else
  {
    v56 = *(_QWORD *)(**(_QWORD **)(v15 + 32) + 48LL);
    v57 = v56 - 24;
    if ( !v56 )
      v57 = 0;
    v58 = sub_38767A0(a1, v55, a5, v57, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v53, v54, a13, a14);
    v59 = *a1;
    v153 = v58;
    v146 = 0;
    if ( *(_BYTE *)(sub_1456040(**(_QWORD **)(a2 + 32)) + 8) == 11 )
    {
      v110 = 2 * (*(_DWORD *)(sub_1456040(**(_QWORD **)(a2 + 32)) + 8) >> 8);
      v111 = (_QWORD **)sub_1456040(**(_QWORD **)(a2 + 32));
      v112 = sub_1644900(*v111, v110);
      v157 = sub_13A5BC0((_QWORD *)a2, v59);
      v113 = sub_14747F0(v59, a2, v112, 0);
      v114 = sub_14747F0(v59, v157, v112, 0);
      v115 = sub_13A5B00(v59, v114, v113, 0, 0);
      v116 = sub_13A5B00(v59, a2, v157, 0, 0);
      v146 = v115 == sub_14747F0(v59, v116, v112, 0);
    }
    v60 = *a1;
    v150 = 0;
    v148 = 0;
    if ( *(_BYTE *)(sub_1456040(**(_QWORD **)(a2 + 32)) + 8) == 11 )
    {
      v127 = 2 * (*(_DWORD *)(sub_1456040(**(_QWORD **)(a2 + 32)) + 8) >> 8);
      v128 = (_QWORD **)sub_1456040(**(_QWORD **)(a2 + 32));
      v129 = sub_1644900(*v128, v127);
      v158 = sub_13A5BC0((_QWORD *)a2, v60);
      v130 = sub_147B0D0(v60, a2, v129, 0);
      v131 = sub_147B0D0(v60, v158, v129, 0);
      v132 = sub_13A5B00(v60, v131, v130, 0, 0);
      v133 = sub_13A5B00(v60, a2, v158, 0, 0);
      v148 = v132 == sub_147B0D0(v60, v133, v129, 0);
    }
  }
  v61 = **(_QWORD **)(v15 + 32);
  v62 = *(_QWORD *)(v61 + 48);
  a1[34] = v61;
  a1[35] = v62;
  if ( v62 != v61 + 40 )
  {
    if ( !v62 )
      BUG();
    v63 = *(_QWORD *)(v62 + 24);
    v166[0] = v63;
    if ( v63 )
    {
      sub_1623A60((__int64)v166, v63, 2);
      v64 = a1[33];
      if ( !v64 )
        goto LABEL_60;
    }
    else
    {
      v64 = a1[33];
      if ( !v64 )
        goto LABEL_62;
    }
    sub_161E7C0((__int64)v160, v64);
LABEL_60:
    v65 = (unsigned __int8 *)v166[0];
    a1[33] = v166[0];
    if ( v65 )
      sub_1623210((__int64)v166, v65, (__int64)v160);
    goto LABEL_62;
  }
  do
LABEL_62:
    v61 = *(_QWORD *)(v61 + 8);
  while ( v61 && (unsigned __int8)(*((_BYTE *)sub_1648700(v61) + 16) - 25) > 9u );
  if ( *(_BYTE *)a1[2] )
  {
    v164[0] = a1[2];
    v164[1] = (__int64)".iv";
    v165 = 771;
  }
  else
  {
    v164[0] = (__int64)".iv";
    v165 = 259;
  }
  if ( v61 )
  {
    v66 = v61;
    v67 = 0;
    while ( 1 )
    {
      v66 = *(_QWORD *)(v66 + 8);
      if ( !v66 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v66) + 16) - 25) <= 9u )
      {
        v66 = *(_QWORD *)(v66 + 8);
        ++v67;
        if ( !v66 )
          goto LABEL_71;
      }
    }
LABEL_71:
    v68 = v67 + 1;
  }
  else
  {
    v68 = 0;
  }
  v167 = 257;
  v69 = sub_1648B60(64);
  v36 = v69;
  if ( v69 )
  {
    v156 = v69;
    sub_15F1EA0(v69, a4, 53, 0, 0, 0);
    *(_DWORD *)(v36 + 56) = v68;
    sub_164B780(v36, v166);
    sub_1648880(v36, *(_DWORD *)(v36 + 56), 1);
  }
  else
  {
    v156 = 0;
  }
  v70 = a1[34];
  if ( v70 )
  {
    v71 = (__int64 *)a1[35];
    sub_157E9D0(v70 + 40, v36);
    v72 = *(_QWORD *)(v36 + 24);
    v73 = *v71;
    *(_QWORD *)(v36 + 32) = v71;
    v73 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v36 + 24) = v73 | v72 & 7;
    *(_QWORD *)(v73 + 8) = v36 + 24;
    *v71 = *v71 & 7 | (v36 + 24);
  }
  sub_164B780(v156, v164);
  sub_12A86E0(v160, v36);
  sub_38740E0((__int64)a1, v36);
  if ( v61 )
  {
    v78 = sub_1648700(v61);
LABEL_89:
    v87 = v78[5];
    if ( !sub_1377F70(v15 + 56, v87) )
    {
      sub_1704F80(v36, (__int64)v137, v87, v88, v89, v90);
      goto LABEL_87;
    }
    if ( a1[26] == v15 )
      v91 = a1[27];
    else
      v91 = sub_157EBA0(v87);
    sub_17050D0(v160, v91);
    v94 = sub_3878BC0(a1, v36, (__int64)v153, a7, a8, a9, a10, v92, v93, a13, a14, v15, a4, a5, v150);
    v95 = v135;
    v96 = v136;
    v97 = v94;
    v98 = *(unsigned __int8 *)(v94 + 16);
    if ( (unsigned __int8)v98 <= 0x17u )
    {
      if ( (_BYTE)v98 != 5 )
        goto LABEL_97;
      v117 = *(unsigned __int16 *)(v97 + 18);
      if ( (unsigned __int16)v117 > 0x17u )
        goto LABEL_97;
      v95 = &loc_80A800;
      if ( !_bittest64((const __int64 *)&v95, v117) )
        goto LABEL_97;
      if ( !v146 )
      {
LABEL_96:
        if ( !v148 )
          goto LABEL_97;
LABEL_115:
        v96 = 1;
        v139 = v97;
        sub_15F2330(v97, 1);
        v97 = v139;
        v100 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
        if ( v100 != *(_DWORD *)(v36 + 56) )
          goto LABEL_98;
        goto LABEL_116;
      }
    }
    else
    {
      if ( (unsigned __int8)v98 > 0x2Fu )
        goto LABEL_97;
      v99 = 0x80A800000000LL;
      if ( !_bittest64(&v99, v98) )
        goto LABEL_97;
      if ( !v146 )
        goto LABEL_96;
    }
    v96 = 1;
    v138 = v97;
    sub_15F2310(v97, 1);
    v97 = v138;
    if ( v148 )
      goto LABEL_115;
LABEL_97:
    v100 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    if ( v100 != *(_DWORD *)(v36 + 56) )
      goto LABEL_98;
LABEL_116:
    v140 = v97;
    sub_15F55D0(v36, v96, v97, (__int64)v95, v76, v77);
    v97 = v140;
    v100 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
LABEL_98:
    v101 = (v100 + 1) & 0xFFFFFFF;
    v102 = v101 | *(_DWORD *)(v36 + 20) & 0xF0000000;
    *(_DWORD *)(v36 + 20) = v102;
    if ( (v102 & 0x40000000) != 0 )
      v79 = *(_QWORD *)(v36 - 8);
    else
      v79 = v156 - 24 * v101;
    v80 = (__int64 *)(v79 + 24LL * (unsigned int)(v101 - 1));
    if ( *v80 )
    {
      v81 = v80[1];
      v82 = v80[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v82 = v81;
      if ( v81 )
        *(_QWORD *)(v81 + 16) = *(_QWORD *)(v81 + 16) & 3LL | v82;
    }
    *v80 = v97;
    v83 = *(_QWORD *)(v97 + 8);
    v80[1] = v83;
    if ( v83 )
    {
      LODWORD(v76) = (_DWORD)v80 + 8;
      *(_QWORD *)(v83 + 16) = (unsigned __int64)(v80 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
    }
    v80[2] = (v97 + 8) | v80[2] & 3;
    *(_QWORD *)(v97 + 8) = v80;
    v84 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    v85 = (unsigned int)(v84 - 1);
    if ( (*(_BYTE *)(v36 + 23) & 0x40) != 0 )
      v75 = *(_QWORD *)(v36 - 8);
    else
      v75 = v156 - 24 * v84;
    v74 = 3LL * *(unsigned int *)(v36 + 56);
    *(_QWORD *)(v75 + 8 * v85 + 24LL * *(unsigned int *)(v36 + 56) + 8) = v87;
LABEL_87:
    while ( 1 )
    {
      v61 = *(_QWORD *)(v61 + 8);
      if ( !v61 )
        break;
      v78 = sub_1648700(v61);
      v86 = *((unsigned __int8 *)v78 + 16);
      v74 = (unsigned int)(v86 - 25);
      if ( (unsigned __int8)(v86 - 25) <= 9u )
        goto LABEL_89;
    }
  }
  sub_16CCD50(v141, (__int64)v169, v74, v75, v76, v77);
  v164[0] = v36;
  v118 = sub_3872C70((__int64)(a1 + 7), v164, v166);
  v119 = (_QWORD *)v166[0];
  if ( !v118 )
  {
    v120 = *((_DWORD *)a1 + 20);
    v121 = *((_DWORD *)a1 + 18);
    ++a1[7];
    v122 = v121 + 1;
    if ( 4 * v122 >= 3 * v120 )
    {
      v120 *= 2;
    }
    else if ( v120 - *((_DWORD *)a1 + 19) - v122 > v120 >> 3 )
    {
LABEL_126:
      *((_DWORD *)a1 + 18) = v122;
      if ( *v119 != -8 )
        --*((_DWORD *)a1 + 19);
      *v119 = v164[0];
      goto LABEL_119;
    }
    sub_3873F30((__int64)(a1 + 7), v120);
    sub_3872C70((__int64)(a1 + 7), v164, v166);
    v119 = (_QWORD *)v166[0];
    v122 = *((_DWORD *)a1 + 18) + 1;
    goto LABEL_126;
  }
LABEL_119:
  if ( v170 != v169[1] )
    _libc_free(v170);
  sub_3870260((__int64)v168);
  return v36;
}
