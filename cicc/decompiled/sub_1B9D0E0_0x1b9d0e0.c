// Function: sub_1B9D0E0
// Address: 0x1b9d0e0
//
void __fastcall sub_1B9D0E0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  __int64 v12; // r13
  __int64 v13; // r8
  char v14; // di
  unsigned int v15; // ecx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r13
  unsigned __int64 v24; // rax
  int v25; // eax
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 **v30; // rax
  __int64 v31; // rax
  unsigned __int64 *v32; // rax
  __int64 v33; // r15
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 *v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdi
  __int64 v43; // r13
  __int64 v44; // rax
  int v45; // ecx
  __int64 v46; // rsi
  __int64 v47; // rdi
  int v48; // ecx
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // r9
  __int64 v52; // rsi
  int v53; // r8d
  int v54; // r9d
  unsigned __int64 v55; // r13
  __int64 *v56; // rdi
  size_t v57; // rdx
  __int64 *v58; // r13
  __int64 v59; // r14
  __int64 v60; // rax
  unsigned int v61; // r14d
  unsigned int v62; // r13d
  __int64 *v63; // r12
  __int64 v64; // rax
  __int64 v65; // rsi
  unsigned int v66; // r12d
  _QWORD *v67; // r13
  __int64 v68; // r14
  unsigned __int64 *v69; // rax
  double v70; // xmm4_8
  double v71; // xmm5_8
  __int64 v72; // r10
  __int64 v73; // r11
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdi
  int v78; // ecx
  __int64 v79; // rsi
  __int64 v80; // rdi
  int v81; // ecx
  unsigned int v82; // edx
  __int64 *v83; // rax
  __int64 v84; // r8
  __int64 v85; // rax
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  unsigned int v89; // eax
  __int64 v90; // rsi
  __int64 v91; // rsi
  __int64 v92; // r14
  __int64 v93; // rax
  __int64 v94; // rsi
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // r14
  _QWORD *v98; // rax
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rcx
  __int64 v102; // r15
  __int64 v103; // rdx
  __int64 v104; // rsi
  __int64 v105; // r13
  unsigned int v106; // edi
  __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 *v110; // rax
  __int64 v111; // rcx
  unsigned __int64 v112; // rdx
  __int64 v113; // rdx
  __int64 v114; // rax
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 v118; // rdx
  __int64 v119; // r14
  __int64 v120; // r13
  __int64 *v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  unsigned __int64 v124; // rax
  _QWORD *v125; // rdi
  int v126; // eax
  __int64 v127; // r14
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rax
  unsigned int v131; // r13d
  __int64 v132; // r14
  _QWORD *v133; // rdi
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rax
  int v137; // eax
  int v138; // eax
  int v139; // r8d
  int v140; // r9d
  __int64 **v141; // [rsp+10h] [rbp-100h]
  __int64 v142; // [rsp+18h] [rbp-F8h]
  __int64 *v143; // [rsp+20h] [rbp-F0h]
  __int64 *v144; // [rsp+28h] [rbp-E8h]
  __int64 v145; // [rsp+30h] [rbp-E0h]
  __int64 v146; // [rsp+30h] [rbp-E0h]
  __int64 v147; // [rsp+38h] [rbp-D8h]
  _QWORD *v148; // [rsp+38h] [rbp-D8h]
  __int64 *v149; // [rsp+38h] [rbp-D8h]
  __int64 v150; // [rsp+40h] [rbp-D0h]
  __int64 v151; // [rsp+40h] [rbp-D0h]
  __int64 v152; // [rsp+48h] [rbp-C8h]
  __int64 v153; // [rsp+48h] [rbp-C8h]
  const char *v154; // [rsp+50h] [rbp-C0h] BYREF
  char v155; // [rsp+60h] [rbp-B0h]
  char v156; // [rsp+61h] [rbp-AFh]
  __int64 v157[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v158; // [rsp+80h] [rbp-90h]
  __int64 *v159; // [rsp+90h] [rbp-80h] BYREF
  __int64 v160; // [rsp+98h] [rbp-78h]
  _WORD s[56]; // [rsp+A0h] [rbp-70h] BYREF

  v10 = a2;
  v12 = sub_13FC520(*(_QWORD *)(a1 + 8));
  v13 = sub_13FCB50(*(_QWORD *)(a1 + 8));
  v14 = *(_BYTE *)(a2 + 23) & 0x40;
  v15 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v15 )
  {
    v16 = 24LL * *(unsigned int *)(a2 + 56) + 8;
    v17 = 0;
    while ( 1 )
    {
      v18 = v10 - 24LL * v15;
      if ( v14 )
        v18 = *(_QWORD *)(v10 - 8);
      if ( v12 == *(_QWORD *)(v18 + v16) )
        break;
      ++v17;
      v16 += 8;
      if ( v15 == (_DWORD)v17 )
        goto LABEL_9;
    }
    v19 = 24 * v17;
    if ( !v14 )
      goto LABEL_8;
  }
  else
  {
LABEL_9:
    v19 = 0x17FFFFFFE8LL;
    if ( !v14 )
    {
LABEL_8:
      v20 = v10 - 24LL * v15;
      goto LABEL_11;
    }
  }
  v20 = *(_QWORD *)(v10 - 8);
LABEL_11:
  v141 = *(__int64 ***)(v20 + v19);
  v21 = 0x17FFFFFFE8LL;
  if ( v15 )
  {
    v22 = 0;
    do
    {
      if ( v13 == *(_QWORD *)(v20 + 24LL * *(unsigned int *)(v10 + 56) + 8 * v22 + 8) )
      {
        v21 = 24 * v22;
        goto LABEL_16;
      }
      ++v22;
    }
    while ( v15 != (_DWORD)v22 );
    v21 = 0x17FFFFFFE8LL;
  }
LABEL_16:
  v143 = (__int64 *)(a1 + 96);
  v144 = *(__int64 **)(v20 + v21);
  v23 = (__int64 *)v141;
  if ( *(_DWORD *)(a1 + 88) > 1u )
  {
    v24 = sub_157EBA0(*(_QWORD *)(a1 + 168));
    sub_17050D0((__int64 *)(a1 + 96), v24);
    v25 = *(_DWORD *)(a1 + 88);
    v26 = *(_QWORD **)(a1 + 120);
    s[0] = 259;
    v159 = (__int64 *)"vector.recur.init";
    v27 = (unsigned int)(v25 - 1);
    v28 = sub_1643350(v26);
    v29 = sub_159C470(v28, v27, 0);
    v30 = (__int64 **)sub_16463B0(*v141, *(_DWORD *)(a1 + 88));
    v31 = sub_1599EF0(v30);
    v23 = (__int64 *)sub_156D8B0((__int64 *)(a1 + 96), v31, (__int64)v141, v29, (__int64)&v159);
  }
  v159 = (__int64 *)v10;
  v32 = sub_1B99AC0((_QWORD *)(a1 + 288), (unsigned __int64 *)&v159);
  sub_17050D0(v143, *(_QWORD *)*v32);
  v157[0] = (__int64)"vector.recur";
  v158 = 259;
  v33 = *v23;
  s[0] = 257;
  v142 = sub_1648B60(64);
  if ( v142 )
  {
    v34 = v142;
    sub_15F1EA0(v142, v33, 53, 0, 0, 0);
    *(_DWORD *)(v142 + 56) = 2;
    sub_164B780(v142, (__int64 *)&v159);
    sub_1648880(v34, *(_DWORD *)(v34 + 56), 1);
  }
  else
  {
    v34 = 0;
  }
  v35 = *(_QWORD *)(a1 + 104);
  if ( v35 )
  {
    v36 = *(__int64 **)(a1 + 112);
    sub_157E9D0(v35 + 40, v142);
    v37 = *(_QWORD *)(v142 + 24);
    v38 = *v36 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v142 + 32) = v36;
    *(_QWORD *)(v142 + 24) = v38 | v37 & 7;
    *(_QWORD *)(v38 + 8) = v142 + 24;
    *v36 = *v36 & 7 | (v142 + 24);
  }
  sub_164B780(v34, v157);
  sub_12A86E0(v143, v142);
  sub_1704F80(v142, (__int64)v23, *(_QWORD *)(a1 + 168), v39, v40, v41);
  v42 = 0;
  v43 = sub_1B9C240((unsigned int *)a1, v144, *(_DWORD *)(a1 + 92) - 1);
  v44 = *(_QWORD *)(a1 + 24);
  v45 = *(_DWORD *)(v44 + 24);
  if ( v45 )
  {
    v46 = *(_QWORD *)(a1 + 200);
    v47 = *(_QWORD *)(v44 + 8);
    v48 = v45 - 1;
    v49 = v48 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
    v50 = (__int64 *)(v47 + 16LL * v49);
    v51 = *v50;
    if ( v46 == *v50 )
    {
LABEL_24:
      v42 = v50[1];
    }
    else
    {
      v137 = 1;
      while ( v51 != -8 )
      {
        v139 = v137 + 1;
        v49 = v48 & (v137 + v49);
        v50 = (__int64 *)(v47 + 16LL * v49);
        v51 = *v50;
        if ( v46 == *v50 )
          goto LABEL_24;
        v137 = v139;
      }
      v42 = 0;
    }
  }
  if ( sub_13FC1A0(v42, v43) || *(_BYTE *)(v43 + 16) == 77 )
    v52 = sub_157EE30(*(_QWORD *)(a1 + 200));
  else
    v52 = *(_QWORD *)(v43 + 32);
  if ( v52 )
    v52 -= 24;
  sub_17050D0(v143, v52);
  v55 = *(unsigned int *)(a1 + 88);
  v56 = (__int64 *)s;
  v159 = (__int64 *)s;
  v160 = 0x800000000LL;
  if ( (unsigned int)v55 > 8 )
  {
    sub_16CD150((__int64)&v159, s, v55, 8, v53, v54);
    v56 = v159;
  }
  v57 = 8 * v55;
  LODWORD(v160) = v55;
  v58 = &v56[v55];
  if ( v56 != v58 )
  {
    memset(v56, 0, v57);
    v58 = v159;
  }
  v59 = (unsigned int)(*(_DWORD *)(a1 + 88) - 1);
  v60 = sub_1643350(*(_QWORD **)(a1 + 120));
  *v58 = sub_159C470(v60, v59, 0);
  v61 = *(_DWORD *)(a1 + 88);
  v62 = 1;
  if ( v61 > 1 )
  {
    v152 = v10;
    do
    {
      v63 = &v159[v62];
      v64 = sub_1643350(*(_QWORD **)(a1 + 120));
      v65 = v61 + v62++ - 1;
      *v63 = sub_159C470(v64, v65, 0);
      v61 = *(_DWORD *)(a1 + 88);
    }
    while ( v61 > v62 );
    v10 = v152;
  }
  v153 = v142;
  if ( *(_DWORD *)(a1 + 92) )
  {
    v150 = v10;
    v66 = 0;
    v67 = (_QWORD *)(a1 + 288);
    do
    {
      v68 = v153;
      v153 = sub_1B9C240((unsigned int *)a1, v144, v66);
      v157[0] = v150;
      v69 = sub_1B99AC0(v67, (unsigned __int64 *)v157);
      v72 = v66;
      v73 = *(_QWORD *)(*v69 + 8LL * v66);
      if ( *(_DWORD *)(a1 + 88) > 1u )
      {
        v147 = *(_QWORD *)(*v69 + 8LL * v66);
        v158 = 257;
        v74 = sub_15A01B0(v159, (unsigned int)v160);
        v75 = sub_14C50F0(v143, v68, v153, v74, (__int64)v157);
        v72 = v66;
        v73 = v147;
        v68 = v75;
      }
      ++v66;
      v145 = v72;
      v148 = (_QWORD *)v73;
      sub_164D160(v73, v68, a3, a4, a5, a6, v70, v71, a9, a10);
      sub_15F20C0(v148);
      v157[0] = v150;
      *(_QWORD *)(*sub_1B99AC0(v67, (unsigned __int64 *)v157) + 8 * v145) = v68;
    }
    while ( *(_DWORD *)(a1 + 92) > v66 );
    v10 = v150;
  }
  v76 = *(_QWORD *)(a1 + 24);
  v77 = 0;
  v78 = *(_DWORD *)(v76 + 24);
  if ( v78 )
  {
    v79 = *(_QWORD *)(a1 + 200);
    v80 = *(_QWORD *)(v76 + 8);
    v81 = v78 - 1;
    v82 = v81 & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
    v83 = (__int64 *)(v80 + 16LL * v82);
    v84 = *v83;
    if ( v79 == *v83 )
    {
LABEL_46:
      v77 = v83[1];
    }
    else
    {
      v138 = 1;
      while ( v84 != -8 )
      {
        v140 = v138 + 1;
        v82 = v81 & (v138 + v82);
        v83 = (__int64 *)(v80 + 16LL * v82);
        v84 = *v83;
        if ( v79 == *v83 )
          goto LABEL_46;
        v138 = v140;
      }
      v77 = 0;
    }
  }
  v85 = sub_13FCB50(v77);
  sub_1704F80(v142, v153, v85, v86, v87, v88);
  if ( *(_DWORD *)(a1 + 88) > 1u )
  {
    v124 = sub_157EBA0(*(_QWORD *)(a1 + 184));
    sub_17050D0(v143, v124);
    v125 = *(_QWORD **)(a1 + 120);
    v157[0] = (__int64)"vector.recur.extract";
    v126 = *(_DWORD *)(a1 + 88);
    v158 = 259;
    v127 = (unsigned int)(v126 - 1);
    v128 = sub_1643350(v125);
    v129 = sub_159C470(v128, v127, 0);
    v130 = sub_156D5F0(v143, v153, v129, (__int64)v157);
    v131 = *(_DWORD *)(a1 + 88);
    v132 = v130;
    if ( v131 > 1 )
    {
      v133 = *(_QWORD **)(a1 + 120);
      v157[0] = (__int64)"vector.recur.extract.for.phi";
      v158 = 259;
      v134 = sub_1643350(v133);
      v135 = sub_159C470(v134, v131 - 2, 0);
      v136 = sub_156D5F0(v143, v153, v135, (__int64)v157);
      v153 = v132;
      v146 = v136;
      goto LABEL_50;
    }
    v153 = v130;
  }
  v146 = 0;
  v89 = *(_DWORD *)(a1 + 92);
  if ( v89 > 1 )
    v146 = sub_1B9C240((unsigned int *)a1, v144, v89 - 2);
LABEL_50:
  v90 = *(_QWORD *)(*(_QWORD *)(a1 + 176) + 48LL);
  if ( v90 )
    v90 -= 24;
  sub_17050D0(v143, v90);
  v91 = *(_QWORD *)v10;
  v154 = "scalar.recur.init";
  v156 = 1;
  v155 = 3;
  v158 = 257;
  v151 = sub_1648B60(64);
  if ( v151 )
  {
    v92 = v151;
    sub_15F1EA0(v151, v91, 53, 0, 0, 0);
    *(_DWORD *)(v151 + 56) = 2;
    sub_164B780(v151, v157);
    sub_1648880(v151, *(_DWORD *)(v151 + 56), 1);
  }
  else
  {
    v92 = 0;
  }
  v93 = *(_QWORD *)(a1 + 104);
  if ( v93 )
  {
    v149 = *(__int64 **)(a1 + 112);
    sub_157E9D0(v93 + 40, v151);
    v94 = *v149;
    v95 = *(_QWORD *)(v151 + 24) & 7LL;
    *(_QWORD *)(v151 + 32) = v149;
    v94 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v151 + 24) = v94 | v95;
    *(_QWORD *)(v94 + 8) = v151 + 24;
    *v149 = *v149 & 7 | (v151 + 24);
  }
  sub_164B780(v92, (__int64 *)&v154);
  sub_12A86E0(v143, v151);
  v96 = *(_QWORD *)(a1 + 176);
  v97 = *(_QWORD *)(v96 + 8);
  if ( v97 )
  {
    while ( 1 )
    {
      v98 = sub_1648700(v97);
      v101 = *((unsigned __int8 *)v98 + 16);
      if ( (unsigned __int8)(v101 - 25) <= 9u )
        break;
      v97 = *(_QWORD *)(v97 + 8);
      if ( !v97 )
        goto LABEL_65;
    }
    v102 = v97;
LABEL_60:
    v103 = v98[5];
    v104 = (__int64)v141;
    if ( *(_QWORD *)(a1 + 184) == v103 )
      v104 = v153;
    sub_1704F80(v151, v104, v103, v101, v99, v100);
    while ( 1 )
    {
      v102 = *(_QWORD *)(v102 + 8);
      if ( !v102 )
        break;
      v98 = sub_1648700(v102);
      v101 = *((unsigned __int8 *)v98 + 16);
      if ( (unsigned __int8)(v101 - 25) <= 9u )
        goto LABEL_60;
    }
    v96 = *(_QWORD *)(a1 + 176);
  }
LABEL_65:
  v105 = 0x17FFFFFFE8LL;
  v106 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  if ( v106 )
  {
    v107 = 24LL * *(unsigned int *)(v10 + 56) + 8;
    v108 = 0;
    do
    {
      v109 = v10 - 24LL * v106;
      if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
        v109 = *(_QWORD *)(v10 - 8);
      if ( v96 == *(_QWORD *)(v109 + v107) )
      {
        v105 = 24 * v108;
        goto LABEL_72;
      }
      ++v108;
      v107 += 8;
    }
    while ( v106 != (_DWORD)v108 );
    v105 = 0x17FFFFFFE8LL;
  }
LABEL_72:
  v110 = (__int64 *)(v105 + sub_13CF970(v10));
  if ( *v110 )
  {
    v111 = v110[1];
    v112 = v110[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v112 = v111;
    if ( v111 )
      *(_QWORD *)(v111 + 16) = *(_QWORD *)(v111 + 16) & 3LL | v112;
  }
  *v110 = v151;
  if ( v151 )
  {
    v113 = *(_QWORD *)(v151 + 8);
    v110[1] = v113;
    if ( v113 )
      *(_QWORD *)(v113 + 16) = (unsigned __int64)(v110 + 1) | *(_QWORD *)(v113 + 16) & 3LL;
    v110[2] = (v151 + 8) | v110[2] & 3;
    *(_QWORD *)(v151 + 8) = v110;
  }
  v157[0] = (__int64)"scalar.recur";
  v158 = 259;
  sub_164B780(v10, v157);
  v114 = sub_157F280(*(_QWORD *)(a1 + 192));
  v119 = v118;
  v120 = v114;
  while ( v119 != v120 )
  {
    if ( (*(_BYTE *)(v120 + 23) & 0x40) != 0 )
      v121 = *(__int64 **)(v120 - 8);
    else
      v121 = (__int64 *)(v120 - 24LL * (*(_DWORD *)(v120 + 20) & 0xFFFFFFF));
    v122 = *v121;
    if ( v10 == v122 && v122 )
      sub_1704F80(v120, v146, *(_QWORD *)(a1 + 184), v115, v116, v117);
    v123 = *(_QWORD *)(v120 + 32);
    if ( !v123 )
      BUG();
    v120 = 0;
    if ( *(_BYTE *)(v123 - 8) == 77 )
      v120 = v123 - 24;
  }
  if ( v159 != (__int64 *)s )
    _libc_free((unsigned __int64)v159);
}
