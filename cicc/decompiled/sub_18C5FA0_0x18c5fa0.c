// Function: sub_18C5FA0
// Address: 0x18c5fa0
//
__int64 __fastcall sub_18C5FA0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rax
  char **v4; // rbx
  char **v5; // r14
  __int64 *v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rax
  char *v9; // rdi
  __int64 *v10; // rbx
  __int64 *i; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  char **v14; // rbx
  char **v15; // r14
  __int64 *v16; // rdi
  __int64 (__fastcall *v17)(__int64); // rax
  char *v18; // rdi
  unsigned int j; // r13d
  unsigned int v20; // r14d
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r14
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 *v29; // rax
  __int64 v30; // rax
  char v31; // al
  __int64 v32; // rsi
  __int64 *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // r13
  __int64 v36; // r12
  _QWORD *v37; // rax
  __int64 v38; // r13
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // rbx
  unsigned __int64 *v48; // r12
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // r12
  _QWORD *v54; // rax
  _QWORD *v55; // rbx
  unsigned __int64 *v56; // r12
  __int64 v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // rax
  unsigned __int64 *v64; // rbx
  unsigned __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  __int64 *v69; // rbx
  __int64 v70; // r12
  __int64 v71; // rax
  __int64 v72; // r13
  _QWORD *v73; // rax
  __int64 v74; // r12
  __int64 v75; // r15
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // r15
  unsigned __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rdx
  unsigned __int8 *v83; // rsi
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 *k; // [rsp+28h] [rbp-298h]
  __int64 v90; // [rsp+30h] [rbp-290h]
  __int64 v91; // [rsp+38h] [rbp-288h]
  _QWORD *v92; // [rsp+40h] [rbp-280h]
  __int64 v93; // [rsp+48h] [rbp-278h]
  __int64 v95; // [rsp+60h] [rbp-260h]
  __int64 v96; // [rsp+60h] [rbp-260h]
  unsigned __int64 *v97; // [rsp+60h] [rbp-260h]
  __int64 *v98; // [rsp+68h] [rbp-258h]
  __int64 v99; // [rsp+68h] [rbp-258h]
  int v100; // [rsp+70h] [rbp-250h]
  __int64 v101; // [rsp+70h] [rbp-250h]
  __int64 v102; // [rsp+70h] [rbp-250h]
  __int64 *v103; // [rsp+78h] [rbp-248h]
  __int64 v104; // [rsp+78h] [rbp-248h]
  __int64 v105; // [rsp+78h] [rbp-248h]
  __int64 v106; // [rsp+78h] [rbp-248h]
  _QWORD v107[2]; // [rsp+80h] [rbp-240h] BYREF
  __int64 v108[2]; // [rsp+90h] [rbp-230h] BYREF
  __int16 v109; // [rsp+A0h] [rbp-220h]
  __int64 *v110; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v111; // [rsp+B8h] [rbp-208h]
  _BYTE v112[16]; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v113; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 v114; // [rsp+D8h] [rbp-1E8h]
  __int64 v115; // [rsp+E0h] [rbp-1E0h]
  __int64 v116; // [rsp+E8h] [rbp-1D8h]
  __int64 *v117; // [rsp+F0h] [rbp-1D0h]
  __int64 *v118; // [rsp+F8h] [rbp-1C8h]
  __int64 v119; // [rsp+100h] [rbp-1C0h]
  __int64 *v120; // [rsp+110h] [rbp-1B0h] BYREF
  __int64 v121; // [rsp+120h] [rbp-1A0h] BYREF
  int v122; // [rsp+130h] [rbp-190h]
  unsigned __int8 *v123; // [rsp+150h] [rbp-170h] BYREF
  __int64 v124; // [rsp+158h] [rbp-168h]
  unsigned __int64 *v125; // [rsp+160h] [rbp-160h]
  __int64 v126; // [rsp+168h] [rbp-158h]
  __int64 v127; // [rsp+170h] [rbp-150h]
  int v128; // [rsp+178h] [rbp-148h]
  __int64 v129; // [rsp+180h] [rbp-140h]
  __int64 v130; // [rsp+188h] [rbp-138h]
  unsigned __int8 *v131; // [rsp+1A0h] [rbp-120h] BYREF
  __int64 v132; // [rsp+1A8h] [rbp-118h]
  unsigned __int64 *v133; // [rsp+1B0h] [rbp-110h]
  __int64 v134; // [rsp+1B8h] [rbp-108h]
  __int64 v135; // [rsp+1C0h] [rbp-100h]
  int v136; // [rsp+1C8h] [rbp-F8h]
  __int64 v137; // [rsp+1D0h] [rbp-F0h]
  __int64 v138; // [rsp+1D8h] [rbp-E8h]
  unsigned __int8 *v139; // [rsp+1F0h] [rbp-D0h] BYREF
  __int64 *v140; // [rsp+1F8h] [rbp-C8h]
  __int64 *v141; // [rsp+200h] [rbp-C0h]
  __int64 *v142; // [rsp+208h] [rbp-B8h]
  __int64 v143; // [rsp+210h] [rbp-B0h]
  int v144; // [rsp+218h] [rbp-A8h]
  __int64 v145; // [rsp+220h] [rbp-A0h]
  __int64 v146; // [rsp+228h] [rbp-98h]
  char *v147; // [rsp+240h] [rbp-80h] BYREF
  __int64 v148; // [rsp+248h] [rbp-78h]
  __int64 (__fastcall *v149)(__int64 *); // [rsp+250h] [rbp-70h] BYREF
  __int64 v150; // [rsp+258h] [rbp-68h]
  __int64 v151; // [rsp+260h] [rbp-60h]
  int v152; // [rsp+268h] [rbp-58h]
  __int64 v153; // [rsp+270h] [rbp-50h]
  __int64 v154; // [rsp+278h] [rbp-48h]

  v2 = a2[2];
  v110 = (__int64 *)v112;
  v111 = 0x200000000LL;
  v3 = (__int64 *)a2[4];
  v139 = (unsigned __int8 *)v2;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v98 = a2 + 1;
  v103 = a2 + 3;
  v140 = a2 + 1;
  v141 = v3;
  v142 = a2 + 3;
  if ( a2 + 3 == v3 )
    goto LABEL_21;
  do
  {
    do
    {
      v4 = &v147;
      v150 = 0;
      v5 = &v147;
      v6 = (__int64 *)&v139;
      v149 = sub_18564C0;
      v7 = sub_18564A0;
      if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
        goto LABEL_4;
      while ( 1 )
      {
        v7 = *(__int64 (__fastcall **)(__int64))((char *)v7 + *v6 - 1);
LABEL_4:
        v8 = v7((__int64)v6);
        if ( v8 )
          break;
        while ( 1 )
        {
          v9 = v5[3];
          v7 = (__int64 (__fastcall *)(__int64))v5[2];
          v4 += 2;
          v5 = v4;
          v6 = (__int64 *)((char *)&v139 + (_QWORD)v9);
          if ( ((unsigned __int8)v7 & 1) != 0 )
            break;
          v8 = v7((__int64)v6);
          if ( v8 )
            goto LABEL_7;
        }
      }
LABEL_7:
      LODWORD(v111) = 0;
      sub_1626560(v8, 19, (__int64)&v110);
      v10 = &v110[(unsigned int)v111];
      for ( i = v110; v10 != i; ++i )
      {
        v13 = sub_18C5B40(*i);
        if ( v13 )
        {
          if ( *(_DWORD *)(v13 + 32) > 0x40u )
            v12 = **(_QWORD **)(v13 + 24);
          else
            v12 = *(_QWORD *)(v13 + 24);
          v147 = (char *)v12;
          sub_18C5D50((__int64)&v113, &v147);
        }
      }
      v14 = &v147;
      v150 = 0;
      v15 = &v147;
      v16 = (__int64 *)&v139;
      v149 = sub_1856470;
      v17 = sub_1856440;
      if ( ((unsigned __int8)sub_1856440 & 1) != 0 )
        goto LABEL_16;
      while ( 2 )
      {
        if ( !(unsigned __int8)v17((__int64)v16) )
        {
          while ( 1 )
          {
            v18 = v15[3];
            v17 = (__int64 (__fastcall *)(__int64))v15[2];
            v14 += 2;
            v15 = v14;
            v16 = (__int64 *)((char *)&v139 + (_QWORD)v18);
            if ( ((unsigned __int8)v17 & 1) != 0 )
              break;
            if ( (unsigned __int8)v17((__int64)v16) )
              goto LABEL_20;
          }
LABEL_16:
          v17 = *(__int64 (__fastcall **)(__int64))((char *)v17 + *v16 - 1);
          continue;
        }
        break;
      }
LABEL_20:
      ;
    }
    while ( v103 != v141 );
LABEL_21:
    ;
  }
  while ( v103 != v142 || v98 != (__int64 *)v139 || v98 != v140 );
  v147 = "cfi.functions";
  LOWORD(v149) = 259;
  v104 = sub_1632310((__int64)a2, (__int64)&v147);
  if ( v104 )
  {
    v100 = sub_161F520(v104);
    if ( v100 )
    {
      for ( j = 0; j != v100; ++j )
      {
        v20 = 2;
        v21 = sub_161F530(v104, j);
        v22 = *(_DWORD *)(v21 + 8);
        v23 = v21;
        if ( v22 > 2 )
        {
          do
          {
            v25 = sub_18C5B40(*(_QWORD *)(v23 + 8 * (v20 - (unsigned __int64)v22)));
            if ( v25 )
            {
              if ( *(_DWORD *)(v25 + 32) > 0x40u )
                v24 = **(_QWORD **)(v25 + 24);
              else
                v24 = *(_QWORD *)(v25 + 24);
              v147 = (char *)v24;
              sub_18C5D50((__int64)&v113, &v147);
              v22 = *(_DWORD *)(v23 + 8);
            }
            ++v20;
          }
          while ( v20 < v22 );
        }
      }
    }
  }
  v26 = (__int64 *)*a2;
  v27 = sub_16471D0((_QWORD *)*a2, 0);
  v28 = sub_16471D0(v26, 0);
  v105 = sub_1643360(v26);
  v29 = (__int64 *)sub_1643270(v26);
  v147 = (char *)&v149;
  v150 = v28;
  v151 = v27;
  v149 = (__int64 (__fastcall *)(__int64 *))v105;
  v148 = 0x300000003LL;
  v30 = sub_1644EA0(v29, &v149, 3, 0);
  v99 = sub_1632080((__int64)a2, (__int64)"__cfi_check", 11, v30, 0);
  if ( v147 != (char *)&v149 )
    _libc_free((unsigned __int64)v147);
  if ( *(_BYTE *)(v99 + 16) )
  {
    sub_15E0C30(0);
    MEMORY[0x20] &= 0xFFFFFFF0;
    BUG();
  }
  sub_15E0C30(v99);
  v31 = *(_BYTE *)(v99 + 32);
  *(_BYTE *)(v99 + 32) = v31 & 0xF0;
  if ( (v31 & 0x30) != 0 )
    *(_BYTE *)(v99 + 33) |= 0x40u;
  sub_15E4CC0(v99, 0x1000u);
  v32 = (__int64)&v147;
  LOWORD(v149) = 260;
  v147 = (char *)(a2 + 30);
  sub_16E1010((__int64)&v120, (__int64)&v147);
  if ( (unsigned int)(v122 - 1) <= 1 || (unsigned int)(v122 - 29) <= 1 )
  {
    v33 = (__int64 *)sub_15E0530(v99);
    v34 = sub_155D020(v33, "target-features", 0xFu, "+thumb-mode", 0xBu);
    v32 = 0xFFFFFFFFLL;
    sub_15E0DA0(v99, -1, (__int64)v34);
  }
  if ( (*(_BYTE *)(v99 + 18) & 1) != 0 )
    sub_15E08E0(v99, v32);
  v35 = *(_QWORD *)(v99 + 88);
  v147 = "CallSiteTypeId";
  v106 = v35;
  v36 = v35 + 80;
  LOWORD(v149) = 259;
  sub_164B780(v35, (__int64 *)&v147);
  v147 = "Addr";
  v90 = v35 + 40;
  LOWORD(v149) = 259;
  sub_164B780(v35 + 40, (__int64 *)&v147);
  LOWORD(v149) = 259;
  v147 = "CFICheckFailData";
  sub_164B780(v35 + 80, (__int64 *)&v147);
  v147 = "entry";
  LOWORD(v149) = 259;
  v37 = (_QWORD *)sub_22077B0(64);
  v38 = (__int64)v37;
  if ( v37 )
    sub_157FB60(v37, (__int64)v26, (__int64)&v147, v99, 0);
  v147 = "exit";
  LOWORD(v149) = 259;
  v39 = (_QWORD *)sub_22077B0(64);
  v91 = (__int64)v39;
  if ( v39 )
    sub_157FB60(v39, (__int64)v26, (__int64)&v147, v99, 0);
  v147 = "fail";
  LOWORD(v149) = 259;
  v40 = (_QWORD *)sub_22077B0(64);
  v93 = (__int64)v40;
  if ( v40 )
    sub_157FB60(v40, (__int64)v26, (__int64)&v147, v99, 0);
  v41 = sub_157E9C0(v93);
  v123 = 0;
  v126 = v41;
  v127 = 0;
  v124 = v93;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v125 = (unsigned __int64 *)(v93 + 40);
  v95 = sub_16471D0(v26, 0);
  v101 = sub_16471D0(v26, 0);
  v42 = (__int64 *)sub_1643270(v26);
  v147 = (char *)&v149;
  v149 = (__int64 (__fastcall *)(__int64 *))v101;
  v150 = v95;
  v148 = 0x200000002LL;
  v43 = sub_1644EA0(v42, &v149, 2, 0);
  v44 = sub_1632080((__int64)a2, (__int64)"__cfi_check_fail", 16, v43, 0);
  v45 = v44;
  if ( v147 != (char *)&v149 )
  {
    v102 = v44;
    _libc_free((unsigned __int64)v147);
    v45 = v102;
  }
  v139 = (unsigned __int8 *)v36;
  LOWORD(v149) = 257;
  v140 = (__int64 *)v90;
  sub_1285290((__int64 *)&v123, *(_QWORD *)(*(_QWORD *)v45 + 24LL), v45, (int)&v139, 2, (__int64)&v147, 0);
  LOWORD(v149) = 257;
  v46 = sub_1648A60(56, 1u);
  v47 = v46;
  if ( v46 )
    sub_15F8320((__int64)v46, v91, 0);
  if ( v124 )
  {
    v48 = v125;
    sub_157E9D0(v124 + 40, (__int64)v47);
    v49 = v47[3];
    v50 = *v48;
    v47[4] = v48;
    v50 &= 0xFFFFFFFFFFFFFFF8LL;
    v47[3] = v50 | v49 & 7;
    *(_QWORD *)(v50 + 8) = v47 + 3;
    *v48 = *v48 & 7 | (unsigned __int64)(v47 + 3);
  }
  sub_164B780((__int64)v47, (__int64 *)&v147);
  if ( v123 )
  {
    v139 = v123;
    sub_1623A60((__int64)&v139, (__int64)v123, 2);
    v51 = v47[6];
    if ( v51 )
      sub_161E7C0((__int64)(v47 + 6), v51);
    v52 = v139;
    v47[6] = v139;
    if ( v52 )
      sub_1623210((__int64)&v139, v52, (__int64)(v47 + 6));
  }
  v134 = sub_157E9C0(v91);
  v53 = v134;
  v132 = v91;
  v131 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v133 = (unsigned __int64 *)(v91 + 40);
  LOWORD(v149) = 257;
  v54 = sub_1648A60(56, 0);
  v55 = v54;
  if ( v54 )
    sub_15F6F90((__int64)v54, v53, 0, 0);
  if ( v132 )
  {
    v56 = v133;
    sub_157E9D0(v132 + 40, (__int64)v55);
    v57 = v55[3];
    v58 = *v56;
    v55[4] = v56;
    v58 &= 0xFFFFFFFFFFFFFFF8LL;
    v55[3] = v58 | v57 & 7;
    *(_QWORD *)(v58 + 8) = v55 + 3;
    *v56 = *v56 & 7 | (unsigned __int64)(v55 + 3);
  }
  sub_164B780((__int64)v55, (__int64 *)&v147);
  if ( v131 )
  {
    v139 = v131;
    sub_1623A60((__int64)&v139, (__int64)v131, 2);
    v59 = v55[6];
    if ( v59 )
      sub_161E7C0((__int64)(v55 + 6), v59);
    v60 = v139;
    v55[6] = v139;
    if ( v60 )
      sub_1623210((__int64)&v139, v60, (__int64)(v55 + 6));
  }
  v61 = sub_157E9C0(v38);
  v140 = (__int64 *)v38;
  LOWORD(v149) = 257;
  v142 = (__int64 *)v61;
  v139 = 0;
  v62 = v118 - v117;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v141 = (__int64 *)(v38 + 40);
  v63 = sub_1648B60(64);
  v92 = (_QWORD *)v63;
  if ( v63 )
    sub_15FFAB0(v63, v106, v93, v62, 0);
  if ( v140 )
  {
    v64 = (unsigned __int64 *)v141;
    sub_157E9D0((__int64)(v140 + 5), (__int64)v92);
    v65 = *v64;
    v66 = v92[3];
    v92[4] = v64;
    v65 &= 0xFFFFFFFFFFFFFFF8LL;
    v92[3] = v65 | v66 & 7;
    *(_QWORD *)(v65 + 8) = v92 + 3;
    *v64 = *v64 & 7 | (unsigned __int64)(v92 + 3);
  }
  sub_164B780((__int64)v92, (__int64 *)&v147);
  if ( v139 )
  {
    v108[0] = (__int64)v139;
    sub_1623A60((__int64)v108, (__int64)v139, 2);
    v67 = v92[6];
    if ( v67 )
      sub_161E7C0((__int64)(v92 + 6), v67);
    v68 = (unsigned __int8 *)v108[0];
    v92[6] = v108[0];
    if ( v68 )
      sub_1623210((__int64)v108, v68, (__int64)(v92 + 6));
  }
  v69 = v117;
  for ( k = v118; k != v69; ++v69 )
  {
    v70 = *v69;
    v71 = sub_1643360(v26);
    v72 = sub_159C470(v71, v70, 0);
    LOWORD(v149) = 259;
    v147 = "test";
    v73 = (_QWORD *)sub_22077B0(64);
    v74 = (__int64)v73;
    if ( v73 )
      sub_157FB60(v73, (__int64)v26, (__int64)&v147, v99, 0);
    v150 = sub_157E9C0(v74);
    v147 = 0;
    v151 = 0;
    v152 = 0;
    v153 = 0;
    v154 = 0;
    v148 = v74;
    v149 = (__int64 (__fastcall *)(__int64 *))(v74 + 40);
    v75 = sub_15E26F0(a2, 208, 0, 0);
    v109 = 257;
    v107[0] = v90;
    v76 = sub_1624210(v72);
    v107[1] = sub_1628DA0(v26, (__int64)v76);
    v96 = sub_1285290((__int64 *)&v147, *(_QWORD *)(v75 + 24), v75, (int)v107, 2, (__int64)v108, 0);
    v109 = 257;
    v77 = sub_1648A60(56, 3u);
    v78 = v77;
    if ( v77 )
      sub_15F83E0((__int64)v77, v91, v93, v96, 0);
    if ( v148 )
    {
      v97 = (unsigned __int64 *)v149;
      sub_157E9D0(v148 + 40, (__int64)v78);
      v79 = *v97;
      v80 = v78[3] & 7LL;
      v78[4] = v97;
      v79 &= 0xFFFFFFFFFFFFFFF8LL;
      v78[3] = v79 | v80;
      *(_QWORD *)(v79 + 8) = v78 + 3;
      *v97 = *v97 & 7 | (unsigned __int64)(v78 + 3);
    }
    sub_164B780((__int64)v78, v108);
    if ( v147 )
    {
      v107[0] = v147;
      sub_1623A60((__int64)v107, (__int64)v147, 2);
      v81 = v78[6];
      v82 = (__int64)(v78 + 6);
      if ( v81 )
      {
        sub_161E7C0((__int64)(v78 + 6), v81);
        v82 = (__int64)(v78 + 6);
      }
      v83 = (unsigned __int8 *)v107[0];
      v78[6] = v107[0];
      if ( v83 )
        sub_1623210((__int64)v107, v83, v82);
    }
    sub_1625C10((__int64)v78, 2, *(_QWORD *)(a1 + 160));
    sub_15FFFB0((__int64)v92, v72, v74, v84, v85, v86);
    if ( v147 )
      sub_161E7C0((__int64)&v147, (__int64)v147);
  }
  if ( v139 )
    sub_161E7C0((__int64)&v139, (__int64)v139);
  if ( v131 )
    sub_161E7C0((__int64)&v131, (__int64)v131);
  if ( v123 )
    sub_161E7C0((__int64)&v123, (__int64)v123);
  if ( v120 != &v121 )
    j_j___libc_free_0(v120, v121 + 1);
  if ( v110 != (__int64 *)v112 )
    _libc_free((unsigned __int64)v110);
  if ( v117 )
    j_j___libc_free_0(v117, v119 - (_QWORD)v117);
  return j___libc_free_0(v114);
}
