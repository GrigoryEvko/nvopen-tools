// Function: sub_17B9360
// Address: 0x17b9360
//
void __fastcall sub_17B9360(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // r15
  _QWORD *v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r14
  __int64 v9; // r13
  _QWORD *v10; // rax
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // rdi
  __int64 **v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rbx
  _QWORD *v30; // r15
  _QWORD *v31; // r13
  unsigned __int64 *v32; // rbx
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 *v37; // rax
  __int64 **v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r14
  _QWORD *v44; // rax
  _QWORD *v45; // rbx
  unsigned __int64 *v46; // r14
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // r14
  _QWORD *v54; // r15
  unsigned __int64 *v55; // rbx
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 v60; // rax
  __int64 v61; // r14
  _QWORD *v62; // rax
  _QWORD *v63; // rbx
  unsigned __int64 *v64; // r13
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  _QWORD *v69; // rax
  _QWORD *v70; // rbx
  unsigned __int64 *v71; // r13
  __int64 v72; // rax
  unsigned __int64 v73; // rcx
  __int64 v74; // rsi
  unsigned __int8 *v75; // rsi
  _QWORD *v76; // r13
  _QWORD *v77; // rax
  _QWORD *v78; // rbx
  unsigned __int64 *v79; // r13
  __int64 v80; // rax
  unsigned __int64 v81; // rcx
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  _QWORD *v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 *v87; // rax
  __int64 *v88; // r10
  __int64 v89; // rax
  unsigned __int64 *v90; // r13
  __int64 v91; // rax
  unsigned __int64 v92; // rcx
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  __int64 v95; // rax
  __int64 *v96; // r14
  __int64 v97; // rax
  __int64 v98; // rcx
  __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // [rsp+0h] [rbp-110h]
  int v102; // [rsp+Ch] [rbp-104h]
  __int64 v103; // [rsp+10h] [rbp-100h]
  __int64 v104; // [rsp+10h] [rbp-100h]
  __int64 v105; // [rsp+10h] [rbp-100h]
  __int64 v106; // [rsp+18h] [rbp-F8h]
  __int64 v107; // [rsp+28h] [rbp-E8h]
  __int64 v109; // [rsp+30h] [rbp-E0h]
  __int64 v110; // [rsp+38h] [rbp-D8h]
  __int64 *v111; // [rsp+38h] [rbp-D8h]
  __int64 *v112; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int8 *v113; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v114[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v115; // [rsp+60h] [rbp-B0h]
  __int64 v116[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v117; // [rsp+80h] [rbp-90h]
  __int64 *v118; // [rsp+90h] [rbp-80h] BYREF
  _QWORD *v119; // [rsp+98h] [rbp-78h]
  __int64 *v120; // [rsp+A0h] [rbp-70h]
  _QWORD *v121; // [rsp+A8h] [rbp-68h]
  __int64 v122; // [rsp+B0h] [rbp-60h]
  int v123; // [rsp+B8h] [rbp-58h]
  __int64 v124; // [rsp+C0h] [rbp-50h]
  __int64 v125; // [rsp+C8h] [rbp-48h]

  v1 = 0xFFFFFFFFLL;
  v2 = sub_17B6EA0(a1);
  *(_WORD *)(v2 + 32) = *(_WORD *)(v2 + 32) & 0xBF00 | 0x4087;
  sub_15E0D50(v2, -1, 26);
  if ( *(_BYTE *)(a1 + 7) )
  {
    v1 = 0xFFFFFFFFLL;
    sub_15E0D50(v2, -1, 28);
  }
  v118 = (__int64 *)"entry";
  LOWORD(v120) = 259;
  v3 = *(_QWORD *)(a1 + 64);
  v4 = (_QWORD *)sub_22077B0(64);
  if ( v4 )
  {
    v1 = v3;
    sub_157FB60(v4, v3, (__int64)&v118, v2, 0);
  }
  v5 = (_QWORD *)sub_157E9C0((__int64)v4);
  v119 = v4;
  v121 = v5;
  v120 = v4 + 5;
  v117 = 257;
  v118 = 0;
  v122 = 0;
  v6 = *(_QWORD *)(a1 + 64);
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v7 = (_QWORD *)sub_22077B0(64);
  v8 = (__int64)v7;
  if ( v7 )
  {
    v1 = v6;
    sub_157FB60(v7, v6, (__int64)v116, v2, 0);
  }
  v117 = 257;
  v9 = *(_QWORD *)(a1 + 64);
  v10 = (_QWORD *)sub_22077B0(64);
  v106 = (__int64)v10;
  if ( v10 )
  {
    v1 = v9;
    sub_157FB60(v10, v9, (__int64)v116, v2, 0);
  }
  v116[0] = (__int64)"exit";
  v117 = 259;
  v11 = *(_QWORD *)(a1 + 64);
  v12 = (_QWORD *)sub_22077B0(64);
  v107 = (__int64)v12;
  if ( v12 )
  {
    v1 = v11;
    sub_157FB60(v12, v11, (__int64)v116, v2, 0);
  }
  if ( (*(_BYTE *)(v2 + 18) & 1) != 0 )
    sub_15E08E0(v2, v1);
  v13 = *(_QWORD *)(v2 + 88);
  v116[0] = (__int64)"predecessor";
  v110 = v13;
  v117 = 259;
  sub_164B780(v13, v116);
  v116[0] = (__int64)"pred";
  v117 = 259;
  v14 = sub_1648A60(64, 1u);
  v15 = (__int64)v14;
  if ( v14 )
    sub_15F9210((__int64)v14, *(_QWORD *)(*(_QWORD *)v110 + 24LL), v110, 0, 0, 0);
  if ( v119 )
  {
    v111 = v120;
    sub_157E9D0((__int64)(v119 + 5), v15);
    v16 = *v111;
    v17 = *(_QWORD *)(v15 + 24) & 7LL;
    *(_QWORD *)(v15 + 32) = v111;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v15 + 24) = v16 | v17;
    *(_QWORD *)(v16 + 8) = v15 + 24;
    *v111 = *v111 & 7 | (v15 + 24);
  }
  sub_164B780(v15, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v18 = *(_QWORD *)(v15 + 48);
    v19 = v15 + 48;
    if ( v18 )
    {
      sub_161E7C0(v15 + 48, v18);
      v19 = v15 + 48;
    }
    v20 = (unsigned __int8 *)v114[0];
    *(_QWORD *)(v15 + 48) = v114[0];
    if ( v20 )
      sub_1623210((__int64)v114, v20, v19);
  }
  v117 = 257;
  v21 = sub_1643350(v121);
  v22 = sub_159C470(v21, 0xFFFFFFFFLL, 0);
  v23 = 3;
  v103 = sub_12AA0C0((__int64 *)&v118, 0x20u, (_BYTE *)v15, v22, (__int64)v116);
  v24 = sub_1648A60(56, 3u);
  if ( v24 )
  {
    v23 = v107;
    sub_15F8650((__int64)v24, v107, v8, v103, (__int64)v4);
  }
  v119 = (_QWORD *)v8;
  v120 = (__int64 *)(v8 + 40);
  v115 = 257;
  v25 = (__int64 **)sub_1643360(v121);
  if ( v25 != *(__int64 ***)v15 )
  {
    if ( *(_BYTE *)(v15 + 16) > 0x10u )
    {
      v117 = 257;
      v95 = sub_15FDBD0(37, v15, (__int64)v25, (__int64)v116, 0);
      v15 = v95;
      if ( v119 )
      {
        v96 = v120;
        sub_157E9D0((__int64)(v119 + 5), v95);
        v97 = *(_QWORD *)(v15 + 24);
        v98 = *v96;
        *(_QWORD *)(v15 + 32) = v96;
        v98 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v15 + 24) = v98 | v97 & 7;
        *(_QWORD *)(v98 + 8) = v15 + 24;
        *v96 = *v96 & 7 | (v15 + 24);
      }
      sub_164B780(v15, v114);
      v23 = (__int64)v118;
      if ( v118 )
      {
        v113 = (unsigned __int8 *)v118;
        sub_1623A60((__int64)&v113, (__int64)v118, 2);
        v99 = *(_QWORD *)(v15 + 48);
        if ( v99 )
          sub_161E7C0(v15 + 48, v99);
        v23 = (__int64)v113;
        *(_QWORD *)(v15 + 48) = v113;
        if ( v23 )
          sub_1623210((__int64)&v113, (unsigned __int8 *)v23, v15 + 48);
      }
    }
    else
    {
      v23 = v15;
      v15 = sub_15A46C0(37, (__int64 ***)v15, v25, 0);
    }
  }
  if ( (*(_BYTE *)(v2 + 18) & 1) != 0 )
    sub_15E08E0(v2, v23);
  v26 = *(_QWORD *)(v2 + 88);
  v116[0] = (__int64)"counters";
  v27 = v26 + 40;
  v104 = v26;
  v117 = 259;
  sub_164B780(v26 + 40, v116);
  v115 = 257;
  v28 = sub_1647230(*(_QWORD **)(a1 + 64), 0);
  v112 = (__int64 *)v15;
  v29 = v28;
  if ( *(_BYTE *)(v104 + 56) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
  {
    v117 = 257;
    if ( !v28 )
    {
      v100 = *(_QWORD *)(v104 + 40);
      if ( *(_BYTE *)(v100 + 8) == 16 )
        v100 = **(_QWORD **)(v100 + 16);
      v29 = *(_QWORD *)(v100 + 24);
    }
    v84 = sub_1648A60(72, 2u);
    v85 = v104;
    v30 = v84;
    if ( v84 )
    {
      v109 = (__int64)v84;
      v105 = (__int64)(v84 - 6);
      v86 = *(_QWORD *)(v85 + 40);
      if ( *(_BYTE *)(v86 + 8) == 16 )
        v86 = **(_QWORD **)(v86 + 16);
      v101 = v85;
      v102 = *(_DWORD *)(v86 + 8) >> 8;
      v87 = (__int64 *)sub_15F9F50(v29, (__int64)&v112, 1);
      v88 = (__int64 *)sub_1646BA0(v87, v102);
      v89 = *(_QWORD *)(v101 + 40);
      if ( *(_BYTE *)(v89 + 8) == 16 || (v89 = *v112, *(_BYTE *)(*v112 + 8) == 16) )
        v88 = sub_16463B0(v88, *(_QWORD *)(v89 + 32));
      sub_15F1EA0((__int64)v30, (__int64)v88, 32, v105, 2, 0);
      v30[7] = v29;
      v30[8] = sub_15F9F50(v29, (__int64)&v112, 1);
      sub_15F9CE0((__int64)v30, v27, (__int64 *)&v112, 1, (__int64)v116);
    }
    else
    {
      v109 = 0;
    }
    if ( v119 )
    {
      v90 = (unsigned __int64 *)v120;
      sub_157E9D0((__int64)(v119 + 5), (__int64)v30);
      v91 = v30[3];
      v92 = *v90;
      v30[4] = v90;
      v92 &= 0xFFFFFFFFFFFFFFF8LL;
      v30[3] = v92 | v91 & 7;
      *(_QWORD *)(v92 + 8) = v30 + 3;
      *v90 = *v90 & 7 | (unsigned __int64)(v30 + 3);
    }
    sub_164B780(v109, v114);
    if ( v118 )
    {
      v113 = (unsigned __int8 *)v118;
      sub_1623A60((__int64)&v113, (__int64)v118, 2);
      v93 = v30[6];
      if ( v93 )
        sub_161E7C0((__int64)(v30 + 6), v93);
      v94 = v113;
      v30[6] = v113;
      if ( v94 )
        sub_1623210((__int64)&v113, v94, (__int64)(v30 + 6));
    }
  }
  else
  {
    BYTE4(v116[0]) = 0;
    v113 = (unsigned __int8 *)v15;
    v30 = (_QWORD *)sub_15A2E80(v28, v27, (__int64 **)&v113, 1u, 0, (__int64)v116, 0);
  }
  v116[0] = (__int64)"counter";
  v117 = 259;
  v31 = sub_1648A60(64, 1u);
  if ( v31 )
    sub_15F9210((__int64)v31, *(_QWORD *)(*v30 + 24LL), (__int64)v30, 0, 0, 0);
  if ( v119 )
  {
    v32 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v31);
    v33 = v31[3];
    v34 = *v32;
    v31[4] = v32;
    v34 &= 0xFFFFFFFFFFFFFFF8LL;
    v31[3] = v34 | v33 & 7;
    *(_QWORD *)(v34 + 8) = v31 + 3;
    *v32 = *v32 & 7 | (unsigned __int64)(v31 + 3);
  }
  sub_164B780((__int64)v31, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v35 = v31[6];
    if ( v35 )
      sub_161E7C0((__int64)(v31 + 6), v35);
    v36 = (unsigned __int8 *)v114[0];
    v31[6] = v114[0];
    if ( v36 )
      sub_1623210((__int64)v114, v36, (__int64)(v31 + 6));
  }
  v117 = 257;
  v37 = (__int64 *)sub_1643360(v121);
  v38 = (__int64 **)sub_1647190(v37, 0);
  v41 = sub_15A06D0(v38, 0, v39, v40);
  v42 = sub_12AA0C0((__int64 *)&v118, 0x20u, v31, v41, (__int64)v116);
  v117 = 257;
  v43 = v42;
  v44 = sub_1648A60(56, 3u);
  v45 = v44;
  if ( v44 )
    sub_15F83E0((__int64)v44, v107, v106, v43, 0);
  if ( v119 )
  {
    v46 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v45);
    v47 = v45[3];
    v48 = *v46;
    v45[4] = v46;
    v48 &= 0xFFFFFFFFFFFFFFF8LL;
    v45[3] = v48 | v47 & 7;
    *(_QWORD *)(v48 + 8) = v45 + 3;
    *v46 = *v46 & 7 | (unsigned __int64)(v45 + 3);
  }
  sub_164B780((__int64)v45, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v49 = v45[6];
    if ( v49 )
      sub_161E7C0((__int64)(v45 + 6), v49);
    v50 = (unsigned __int8 *)v114[0];
    v45[6] = v114[0];
    if ( v50 )
      sub_1623210((__int64)v114, v50, (__int64)(v45 + 6));
  }
  v117 = 257;
  v119 = (_QWORD *)v106;
  v120 = (__int64 *)(v106 + 40);
  v51 = sub_1643360(v121);
  v52 = sub_159C470(v51, 1, 0);
  v115 = 257;
  v53 = v52;
  v54 = sub_1648A60(64, 1u);
  if ( v54 )
    sub_15F9210((__int64)v54, *(_QWORD *)(*v31 + 24LL), (__int64)v31, 0, 0, 0);
  if ( v119 )
  {
    v55 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v54);
    v56 = v54[3];
    v57 = *v55;
    v54[4] = v55;
    v57 &= 0xFFFFFFFFFFFFFFF8LL;
    v54[3] = v57 | v56 & 7;
    *(_QWORD *)(v57 + 8) = v54 + 3;
    *v55 = *v55 & 7 | (unsigned __int64)(v54 + 3);
  }
  sub_164B780((__int64)v54, v114);
  if ( v118 )
  {
    v113 = (unsigned __int8 *)v118;
    sub_1623A60((__int64)&v113, (__int64)v118, 2);
    v58 = v54[6];
    if ( v58 )
      sub_161E7C0((__int64)(v54 + 6), v58);
    v59 = v113;
    v54[6] = v113;
    if ( v59 )
      sub_1623210((__int64)&v113, v59, (__int64)(v54 + 6));
  }
  v60 = sub_12899C0((__int64 *)&v118, (__int64)v54, v53, (__int64)v116, 0, 0);
  v117 = 257;
  v61 = v60;
  v62 = sub_1648A60(64, 2u);
  v63 = v62;
  if ( v62 )
    sub_15F9650((__int64)v62, v61, (__int64)v31, 0, 0);
  if ( v119 )
  {
    v64 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v63);
    v65 = v63[3];
    v66 = *v64;
    v63[4] = v64;
    v66 &= 0xFFFFFFFFFFFFFFF8LL;
    v63[3] = v66 | v65 & 7;
    *(_QWORD *)(v66 + 8) = v63 + 3;
    *v64 = *v64 & 7 | (unsigned __int64)(v63 + 3);
  }
  sub_164B780((__int64)v63, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v67 = v63[6];
    if ( v67 )
      sub_161E7C0((__int64)(v63 + 6), v67);
    v68 = (unsigned __int8 *)v114[0];
    v63[6] = v114[0];
    if ( v68 )
      sub_1623210((__int64)v114, v68, (__int64)(v63 + 6));
  }
  v117 = 257;
  v69 = sub_1648A60(56, 1u);
  v70 = v69;
  if ( v69 )
    sub_15F8320((__int64)v69, v107, 0);
  if ( v119 )
  {
    v71 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v70);
    v72 = v70[3];
    v73 = *v71;
    v70[4] = v71;
    v73 &= 0xFFFFFFFFFFFFFFF8LL;
    v70[3] = v73 | v72 & 7;
    *(_QWORD *)(v73 + 8) = v70 + 3;
    *v71 = *v71 & 7 | (unsigned __int64)(v70 + 3);
  }
  sub_164B780((__int64)v70, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v74 = v70[6];
    if ( v74 )
      sub_161E7C0((__int64)(v70 + 6), v74);
    v75 = (unsigned __int8 *)v114[0];
    v70[6] = v114[0];
    if ( v75 )
      sub_1623210((__int64)v114, v75, (__int64)(v70 + 6));
  }
  v76 = v121;
  v119 = (_QWORD *)v107;
  v120 = (__int64 *)(v107 + 40);
  v117 = 257;
  v77 = sub_1648A60(56, 0);
  v78 = v77;
  if ( v77 )
    sub_15F6F90((__int64)v77, (__int64)v76, 0, 0);
  if ( v119 )
  {
    v79 = (unsigned __int64 *)v120;
    sub_157E9D0((__int64)(v119 + 5), (__int64)v78);
    v80 = v78[3];
    v81 = *v79;
    v78[4] = v79;
    v81 &= 0xFFFFFFFFFFFFFFF8LL;
    v78[3] = v81 | v80 & 7;
    *(_QWORD *)(v81 + 8) = v78 + 3;
    *v79 = *v79 & 7 | (unsigned __int64)(v78 + 3);
  }
  sub_164B780((__int64)v78, v116);
  if ( v118 )
  {
    v114[0] = (__int64)v118;
    sub_1623A60((__int64)v114, (__int64)v118, 2);
    v82 = v78[6];
    if ( v82 )
      sub_161E7C0((__int64)(v78 + 6), v82);
    v83 = (unsigned __int8 *)v114[0];
    v78[6] = v114[0];
    if ( v83 )
      sub_1623210((__int64)v114, v83, (__int64)(v78 + 6));
    if ( v118 )
      sub_161E7C0((__int64)&v118, (__int64)v118);
  }
}
