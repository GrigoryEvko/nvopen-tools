// Function: sub_196A9C0
// Address: 0x196a9c0
//
void __fastcall sub_196A9C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  unsigned __int8 *v13; // rsi
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 *v16; // r14
  unsigned __int8 **v17; // r15
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  char v20; // di
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // r15
  unsigned int v27; // ebx
  unsigned __int8 *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // r9
  __int64 *v32; // rax
  __int64 v33; // r8
  _QWORD *v34; // r15
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // r12
  __int64 v40; // r12
  unsigned __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 v47; // r15
  unsigned __int8 *v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r13
  int v57; // eax
  __int64 v58; // rax
  int v59; // edx
  __int64 v60; // rdx
  __int64 **v61; // rax
  __int64 *v62; // rcx
  unsigned __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rdx
  int v69; // eax
  __int64 v70; // rax
  int v71; // edx
  __int64 v72; // rdx
  __int64 *v73; // rax
  __int64 v74; // rcx
  unsigned __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int16 v80; // cx
  bool v81; // zf
  __int64 v82; // rdx
  unsigned __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rcx
  unsigned __int64 v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rsi
  unsigned __int8 *v90; // rsi
  __int64 v91; // rsi
  unsigned __int8 *v92; // rsi
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  _QWORD *v95; // rax
  __int64 v96; // r9
  _QWORD **v97; // rax
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 v100; // r9
  __int16 v101; // r11
  __int64 v102; // r8
  unsigned __int64 v103; // rsi
  __int64 v104; // rax
  __int64 v105; // rsi
  __int64 v106; // rdx
  unsigned __int8 *v107; // rsi
  __int64 v108; // [rsp+0h] [rbp-130h]
  __int16 v109; // [rsp+8h] [rbp-128h]
  __int64 v110; // [rsp+8h] [rbp-128h]
  __int64 v111; // [rsp+10h] [rbp-120h]
  __int64 v112; // [rsp+10h] [rbp-120h]
  __int16 v113; // [rsp+10h] [rbp-120h]
  __int16 v114; // [rsp+18h] [rbp-118h]
  _QWORD *v115; // [rsp+18h] [rbp-118h]
  unsigned __int64 *v116; // [rsp+18h] [rbp-118h]
  __int64 v117; // [rsp+18h] [rbp-118h]
  unsigned __int64 v118; // [rsp+20h] [rbp-110h]
  __int64 v119; // [rsp+28h] [rbp-108h]
  __int64 v120; // [rsp+30h] [rbp-100h]
  __int64 v121; // [rsp+30h] [rbp-100h]
  __int64 v123; // [rsp+40h] [rbp-F0h]
  __int64 v125; // [rsp+48h] [rbp-E8h]
  __int64 v126; // [rsp+48h] [rbp-E8h]
  __int64 v127; // [rsp+48h] [rbp-E8h]
  unsigned __int8 *v129; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v130[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v131; // [rsp+80h] [rbp-B0h]
  unsigned __int8 *v132[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v133; // [rsp+A0h] [rbp-90h]
  unsigned __int8 *v134; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v135; // [rsp+B8h] [rbp-78h]
  unsigned __int64 *v136; // [rsp+C0h] [rbp-70h]
  __int64 v137; // [rsp+C8h] [rbp-68h]
  __int64 v138; // [rsp+D0h] [rbp-60h]
  int v139; // [rsp+D8h] [rbp-58h]
  __int64 v140; // [rsp+E0h] [rbp-50h]
  __int64 v141; // [rsp+E8h] [rbp-48h]

  v119 = sub_13FC520(*a1);
  v6 = sub_157EBA0(a2);
  v7 = sub_16498A0(v6);
  v140 = 0;
  v141 = 0;
  v8 = *(unsigned __int8 **)(v6 + 48);
  v137 = v7;
  v139 = 0;
  v9 = *(_QWORD *)(v6 + 40);
  v134 = 0;
  v135 = v9;
  v138 = 0;
  v136 = (unsigned __int64 *)(v6 + 24);
  v132[0] = v8;
  if ( v8 )
  {
    sub_1623A60((__int64)v132, (__int64)v8, 2);
    if ( v134 )
      sub_161E7C0((__int64)&v134, (__int64)v134);
    v134 = v132[0];
    if ( v132[0] )
      sub_1623210((__int64)v132, v132[0], (__int64)&v134);
  }
  v129 = (unsigned __int8 *)a5;
  v130[0] = *a5;
  v10 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(v135 + 56) + 40LL), 32, v130, 1);
  v133 = 257;
  v11 = sub_1285290((__int64 *)&v134, *(_QWORD *)(*(_QWORD *)v10 + 24LL), v10, (int)&v129, 1, (__int64)v132, 0);
  v12 = v11;
  v13 = *(unsigned __int8 **)(a3 + 48);
  v132[0] = v13;
  if ( !v13 )
  {
    v14 = v11 + 48;
    if ( (unsigned __int8 **)(v11 + 48) == v132 )
      goto LABEL_10;
    v93 = *(_QWORD *)(v11 + 48);
    if ( !v93 )
      goto LABEL_10;
LABEL_108:
    sub_161E7C0(v14, v93);
    goto LABEL_109;
  }
  v14 = v11 + 48;
  sub_1623A60((__int64)v132, (__int64)v13, 2);
  if ( (unsigned __int8 **)(v12 + 48) == v132 )
  {
    if ( v132[0] )
      sub_161E7C0((__int64)v132, (__int64)v132[0]);
    goto LABEL_10;
  }
  v93 = *(_QWORD *)(v12 + 48);
  if ( v93 )
    goto LABEL_108;
LABEL_109:
  v94 = v132[0];
  *(unsigned __int8 **)(v12 + 48) = v132[0];
  if ( v94 )
    sub_1623210((__int64)v132, v94, v14);
LABEL_10:
  v15 = *(_QWORD *)a4;
  v133 = 257;
  v16 = (__int64 *)sub_1904B50((__int64 *)&v134, v12, v15, (__int64 *)v132);
  if ( v16 == (__int64 *)v12 )
    goto LABEL_15;
  v17 = (unsigned __int8 **)(v16 + 6);
  v18 = *(unsigned __int8 **)(a3 + 48);
  v132[0] = v18;
  if ( !v18 )
  {
    if ( v17 == v132 )
      goto LABEL_15;
    v91 = v16[6];
    if ( !v91 )
      goto LABEL_15;
LABEL_104:
    sub_161E7C0((__int64)(v16 + 6), v91);
    goto LABEL_105;
  }
  sub_1623A60((__int64)v132, (__int64)v18, 2);
  if ( v17 == v132 )
  {
    if ( v132[0] )
      sub_161E7C0((__int64)v132, (__int64)v132[0]);
    goto LABEL_15;
  }
  v91 = v16[6];
  if ( v91 )
    goto LABEL_104;
LABEL_105:
  v92 = v132[0];
  v16[6] = (__int64)v132[0];
  if ( v92 )
    sub_1623210((__int64)v132, v92, (__int64)(v16 + 6));
LABEL_15:
  v19 = 0x17FFFFFFE8LL;
  v20 = *(_BYTE *)(a4 + 23) & 0x40;
  v21 = *(_DWORD *)(a4 + 20) & 0xFFFFFFF;
  if ( v21 )
  {
    v22 = 24LL * *(unsigned int *)(a4 + 56) + 8;
    v23 = 0;
    do
    {
      v24 = a4 - 24LL * v21;
      if ( v20 )
        v24 = *(_QWORD *)(a4 - 8);
      if ( v119 == *(_QWORD *)(v24 + v22) )
      {
        v19 = 24 * v23;
        goto LABEL_22;
      }
      ++v23;
      v22 += 8;
    }
    while ( v21 != (_DWORD)v23 );
    v19 = 0x17FFFFFFE8LL;
  }
LABEL_22:
  if ( v20 )
    v25 = *(_QWORD *)(a4 - 8);
  else
    v25 = a4 - 24LL * v21;
  v26 = *(_QWORD *)(v25 + v19);
  if ( *(_BYTE *)(v26 + 16) == 13 )
  {
    v27 = *(_DWORD *)(v26 + 32);
    if ( v27 <= 0x40 )
    {
      v123 = (__int64)v16;
      if ( !*(_QWORD *)(v26 + 24) )
        goto LABEL_31;
    }
    else
    {
      v123 = (__int64)v16;
      if ( v27 == (unsigned int)sub_16A57B0(v26 + 24) )
        goto LABEL_31;
    }
  }
  v133 = 257;
  v123 = sub_12899C0((__int64 *)&v134, (__int64)v16, v26, (__int64)v132, 0, 0);
  v28 = *(unsigned __int8 **)(a3 + 48);
  v132[0] = v28;
  if ( !v28 )
  {
    v29 = v123 + 48;
    if ( (unsigned __int8 **)(v123 + 48) == v132 )
      goto LABEL_31;
    v89 = *(_QWORD *)(v123 + 48);
    if ( !v89 )
      goto LABEL_31;
LABEL_99:
    sub_161E7C0(v29, v89);
    goto LABEL_100;
  }
  v29 = v123 + 48;
  sub_1623A60((__int64)v132, (__int64)v28, 2);
  if ( (unsigned __int8 **)(v123 + 48) == v132 )
  {
    if ( v132[0] )
      sub_161E7C0((__int64)v132, (__int64)v132[0]);
    goto LABEL_31;
  }
  v89 = *(_QWORD *)(v123 + 48);
  if ( v89 )
    goto LABEL_99;
LABEL_100:
  v90 = v132[0];
  *(unsigned __int8 **)(v123 + 48) = v132[0];
  if ( v90 )
    sub_1623210((__int64)v132, v90, v29);
LABEL_31:
  v30 = *(_QWORD *)(v6 - 72);
  v31 = sub_15A0680(*v16, 0, 0);
  v32 = *(__int64 **)(v30 - 48);
  if ( !v32 || (v33 = (__int64)v16, v32 != a5) )
  {
    v33 = v31;
    v31 = (__int64)v16;
  }
  v131 = 257;
  if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v31 + 16) > 0x10u )
  {
    v111 = v33;
    v114 = *(_WORD *)(v30 + 18) & 0x7FFF;
    v126 = v31;
    v133 = 257;
    v95 = sub_1648A60(56, 2u);
    v96 = v126;
    v34 = v95;
    if ( v95 )
    {
      v127 = (__int64)v95;
      v97 = *(_QWORD ***)v111;
      if ( *(_BYTE *)(*(_QWORD *)v111 + 8LL) == 16 )
      {
        v108 = v111;
        v109 = v114;
        v112 = v96;
        v115 = v97[4];
        v98 = (__int64 *)sub_1643320(*v97);
        v99 = (__int64)sub_16463B0(v98, (unsigned int)v115);
        v100 = v112;
        v101 = v109;
        v102 = v108;
      }
      else
      {
        v110 = v111;
        v113 = v114;
        v117 = v96;
        v99 = sub_1643320(*v97);
        v102 = v110;
        v101 = v113;
        v100 = v117;
      }
      sub_15FEC10((__int64)v34, v99, 51, v101, v102, v100, (__int64)v132, 0);
    }
    else
    {
      v127 = 0;
    }
    if ( v135 )
    {
      v116 = v136;
      sub_157E9D0(v135 + 40, (__int64)v34);
      v103 = *v116;
      v104 = v34[3] & 7LL;
      v34[4] = v116;
      v103 &= 0xFFFFFFFFFFFFFFF8LL;
      v34[3] = v103 | v104;
      *(_QWORD *)(v103 + 8) = v34 + 3;
      *v116 = *v116 & 7 | (unsigned __int64)(v34 + 3);
    }
    sub_164B780(v127, v130);
    if ( v134 )
    {
      v129 = v134;
      sub_1623A60((__int64)&v129, (__int64)v134, 2);
      v105 = v34[6];
      v106 = (__int64)(v34 + 6);
      if ( v105 )
      {
        sub_161E7C0((__int64)(v34 + 6), v105);
        v106 = (__int64)(v34 + 6);
      }
      v107 = v129;
      v34[6] = v129;
      if ( v107 )
        sub_1623210((__int64)&v129, v107, v106);
    }
  }
  else
  {
    v34 = (_QWORD *)sub_15A37B0(*(_WORD *)(v30 + 18) & 0x7FFF, (_QWORD *)v33, (_QWORD *)v31, 0);
  }
  if ( *(_QWORD *)(v6 - 72) )
  {
    v35 = *(_QWORD *)(v6 - 64);
    v36 = *(_QWORD *)(v6 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v36 = v35;
    if ( v35 )
      *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
  }
  *(_QWORD *)(v6 - 72) = v34;
  if ( v34 )
  {
    v37 = v34[1];
    *(_QWORD *)(v6 - 64) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = (v6 - 64) | *(_QWORD *)(v37 + 16) & 3LL;
    v38 = *(_QWORD *)(v6 - 56);
    v39 = v6 - 72;
    *(_QWORD *)(v39 + 16) = (unsigned __int64)(v34 + 1) | v38 & 3;
    v34[1] = v39;
  }
  sub_1AEB370(v30, a1[5]);
  v40 = **(_QWORD **)(*a1 + 32);
  v41 = sub_157EBA0(v40);
  v42 = *(_QWORD *)(v40 + 48);
  v43 = *(_QWORD *)(v41 - 72);
  v118 = v41;
  v44 = *v16;
  v133 = 259;
  v125 = v44;
  if ( v42 )
    v42 -= 24;
  v132[0] = "tcphi";
  v120 = v42;
  v45 = sub_1648B60(64);
  v46 = v120;
  v47 = v45;
  if ( v45 )
  {
    v121 = v45;
    sub_15F1EA0(v45, v125, 53, 0, 0, v46);
    *(_DWORD *)(v47 + 56) = 2;
    sub_164B780(v47, (__int64 *)v132);
    sub_1648880(v47, *(_DWORD *)(v47 + 56), 1);
  }
  else
  {
    v121 = 0;
  }
  v48 = *(unsigned __int8 **)(v43 + 48);
  v135 = *(_QWORD *)(v43 + 40);
  v136 = (unsigned __int64 *)(v43 + 24);
  v132[0] = v48;
  if ( v48 )
  {
    sub_1623A60((__int64)v132, (__int64)v48, 2);
    v49 = v134;
    if ( !v134 )
      goto LABEL_50;
  }
  else
  {
    v49 = v134;
    if ( !v134 )
      goto LABEL_52;
  }
  sub_161E7C0((__int64)&v134, (__int64)v49);
LABEL_50:
  v134 = v132[0];
  if ( v132[0] )
    sub_1623210((__int64)v132, v132[0], (__int64)&v134);
LABEL_52:
  v132[0] = "tcdec";
  v133 = 259;
  v50 = sub_15A0680(v125, 1, 0);
  v51 = v47;
  v56 = sub_156E1C0((__int64 *)&v134, v47, v50, (__int64)v132, 0, 1u);
  v57 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  if ( v57 == *(_DWORD *)(v47 + 56) )
  {
    sub_15F55D0(v47, v47, v52, v53, v54, v55);
    v57 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  }
  v58 = (v57 + 1) & 0xFFFFFFF;
  v59 = v58 | *(_DWORD *)(v47 + 20) & 0xF0000000;
  *(_DWORD *)(v47 + 20) = v59;
  if ( (v59 & 0x40000000) != 0 )
    v60 = *(_QWORD *)(v47 - 8);
  else
    v60 = v121 - 24 * v58;
  v61 = (__int64 **)(v60 + 24LL * (unsigned int)(v58 - 1));
  if ( *v61 )
  {
    v62 = v61[1];
    v63 = (unsigned __int64)v61[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v63 = v62;
    if ( v62 )
    {
      v51 = v62[2] & 3;
      v62[2] = v51 | v63;
    }
  }
  *v61 = v16;
  v64 = v16[1];
  v61[1] = (__int64 *)v64;
  if ( v64 )
  {
    v51 = (unsigned __int64)(v61 + 1) | *(_QWORD *)(v64 + 16) & 3LL;
    *(_QWORD *)(v64 + 16) = v51;
  }
  v61[2] = (__int64 *)((unsigned __int64)(v16 + 1) | (unsigned __int64)v61[2] & 3);
  v16[1] = (__int64)v61;
  v65 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  v66 = (unsigned int)(v65 - 1);
  if ( (*(_BYTE *)(v47 + 23) & 0x40) != 0 )
    v67 = *(_QWORD *)(v47 - 8);
  else
    v67 = v121 - 24 * v65;
  v68 = 3LL * *(unsigned int *)(v47 + 56);
  *(_QWORD *)(v67 + 8 * v66 + 24LL * *(unsigned int *)(v47 + 56) + 8) = v119;
  v69 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  if ( v69 == *(_DWORD *)(v47 + 56) )
  {
    sub_15F55D0(v47, v51, v68, v67, v54, v55);
    v69 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  }
  v70 = (v69 + 1) & 0xFFFFFFF;
  v71 = v70 | *(_DWORD *)(v47 + 20) & 0xF0000000;
  *(_DWORD *)(v47 + 20) = v71;
  if ( (v71 & 0x40000000) != 0 )
    v72 = *(_QWORD *)(v47 - 8);
  else
    v72 = v121 - 24 * v70;
  v73 = (__int64 *)(v72 + 24LL * (unsigned int)(v70 - 1));
  if ( *v73 )
  {
    v74 = v73[1];
    v75 = v73[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v75 = v74;
    if ( v74 )
      *(_QWORD *)(v74 + 16) = *(_QWORD *)(v74 + 16) & 3LL | v75;
  }
  *v73 = v56;
  if ( v56 )
  {
    v76 = *(_QWORD *)(v56 + 8);
    v73[1] = v76;
    if ( v76 )
      *(_QWORD *)(v76 + 16) = (unsigned __int64)(v73 + 1) | *(_QWORD *)(v76 + 16) & 3LL;
    v73[2] = (v56 + 8) | v73[2] & 3;
    *(_QWORD *)(v56 + 8) = v73;
  }
  v77 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v47 + 23) & 0x40) != 0 )
    v78 = *(_QWORD *)(v47 - 8);
  else
    v78 = v121 - 24 * v77;
  *(_QWORD *)(v78 + 8LL * (unsigned int)(v77 - 1) + 24LL * *(unsigned int *)(v47 + 56) + 8) = v40;
  v79 = *(_QWORD *)(v118 - 24);
  if ( v40 != v79 || (v80 = 34, !v79) )
    v80 = 41;
  v81 = *(_QWORD *)(v43 - 48) == 0;
  *(_WORD *)(v43 + 18) = v80 | *(_WORD *)(v43 + 18) & 0x8000;
  if ( !v81 )
  {
    v82 = *(_QWORD *)(v43 - 40);
    v83 = *(_QWORD *)(v43 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v83 = v82;
    if ( v82 )
      *(_QWORD *)(v82 + 16) = *(_QWORD *)(v82 + 16) & 3LL | v83;
  }
  *(_QWORD *)(v43 - 48) = v56;
  if ( v56 )
  {
    v84 = *(_QWORD *)(v56 + 8);
    *(_QWORD *)(v43 - 40) = v84;
    if ( v84 )
      *(_QWORD *)(v84 + 16) = (v43 - 40) | *(_QWORD *)(v84 + 16) & 3LL;
    *(_QWORD *)(v43 - 32) = (v56 + 8) | *(_QWORD *)(v43 - 32) & 3LL;
    *(_QWORD *)(v56 + 8) = v43 - 48;
  }
  v85 = sub_15A0680(v125, 0, 0);
  if ( *(_QWORD *)(v43 - 24) )
  {
    v86 = *(_QWORD *)(v43 - 16);
    v87 = *(_QWORD *)(v43 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v87 = v86;
    if ( v86 )
      *(_QWORD *)(v86 + 16) = *(_QWORD *)(v86 + 16) & 3LL | v87;
  }
  *(_QWORD *)(v43 - 24) = v85;
  if ( v85 )
  {
    v88 = *(_QWORD *)(v85 + 8);
    *(_QWORD *)(v43 - 16) = v88;
    if ( v88 )
      *(_QWORD *)(v88 + 16) = (v43 - 16) | *(_QWORD *)(v88 + 16) & 3LL;
    *(_QWORD *)(v43 - 8) = (v85 + 8) | *(_QWORD *)(v43 - 8) & 3LL;
    *(_QWORD *)(v85 + 8) = v43 - 24;
  }
  sub_1648F20(a3, v123, v40);
  sub_1465150(a1[4], *a1);
  if ( v134 )
    sub_161E7C0((__int64)&v134, (__int64)v134);
}
