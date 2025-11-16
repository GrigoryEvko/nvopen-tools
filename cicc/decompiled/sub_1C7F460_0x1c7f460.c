// Function: sub_1C7F460
// Address: 0x1c7f460
//
void __fastcall sub_1C7F460(
        __int64 *a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16)
{
  _QWORD *v16; // rbx
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r12
  unsigned __int64 v20; // r14
  unsigned __int8 *v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  _QWORD *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // eax
  __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rsi
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // r11
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // r11
  __int64 v58; // rsi
  __int64 v59; // rdx
  unsigned __int8 *v60; // rsi
  __int64 v61; // rax
  unsigned __int8 *v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rcx
  unsigned __int8 **v65; // r8
  __int64 v66; // r9
  __int64 v67; // r15
  int v68; // eax
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // rcx
  unsigned __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rax
  __int64 v79; // r14
  _QWORD *v80; // rax
  _QWORD *v81; // rbx
  unsigned __int64 *v82; // r12
  __int64 v83; // rax
  unsigned __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rax
  unsigned __int8 *v92; // rsi
  __int64 v93; // r15
  unsigned int v94; // r12d
  __int64 v95; // rsi
  __int64 v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // r14
  unsigned __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // rsi
  __int64 v102; // rdx
  unsigned __int8 *v103; // rsi
  __int64 v104; // [rsp+8h] [rbp-178h]
  __int64 v106; // [rsp+18h] [rbp-168h]
  __int64 v107; // [rsp+20h] [rbp-160h]
  __int64 *v110; // [rsp+30h] [rbp-150h]
  __int64 v111; // [rsp+30h] [rbp-150h]
  __int64 v113; // [rsp+38h] [rbp-148h]
  _QWORD *v114; // [rsp+38h] [rbp-148h]
  __int64 v115; // [rsp+38h] [rbp-148h]
  __int64 v116; // [rsp+38h] [rbp-148h]
  __int64 v117; // [rsp+38h] [rbp-148h]
  __int64 *v118; // [rsp+38h] [rbp-148h]
  __int64 v120; // [rsp+58h] [rbp-128h]
  __int64 v121; // [rsp+58h] [rbp-128h]
  __int64 v122; // [rsp+58h] [rbp-128h]
  unsigned __int64 *v123; // [rsp+58h] [rbp-128h]
  unsigned __int8 *v124; // [rsp+68h] [rbp-118h] BYREF
  __int64 v125[2]; // [rsp+70h] [rbp-110h] BYREF
  char v126; // [rsp+80h] [rbp-100h]
  char v127; // [rsp+81h] [rbp-FFh]
  __int64 v128[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v129; // [rsp+A0h] [rbp-E0h]
  unsigned __int8 *v130[2]; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int64 v131; // [rsp+C0h] [rbp-C0h]
  __int64 v132; // [rsp+C8h] [rbp-B8h]
  __int64 v133; // [rsp+D0h] [rbp-B0h]
  int v134; // [rsp+D8h] [rbp-A8h]
  __int64 v135; // [rsp+E0h] [rbp-A0h]
  __int64 v136; // [rsp+E8h] [rbp-98h]
  unsigned __int8 *v137; // [rsp+100h] [rbp-80h] BYREF
  __int64 v138; // [rsp+108h] [rbp-78h]
  __int64 *v139; // [rsp+110h] [rbp-70h]
  __int64 v140; // [rsp+118h] [rbp-68h]
  __int64 v141; // [rsp+120h] [rbp-60h]
  int v142; // [rsp+128h] [rbp-58h]
  __int64 v143; // [rsp+130h] [rbp-50h]
  __int64 v144; // [rsp+138h] [rbp-48h]

  if ( *(_BYTE *)(a4 + 16) != 13 )
    goto LABEL_5;
  v16 = *(_QWORD **)(a4 + 24);
  if ( *(_DWORD *)(a4 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  if ( (unsigned int)dword_4FBD560 < (unsigned __int64)v16 )
  {
LABEL_5:
    v17 = (_QWORD *)a1[5];
    LOWORD(v139) = 259;
    v137 = "memset.exit";
    v106 = (__int64)v17;
    v107 = sub_157FBF0(v17, a1 + 3, (__int64)&v137);
    v137 = "memset.loop";
    LOWORD(v139) = 259;
    v18 = (_QWORD *)sub_22077B0(64);
    v19 = (__int64)v18;
    if ( v18 )
      sub_157FB60(v18, a15, (__int64)&v137, a16, v107);
    v20 = sub_157EBA0(v106);
    v130[0] = 0;
    v132 = sub_16498A0(v20);
    v133 = 0;
    v134 = 0;
    v135 = 0;
    v136 = 0;
    v130[1] = *(unsigned __int8 **)(v20 + 40);
    v131 = v20 + 24;
    v21 = *(unsigned __int8 **)(v20 + 48);
    v137 = v21;
    if ( v21 )
    {
      sub_1623A60((__int64)&v137, (__int64)v21, 2);
      v130[0] = v137;
      if ( v137 )
        sub_1623210((__int64)&v137, v137, (__int64)v130);
    }
    LOWORD(v139) = 257;
    v22 = *(_QWORD *)a4;
    v23 = sub_15A0680(*(_QWORD *)a4, 0, 0);
    v120 = sub_12AA0C0((__int64 *)v130, 0x22u, (_BYTE *)a4, v23, (__int64)&v137);
    v24 = sub_1648A60(56, 3u);
    v27 = v24;
    if ( v24 )
      sub_15F83E0((__int64)v24, v19, v107, v120, 0);
    sub_1AA6530(v20, v27, a7, a8, a9, a10, v25, v26, a13, a14);
    v28 = sub_157E9C0(v19);
    v138 = v19;
    v140 = v28;
    v139 = (__int64 *)(v19 + 40);
    v129 = 257;
    v137 = 0;
    v141 = 0;
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v127 = 1;
    v125[0] = (__int64)"index";
    v126 = 3;
    v29 = sub_1648B60(64);
    v30 = v29;
    if ( v29 )
    {
      v121 = v29;
      sub_15F1EA0(v29, v22, 53, 0, 0, 0);
      *(_DWORD *)(v30 + 56) = 0;
      sub_164B780(v30, v128);
      sub_1648880(v30, *(_DWORD *)(v30 + 56), 1);
    }
    else
    {
      v121 = 0;
    }
    if ( v138 )
    {
      v31 = v139;
      sub_157E9D0(v138 + 40, v30);
      v32 = *(_QWORD *)(v30 + 24);
      v33 = *v31;
      *(_QWORD *)(v30 + 32) = v31;
      v33 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v30 + 24) = v33 | v32 & 7;
      *(_QWORD *)(v33 + 8) = v30 + 24;
      *v31 = *v31 & 7 | (v30 + 24);
    }
    sub_164B780(v121, v125);
    if ( v137 )
    {
      v124 = v137;
      sub_1623A60((__int64)&v124, (__int64)v137, 2);
      v34 = *(_QWORD *)(v30 + 48);
      v35 = v30 + 48;
      if ( v34 )
      {
        sub_161E7C0(v30 + 48, v34);
        v35 = v30 + 48;
      }
      v36 = v124;
      *(_QWORD *)(v30 + 48) = v124;
      if ( v36 )
        sub_1623210((__int64)&v124, v36, v35);
    }
    v38 = sub_15A0680(v22, 0, 0);
    v41 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    if ( v41 == *(_DWORD *)(v30 + 56) )
    {
      v104 = v38;
      sub_15F55D0(v30, 0, v37, v38, v39, v40);
      v38 = v104;
      v41 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    }
    v42 = (v41 + 1) & 0xFFFFFFF;
    v43 = v42 | *(_DWORD *)(v30 + 20) & 0xF0000000;
    *(_DWORD *)(v30 + 20) = v43;
    if ( (v43 & 0x40000000) != 0 )
      v44 = *(_QWORD *)(v30 - 8);
    else
      v44 = v121 - 24 * v42;
    v45 = (__int64 *)(v44 + 24LL * (unsigned int)(v42 - 1));
    if ( *v45 )
    {
      v46 = v45[1];
      v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v47 = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
    }
    *v45 = v38;
    if ( v38 )
    {
      v48 = *(_QWORD *)(v38 + 8);
      v45[1] = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
      v45[2] = (v38 + 8) | v45[2] & 3;
      *(_QWORD *)(v38 + 8) = v45;
    }
    v49 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
      v50 = *(_QWORD *)(v30 - 8);
    else
      v50 = v121 - 24 * v49;
    *(_QWORD *)(v50 + 8LL * (unsigned int)(v49 - 1) + 24LL * *(unsigned int *)(v30 + 56) + 8) = v106;
    v127 = 1;
    v125[0] = (__int64)"dst.gep";
    v126 = 3;
    v51 = sub_12815B0((__int64 *)&v137, a2, a3, v30, (__int64)v125);
    v129 = 257;
    v113 = v51;
    v52 = sub_1648A60(64, 2u);
    v53 = (__int64)v52;
    if ( v52 )
    {
      v54 = v113;
      v114 = v52;
      sub_15F9650((__int64)v52, a5, v54, a6, 0);
      v53 = (__int64)v114;
    }
    if ( v138 )
    {
      v115 = v53;
      v110 = v139;
      sub_157E9D0(v138 + 40, v53);
      v53 = v115;
      v55 = *(_QWORD *)(v115 + 24);
      v56 = *v110;
      *(_QWORD *)(v115 + 32) = v110;
      v56 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v115 + 24) = v56 | v55 & 7;
      *(_QWORD *)(v56 + 8) = v115 + 24;
      *v110 = *v110 & 7 | (v115 + 24);
    }
    v116 = v53;
    sub_164B780(v53, v128);
    if ( v137 )
    {
      v124 = v137;
      sub_1623A60((__int64)&v124, (__int64)v137, 2);
      v57 = v116;
      v58 = *(_QWORD *)(v116 + 48);
      v59 = v116 + 48;
      if ( v58 )
      {
        v111 = v116;
        v117 = v116 + 48;
        sub_161E7C0(v117, v58);
        v57 = v111;
        v59 = v117;
      }
      v60 = v124;
      *(_QWORD *)(v57 + 48) = v124;
      if ( v60 )
        sub_1623210((__int64)&v124, v60, v59);
    }
    v127 = 1;
    v125[0] = (__int64)"inc";
    v126 = 3;
    v61 = sub_15A0680(v22, 1, 0);
    v62 = (unsigned __int8 *)v61;
    if ( *(_BYTE *)(v30 + 16) > 0x10u || *(_BYTE *)(v61 + 16) > 0x10u )
    {
      v129 = 257;
      v87 = sub_15FB440(11, (__int64 *)v30, v61, (__int64)v128, 0);
      v67 = v87;
      if ( v138 )
      {
        v118 = v139;
        sub_157E9D0(v138 + 40, v87);
        v88 = *v118;
        v89 = *(_QWORD *)(v67 + 24) & 7LL;
        *(_QWORD *)(v67 + 32) = v118;
        v88 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v67 + 24) = v88 | v89;
        *(_QWORD *)(v88 + 8) = v67 + 24;
        *v118 = *v118 & 7 | (v67 + 24);
      }
      sub_164B780(v67, v125);
      v62 = v137;
      if ( v137 )
      {
        v124 = v137;
        sub_1623A60((__int64)&v124, (__int64)v137, 2);
        v90 = *(_QWORD *)(v67 + 48);
        v65 = &v124;
        v63 = v67 + 48;
        if ( v90 )
        {
          sub_161E7C0(v67 + 48, v90);
          v65 = &v124;
          v63 = v67 + 48;
        }
        v62 = v124;
        *(_QWORD *)(v67 + 48) = v124;
        if ( v62 )
        {
          sub_1623210((__int64)&v124, v62, v63);
          v68 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
          if ( v68 != *(_DWORD *)(v30 + 56) )
            goto LABEL_47;
          goto LABEL_83;
        }
      }
    }
    else
    {
      v67 = sub_15A2B30((__int64 *)v30, v61, 0, 0, *(double *)a7.m128_u64, a8, a9);
    }
    v68 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    if ( v68 != *(_DWORD *)(v30 + 56) )
    {
LABEL_47:
      v69 = (v68 + 1) & 0xFFFFFFF;
      v70 = v69 | *(_DWORD *)(v30 + 20) & 0xF0000000;
      *(_DWORD *)(v30 + 20) = v70;
      if ( (v70 & 0x40000000) != 0 )
        v71 = *(_QWORD *)(v30 - 8);
      else
        v71 = v121 - 24 * v69;
      v72 = (__int64 *)(v71 + 24LL * (unsigned int)(v69 - 1));
      if ( *v72 )
      {
        v73 = v72[1];
        v74 = v72[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v74 = v73;
        if ( v73 )
          *(_QWORD *)(v73 + 16) = *(_QWORD *)(v73 + 16) & 3LL | v74;
      }
      *v72 = v67;
      if ( v67 )
      {
        v75 = *(_QWORD *)(v67 + 8);
        v72[1] = v75;
        if ( v75 )
          *(_QWORD *)(v75 + 16) = (unsigned __int64)(v72 + 1) | *(_QWORD *)(v75 + 16) & 3LL;
        v72[2] = (v67 + 8) | v72[2] & 3;
        *(_QWORD *)(v67 + 8) = v72;
      }
      v76 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
        v77 = *(_QWORD *)(v30 - 8);
      else
        v77 = v121 - 24 * v76;
      *(_QWORD *)(v77 + 8LL * (unsigned int)(v76 - 1) + 24LL * *(unsigned int *)(v30 + 56) + 8) = v19;
      v127 = 1;
      v125[0] = (__int64)"cmp";
      v126 = 3;
      v78 = sub_12AA0C0((__int64 *)&v137, 0x24u, (_BYTE *)v67, a4, (__int64)v125);
      v129 = 257;
      v79 = v78;
      v80 = sub_1648A60(56, 3u);
      v81 = v80;
      if ( v80 )
        sub_15F83E0((__int64)v80, v19, v107, v79, 0);
      if ( v138 )
      {
        v82 = (unsigned __int64 *)v139;
        sub_157E9D0(v138 + 40, (__int64)v81);
        v83 = v81[3];
        v84 = *v82;
        v81[4] = v82;
        v84 &= 0xFFFFFFFFFFFFFFF8LL;
        v81[3] = v84 | v83 & 7;
        *(_QWORD *)(v84 + 8) = v81 + 3;
        *v82 = *v82 & 7 | (unsigned __int64)(v81 + 3);
      }
      sub_164B780((__int64)v81, v128);
      if ( v137 )
      {
        v124 = v137;
        sub_1623A60((__int64)&v124, (__int64)v137, 2);
        v85 = v81[6];
        if ( v85 )
          sub_161E7C0((__int64)(v81 + 6), v85);
        v86 = v124;
        v81[6] = v124;
        if ( v86 )
          sub_1623210((__int64)&v124, v86, (__int64)(v81 + 6));
        if ( v137 )
          sub_161E7C0((__int64)&v137, (__int64)v137);
      }
      if ( v130[0] )
        sub_161E7C0((__int64)v130, (__int64)v130[0]);
      return;
    }
LABEL_83:
    sub_15F55D0(v30, (__int64)v62, v63, v64, (__int64)v65, v66);
    v68 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    goto LABEL_47;
  }
  v137 = 0;
  v140 = sub_16498A0((__int64)a1);
  v91 = a1[5];
  v92 = (unsigned __int8 *)a1[6];
  v141 = 0;
  v142 = 0;
  v138 = v91;
  v143 = 0;
  v144 = 0;
  v139 = a1 + 3;
  v130[0] = v92;
  if ( v92 )
  {
    sub_1623A60((__int64)v130, (__int64)v92, 2);
    if ( v137 )
      sub_161E7C0((__int64)&v137, (__int64)v137);
    v137 = v130[0];
    if ( v130[0] )
      sub_1623210((__int64)v130, v130[0], (__int64)&v137);
  }
  v93 = *(_QWORD *)a4;
  if ( v16 )
  {
    v94 = 0;
    v95 = 0;
    do
    {
      v130[0] = "dst.gep.unroll";
      LOWORD(v131) = 259;
      v96 = sub_15A0680(v93, v95, 0);
      v122 = sub_12815B0((__int64 *)&v137, a2, a3, v96, (__int64)v130);
      LOWORD(v131) = 257;
      v97 = sub_1648A60(64, 2u);
      v98 = v97;
      if ( v97 )
        sub_15F9650((__int64)v97, a5, v122, a6, 0);
      if ( v138 )
      {
        v123 = (unsigned __int64 *)v139;
        sub_157E9D0(v138 + 40, (__int64)v98);
        v99 = *v123;
        v100 = v98[3] & 7LL;
        v98[4] = v123;
        v99 &= 0xFFFFFFFFFFFFFFF8LL;
        v98[3] = v99 | v100;
        *(_QWORD *)(v99 + 8) = v98 + 3;
        *v123 = *v123 & 7 | (unsigned __int64)(v98 + 3);
      }
      sub_164B780((__int64)v98, (__int64 *)v130);
      if ( v137 )
      {
        v128[0] = (__int64)v137;
        sub_1623A60((__int64)v128, (__int64)v137, 2);
        v101 = v98[6];
        v102 = (__int64)(v98 + 6);
        if ( v101 )
        {
          sub_161E7C0((__int64)(v98 + 6), v101);
          v102 = (__int64)(v98 + 6);
        }
        v103 = (unsigned __int8 *)v128[0];
        v98[6] = v128[0];
        if ( v103 )
          sub_1623210((__int64)v128, v103, v102);
      }
      v95 = ++v94;
    }
    while ( (_QWORD *)v94 != v16 );
  }
  if ( v137 )
    sub_161E7C0((__int64)&v137, (__int64)v137);
}
