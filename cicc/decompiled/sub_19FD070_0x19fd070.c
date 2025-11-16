// Function: sub_19FD070
// Address: 0x19fd070
//
__int64 __fastcall sub_19FD070(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v11; // r14
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int8 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int8 **v28; // r15
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // r12
  __int64 v32; // rax
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rsi
  __int64 v38; // rax
  double v39; // xmm4_8
  double v40; // xmm5_8
  __int64 v41; // rsi
  __int64 v42; // rdx
  unsigned __int8 *v43; // rsi
  __int64 v44; // rcx
  unsigned __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rdx
  unsigned __int8 *v49; // rsi
  _QWORD *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rcx
  __int64 v55; // rsi
  __int64 v56; // rdx
  unsigned __int8 *v57; // rsi
  __int64 *v58; // rax
  _QWORD *v59; // rax
  char v60; // al
  __int64 v61; // rcx
  _QWORD *v62; // rax
  _QWORD *v63; // rbx
  unsigned __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rsi
  __int64 v71; // rsi
  int v72; // eax
  __int64 v73; // rax
  int v74; // edx
  __int64 v75; // rdx
  unsigned __int8 *v76; // rax
  __int64 v77; // rcx
  unsigned __int64 v78; // rdx
  __int64 v79; // rdx
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rdx
  int v84; // eax
  __int64 v85; // rax
  int v86; // edx
  __int64 v87; // rdx
  _QWORD *v88; // rax
  __int64 v89; // rcx
  unsigned __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 *v96; // [rsp+8h] [rbp-148h]
  __int64 v97; // [rsp+8h] [rbp-148h]
  __int64 v98; // [rsp+10h] [rbp-140h]
  unsigned __int64 *v99; // [rsp+10h] [rbp-140h]
  _QWORD *v100; // [rsp+10h] [rbp-140h]
  __int64 v101; // [rsp+10h] [rbp-140h]
  __int64 v102; // [rsp+10h] [rbp-140h]
  __int64 v103; // [rsp+10h] [rbp-140h]
  unsigned __int8 v104; // [rsp+1Fh] [rbp-131h]
  __int64 v106; // [rsp+38h] [rbp-118h]
  __int64 v107; // [rsp+38h] [rbp-118h]
  unsigned __int64 *v108; // [rsp+38h] [rbp-118h]
  __int64 v109; // [rsp+40h] [rbp-110h]
  __int64 v110; // [rsp+40h] [rbp-110h]
  unsigned __int8 v111; // [rsp+48h] [rbp-108h]
  __int64 v112; // [rsp+48h] [rbp-108h]
  __int64 *v114; // [rsp+58h] [rbp-F8h]
  __int64 v115; // [rsp+60h] [rbp-F0h]
  __int64 v116; // [rsp+68h] [rbp-E8h]
  _QWORD *v117; // [rsp+68h] [rbp-E8h]
  __int64 v118; // [rsp+78h] [rbp-D8h]
  __int64 *v119; // [rsp+78h] [rbp-D8h]
  __int64 v120; // [rsp+78h] [rbp-D8h]
  _QWORD *v121; // [rsp+78h] [rbp-D8h]
  int v122; // [rsp+84h] [rbp-CCh] BYREF
  unsigned __int8 *v123; // [rsp+88h] [rbp-C8h] BYREF
  __int64 v124[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v125; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v126[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v127; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v128; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD *v129; // [rsp+D8h] [rbp-78h]
  __int64 *v130; // [rsp+E0h] [rbp-70h]
  __int64 v131; // [rsp+E8h] [rbp-68h]
  __int64 v132; // [rsp+F0h] [rbp-60h]
  int v133; // [rsp+F8h] [rbp-58h]
  __int64 v134; // [rsp+100h] [rbp-50h]
  __int64 v135; // [rsp+108h] [rbp-48h]

  v111 = 0;
  v118 = *(_QWORD *)(a1 + 80);
  v115 = a1 + 72;
  while ( v118 != v115 )
  {
    while ( 1 )
    {
      v116 = v118;
      v11 = *(__int64 **)(v118 + 24);
      v12 = (__int64 *)(v118 + 16);
      v118 = *(_QWORD *)(v118 + 8);
      if ( v11 != v12 )
        break;
LABEL_17:
      if ( v115 == v118 )
        return v111;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v11 )
          BUG();
        if ( *((_BYTE *)v11 - 8) != 78 )
          goto LABEL_8;
        v14 = *(v11 - 6);
        if ( *(_BYTE *)(v14 + 16) )
          goto LABEL_8;
        if ( !(unsigned __int8)sub_1560260(v11 + 4, -1, 21) )
        {
          v13 = *(v11 - 6);
          if ( *(_BYTE *)(v13 + 16) )
            break;
          v128 = *(unsigned __int8 **)(v13 + 112);
          if ( !(unsigned __int8)sub_1560260(&v128, -1, 21) )
            break;
        }
        if ( (unsigned __int8)sub_1560260(v11 + 4, -1, 5) )
          break;
        v15 = *(v11 - 6);
        if ( *(_BYTE *)(v15 + 16) )
          goto LABEL_8;
        v128 = *(unsigned __int8 **)(v15 + 112);
        if ( (unsigned __int8)sub_1560260(&v128, -1, 5) )
          break;
        v11 = (__int64 *)v11[1];
        if ( v12 == v11 )
          goto LABEL_17;
      }
      if ( (*(_BYTE *)(v14 + 32) & 0xFu) - 7 <= 1 )
        goto LABEL_8;
      if ( !sub_149CB50(*a2, v14, (unsigned int *)&v122) )
        goto LABEL_8;
      if ( (((int)*(unsigned __int8 *)(*a2 + v122 / 4) >> (2 * (v122 & 3))) & 3) == 0 )
        goto LABEL_8;
      if ( (unsigned int)(v122 - 353) > 1 )
        goto LABEL_8;
      v17 = sub_14A2FF0(a3);
      if ( !v17 || (unsigned __int8)sub_1560260(v11 + 4, -1, 36) )
        goto LABEL_8;
      v114 = v11 - 3;
      if ( *((char *)v11 - 1) >= 0 )
        goto LABEL_129;
      v18 = sub_1648A40((__int64)v114);
      v20 = v19 + v18;
      v21 = 0;
      v109 = v20;
      if ( *((char *)v11 - 1) < 0 )
        v21 = sub_1648A40((__int64)v114);
      if ( !(unsigned int)((v109 - v21) >> 4) )
      {
LABEL_129:
        v22 = *(v11 - 6);
        if ( !*(_BYTE *)(v22 + 16) )
        {
          v128 = *(unsigned __int8 **)(v22 + 112);
          if ( (unsigned __int8)sub_1560260(&v128, -1, 36) )
            goto LABEL_8;
        }
      }
      if ( (unsigned __int8)sub_1560260(v11 + 4, -1, 37) )
        goto LABEL_8;
      if ( *((char *)v11 - 1) < 0 )
      {
        v23 = sub_1648A40((__int64)v114);
        v25 = v23 + v24;
        v26 = *((char *)v11 - 1) >= 0 ? 0LL : sub_1648A40((__int64)v114);
        if ( v26 != v25 )
          break;
      }
LABEL_119:
      v95 = *(v11 - 6);
      if ( *(_BYTE *)(v95 + 16) )
        goto LABEL_36;
      v128 = *(unsigned __int8 **)(v95 + 112);
      if ( !(unsigned __int8)sub_1560260(&v128, -1, 37) )
        goto LABEL_36;
LABEL_8:
      v11 = (__int64 *)v11[1];
      if ( v12 == v11 )
        goto LABEL_17;
    }
    while ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) <= 1u )
    {
      v26 += 16;
      if ( v25 == v26 )
        goto LABEL_119;
    }
LABEL_36:
    v27 = v11[1];
    v104 = v17;
    v28 = (unsigned __int8 **)v11;
    if ( v27 == v11[2] + 40 || !v27 )
      v29 = 0;
    else
      v29 = v27 - 24;
    v112 = v116 - 24;
    v30 = sub_1AA8CA0(v116 - 24, v29, 0, 0);
    v31 = *(_QWORD *)(v30 + 48);
    v32 = sub_157E9C0(v30);
    v128 = 0;
    v131 = v32;
    v132 = 0;
    v133 = 0;
    v134 = 0;
    v135 = 0;
    v129 = (_QWORD *)v30;
    v130 = (__int64 *)v31;
    if ( v31 != v30 + 40 )
    {
      if ( !v31 )
        BUG();
      v33 = *(unsigned __int8 **)(v31 + 24);
      v126[0] = v33;
      if ( v33 )
      {
        sub_1623A60((__int64)v126, (__int64)v33, 2);
        if ( v128 )
          sub_161E7C0((__int64)&v128, (__int64)v128);
        v128 = v126[0];
        if ( v126[0] )
          sub_1623210((__int64)v126, v126[0], (__int64)&v128);
      }
    }
    v34 = (__int64)*(v28 - 3);
    v125 = 257;
    v106 = v34;
    v127 = 257;
    v35 = sub_1648B60(64);
    v36 = v35;
    if ( v35 )
    {
      v110 = v35;
      sub_15F1EA0(v35, v106, 53, 0, 0, 0);
      *(_DWORD *)(v36 + 56) = 2;
      sub_164B780(v36, (__int64 *)v126);
      sub_1648880(v36, *(_DWORD *)(v36 + 56), 1);
    }
    else
    {
      v110 = 0;
    }
    if ( v129 )
    {
      v119 = v130;
      sub_157E9D0((__int64)(v129 + 5), v36);
      v37 = *v119;
      v38 = *(_QWORD *)(v36 + 24) & 7LL;
      *(_QWORD *)(v36 + 32) = v119;
      v37 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v36 + 24) = v37 | v38;
      *(_QWORD *)(v37 + 8) = v36 + 24;
      *v119 = *v119 & 7 | (v36 + 24);
    }
    sub_164B780(v110, v124);
    if ( v128 )
    {
      v123 = v128;
      sub_1623A60((__int64)&v123, (__int64)v128, 2);
      v41 = *(_QWORD *)(v36 + 48);
      v42 = v36 + 48;
      if ( v41 )
      {
        sub_161E7C0(v36 + 48, v41);
        v42 = v36 + 48;
      }
      v43 = v123;
      *(_QWORD *)(v36 + 48) = v123;
      if ( v43 )
        sub_1623210((__int64)&v123, v43, v42);
    }
    sub_164D160((__int64)v114, v36, a4, a5, a6, a7, v39, v40, a10, a11);
    v44 = *(_QWORD *)(v116 + 32);
    v126[0] = "call.sqrt";
    v98 = v44;
    v127 = 259;
    v120 = sub_157E9C0(v112);
    v117 = (_QWORD *)sub_22077B0(64);
    if ( v117 )
      sub_157FB60(v117, v120, (__int64)v126, v98, v30);
    v129 = v117;
    v130 = v117 + 5;
    v121 = (_QWORD *)sub_15F4880((__int64)v114);
    v127 = 257;
    if ( v129 )
    {
      v99 = (unsigned __int64 *)v130;
      sub_157E9D0((__int64)(v129 + 5), (__int64)v121);
      v45 = *v99;
      v46 = v121[3];
      v121[4] = v99;
      v45 &= 0xFFFFFFFFFFFFFFF8LL;
      v121[3] = v45 | v46 & 7;
      *(_QWORD *)(v45 + 8) = v121 + 3;
      *v99 = *v99 & 7 | (unsigned __int64)(v121 + 3);
    }
    sub_164B780((__int64)v121, (__int64 *)v126);
    if ( v128 )
    {
      v124[0] = (__int64)v128;
      sub_1623A60((__int64)v124, (__int64)v128, 2);
      v47 = v121[6];
      v48 = (__int64)(v121 + 6);
      if ( v47 )
      {
        sub_161E7C0((__int64)(v121 + 6), v47);
        v48 = (__int64)(v121 + 6);
      }
      v49 = (unsigned __int8 *)v124[0];
      v121[6] = v124[0];
      if ( v49 )
        sub_1623210((__int64)v124, v49, v48);
    }
    v127 = 257;
    v50 = sub_1648A60(56, 1u);
    v51 = (__int64)v50;
    if ( v50 )
    {
      v100 = v50;
      sub_15F8320((__int64)v50, v30, 0);
      v51 = (__int64)v100;
    }
    if ( v129 )
    {
      v101 = v51;
      v96 = v130;
      sub_157E9D0((__int64)(v129 + 5), v51);
      v51 = v101;
      v52 = *(_QWORD *)(v101 + 24);
      v53 = *v96;
      *(_QWORD *)(v101 + 32) = v96;
      v53 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v101 + 24) = v53 | v52 & 7;
      *(_QWORD *)(v53 + 8) = v101 + 24;
      *v96 = *v96 & 7 | (v101 + 24);
    }
    v102 = v51;
    sub_164B780(v51, (__int64 *)v126);
    if ( v128 )
    {
      v124[0] = (__int64)v128;
      sub_1623A60((__int64)v124, (__int64)v128, 2);
      v54 = v102;
      v55 = *(_QWORD *)(v102 + 48);
      v56 = v102 + 48;
      if ( v55 )
      {
        v97 = v102;
        v103 = v102 + 48;
        sub_161E7C0(v103, v55);
        v54 = v97;
        v56 = v103;
      }
      v57 = (unsigned __int8 *)v124[0];
      *(_QWORD *)(v54 + 48) = v124[0];
      if ( v57 )
        sub_1623210((__int64)v124, v57, v56);
    }
    v126[0] = v28[4];
    v58 = (__int64 *)sub_16498A0((__int64)v114);
    v126[0] = (unsigned __int8 *)sub_1563AB0((__int64 *)v126, v58, -1, 36);
    v28[4] = v126[0];
    v59 = (_QWORD *)sub_157EBA0(v112);
    sub_15F20C0(v59);
    v130 = v12;
    v129 = (_QWORD *)v112;
    v60 = sub_14A3020(a3);
    v127 = 257;
    if ( v60 )
    {
      v61 = sub_1289B20((__int64 *)&v128, 7u, v114, (__int64)v114, (__int64)v126, 0);
    }
    else
    {
      a4 = 0;
      v94 = sub_15A10B0(v106, 0.0);
      v61 = sub_1289B20(
              (__int64 *)&v128,
              3u,
              (_BYTE *)v114[-3 * (*((_DWORD *)v28 - 1) & 0xFFFFFFF)],
              v94,
              (__int64)v126,
              0);
    }
    v107 = v61;
    v127 = 257;
    v62 = sub_1648A60(56, 3u);
    v63 = v62;
    if ( v62 )
      sub_15F83E0((__int64)v62, v30, (__int64)v117, v107, 0);
    if ( v129 )
    {
      v108 = (unsigned __int64 *)v130;
      sub_157E9D0((__int64)(v129 + 5), (__int64)v63);
      v64 = *v108;
      v65 = v63[3] & 7LL;
      v63[4] = v108;
      v64 &= 0xFFFFFFFFFFFFFFF8LL;
      v63[3] = v64 | v65;
      *(_QWORD *)(v64 + 8) = v63 + 3;
      *v108 = *v108 & 7 | (unsigned __int64)(v63 + 3);
    }
    sub_164B780((__int64)v63, (__int64 *)v126);
    v70 = (__int64)v128;
    if ( v128 )
    {
      v124[0] = (__int64)v128;
      sub_1623A60((__int64)v124, (__int64)v128, 2);
      v71 = v63[6];
      if ( v71 )
        sub_161E7C0((__int64)(v63 + 6), v71);
      v70 = v124[0];
      v63[6] = v124[0];
      if ( v70 )
        sub_1623210((__int64)v124, (unsigned __int8 *)v70, (__int64)(v63 + 6));
    }
    v72 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    if ( v72 == *(_DWORD *)(v36 + 56) )
    {
      sub_15F55D0(v36, v70, v66, v67, v68, v69);
      v72 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    }
    v73 = (v72 + 1) & 0xFFFFFFF;
    v74 = v73 | *(_DWORD *)(v36 + 20) & 0xF0000000;
    *(_DWORD *)(v36 + 20) = v74;
    if ( (v74 & 0x40000000) != 0 )
      v75 = *(_QWORD *)(v36 - 8);
    else
      v75 = v110 - 24 * v73;
    v76 = (unsigned __int8 *)(v75 + 24LL * (unsigned int)(v73 - 1));
    if ( *(_QWORD *)v76 )
    {
      v77 = *((_QWORD *)v76 + 1);
      v78 = *((_QWORD *)v76 + 2) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v78 = v77;
      if ( v77 )
      {
        v70 = *(_QWORD *)(v77 + 16) & 3LL;
        *(_QWORD *)(v77 + 16) = v70 | v78;
      }
    }
    *(_QWORD *)v76 = v114;
    v79 = (__int64)*(v28 - 2);
    *((_QWORD *)v76 + 1) = v79;
    if ( v79 )
    {
      v70 = (unsigned __int64)(v76 + 8) | *(_QWORD *)(v79 + 16) & 3LL;
      *(_QWORD *)(v79 + 16) = v70;
    }
    *((_QWORD *)v76 + 2) = (unsigned __int64)(v28 - 2) | *((_QWORD *)v76 + 2) & 3LL;
    *(v28 - 2) = v76;
    v80 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    v81 = (unsigned int)(v80 - 1);
    if ( (*(_BYTE *)(v36 + 23) & 0x40) != 0 )
      v82 = *(_QWORD *)(v36 - 8);
    else
      v82 = v110 - 24 * v80;
    v83 = 3LL * *(unsigned int *)(v36 + 56);
    *(_QWORD *)(v82 + 8 * v81 + 24LL * *(unsigned int *)(v36 + 56) + 8) = v112;
    v84 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    if ( v84 == *(_DWORD *)(v36 + 56) )
    {
      sub_15F55D0(v36, v70, v83, v82, v68, v69);
      v84 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    }
    v85 = (v84 + 1) & 0xFFFFFFF;
    v86 = v85 | *(_DWORD *)(v36 + 20) & 0xF0000000;
    *(_DWORD *)(v36 + 20) = v86;
    if ( (v86 & 0x40000000) != 0 )
      v87 = *(_QWORD *)(v36 - 8);
    else
      v87 = v110 - 24 * v85;
    v88 = (_QWORD *)(v87 + 24LL * (unsigned int)(v85 - 1));
    if ( *v88 )
    {
      v89 = v88[1];
      v90 = v88[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v90 = v89;
      if ( v89 )
        *(_QWORD *)(v89 + 16) = *(_QWORD *)(v89 + 16) & 3LL | v90;
    }
    *v88 = v121;
    if ( v121 )
    {
      v91 = v121[1];
      v88[1] = v91;
      if ( v91 )
        *(_QWORD *)(v91 + 16) = (unsigned __int64)(v88 + 1) | *(_QWORD *)(v91 + 16) & 3LL;
      v88[2] = v88[2] & 3LL | (unsigned __int64)(v121 + 1);
      v121[1] = v88;
    }
    v92 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v36 + 23) & 0x40) != 0 )
      v93 = *(_QWORD *)(v36 - 8);
    else
      v93 = v110 - 24 * v92;
    *(_QWORD *)(v93 + 8LL * (unsigned int)(v92 - 1) + 24LL * *(unsigned int *)(v36 + 56) + 8) = v117;
    v118 = v30 + 24;
    if ( v128 )
      sub_161E7C0((__int64)&v128, (__int64)v128);
    v111 = v104;
  }
  return v111;
}
