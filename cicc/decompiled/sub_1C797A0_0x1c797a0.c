// Function: sub_1C797A0
// Address: 0x1c797a0
//
__int64 __fastcall sub_1C797A0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  unsigned int v20; // r12d
  unsigned __int64 v22; // rax
  __int64 v23; // r15
  int v24; // edx
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // r14
  _QWORD *v30; // r13
  double v31; // xmm4_8
  double v32; // xmm5_8
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  __int64 v41; // rcx
  __int64 v42; // rcx
  unsigned __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r15
  __int64 i; // r14
  unsigned int v47; // ecx
  __int64 v48; // r9
  unsigned int v49; // esi
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rsi
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned __int64 v59; // rax
  __int64 v60; // r15
  __int64 j; // r14
  unsigned int v62; // ecx
  __int64 v63; // r9
  unsigned int v64; // esi
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int64 v67; // rax
  __int64 v68; // rsi
  _QWORD *v69; // r15
  __int64 v70; // r15
  __int64 v71; // rax
  __int64 v72; // r15
  _QWORD *v73; // rax
  _QWORD *v74; // r14
  unsigned __int64 *v75; // r15
  __int64 v76; // rax
  unsigned __int64 v77; // rsi
  __int64 *v78; // r8
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  __int64 v81; // r15
  __int64 k; // r14
  unsigned int v83; // ecx
  __int64 v84; // r9
  unsigned int v85; // esi
  __int64 v86; // rax
  __int64 v87; // rdx
  int v88; // edx
  int v89; // r10d
  __int64 v90; // rsi
  unsigned __int8 *v91; // rsi
  __int64 v92; // rsi
  unsigned __int64 v93; // rcx
  __int64 v94; // rcx
  __int64 v95; // rcx
  unsigned __int64 v96; // rax
  __int64 v97; // rcx
  __int64 v98; // rsi
  unsigned __int64 v99; // rdx
  __int64 v100; // rdx
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 v103; // rcx
  unsigned __int64 v104; // rsi
  __int64 v105; // rcx
  unsigned __int64 v106; // rsi
  __int64 v107; // [rsp+8h] [rbp-158h]
  __int64 v108; // [rsp+10h] [rbp-150h]
  __int64 v109; // [rsp+18h] [rbp-148h]
  __int64 *v110; // [rsp+18h] [rbp-148h]
  __int64 *v111; // [rsp+18h] [rbp-148h]
  __int64 *v112; // [rsp+18h] [rbp-148h]
  __int64 v113; // [rsp+28h] [rbp-138h] BYREF
  __int64 v114; // [rsp+30h] [rbp-130h] BYREF
  __int64 v115; // [rsp+38h] [rbp-128h] BYREF
  __int64 v116; // [rsp+40h] [rbp-120h] BYREF
  __int64 v117; // [rsp+48h] [rbp-118h] BYREF
  __int64 v118; // [rsp+50h] [rbp-110h] BYREF
  __int64 v119; // [rsp+58h] [rbp-108h] BYREF
  char v120[8]; // [rsp+60h] [rbp-100h] BYREF
  __int64 v121; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v122; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v123; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v124; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v125; // [rsp+88h] [rbp-D8h] BYREF
  __int64 v126; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v127; // [rsp+98h] [rbp-C8h] BYREF
  __int64 v128; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v129; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v130; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v131; // [rsp+B8h] [rbp-A8h] BYREF
  __int64 v132[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int16 v133; // [rsp+D0h] [rbp-90h]
  __int64 v134; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v135; // [rsp+E8h] [rbp-78h]
  unsigned __int64 *v136; // [rsp+F0h] [rbp-70h]
  __int64 v137; // [rsp+F8h] [rbp-68h]
  __int64 v138; // [rsp+100h] [rbp-60h]
  int v139; // [rsp+108h] [rbp-58h]
  __int64 v140; // [rsp+110h] [rbp-50h]
  __int64 v141; // [rsp+118h] [rbp-48h]

  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(_QWORD *)(a1 + 40);
  v113 = 0;
  v114 = 0;
  v12 = *(_QWORD *)(a1 + 160);
  v13 = *(unsigned int *)(v10 + 48);
  v14 = *(_QWORD *)a1;
  if ( !(_DWORD)v13 )
    goto LABEL_135;
  v15 = *(_QWORD *)(a1 + 168);
  v16 = *(_QWORD *)(v10 + 32);
  v17 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v18 = (__int64 *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( v15 != *v18 )
  {
    v88 = 1;
    while ( v19 != -8 )
    {
      v89 = v88 + 1;
      v17 = (v13 - 1) & (v88 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v15 == *v18 )
        goto LABEL_3;
      v88 = v89;
    }
LABEL_135:
    BUG();
  }
LABEL_3:
  if ( v18 == (__int64 *)(v16 + 16 * v13) )
    goto LABEL_135;
  if ( v12 != **(_QWORD **)(v18[1] + 8) )
    return 0;
  v22 = sub_157EBA0(v12);
  if ( *(_BYTE *)(v22 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v22 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v23 = *(_QWORD *)(v22 - 72);
  if ( *(_BYTE *)(v23 + 16) != 75 )
    return 0;
  v24 = *(unsigned __int16 *)(v23 + 18);
  BYTE1(v24) &= ~0x80u;
  if ( v24 == 33 )
  {
    if ( v15 != *(_QWORD *)(v22 - 48) )
      return 0;
  }
  else if ( v24 != 32 || v15 != *(_QWORD *)(v22 - 24) )
  {
    return 0;
  }
  if ( v11 == *(_QWORD *)(v23 - 48) && sub_13FC1A0(v14, *(_QWORD *)(v23 - 24)) )
  {
    v107 = *(_QWORD *)(v23 - 24);
  }
  else
  {
    if ( v11 != *(_QWORD *)(v23 - 24) || !sub_13FC1A0(v14, *(_QWORD *)(v23 - 48)) )
      return 0;
    v107 = *(_QWORD *)(v23 - 48);
  }
  if ( !(unsigned __int8)sub_1C73930(
                           *(_QWORD *)a1,
                           *(_QWORD *)(a1 + 184),
                           *(_QWORD *)(a1 + 152),
                           *(_QWORD *)(a1 + 160),
                           *(_QWORD *)(a1 + 168),
                           *(_QWORD *)(a1 + 176),
                           *(_QWORD *)(a1 + 192),
                           *(_QWORD *)(a1 + 56),
                           *(_QWORD *)(a1 + 64),
                           &v113,
                           &v114) )
    return 0;
  v20 = sub_1C75600(
          *(_QWORD *)a1,
          *(_QWORD *)(a1 + 184),
          *(_QWORD *)(a1 + 160),
          *(_QWORD *)(a1 + 168),
          *(_QWORD *)(a1 + 176),
          *(_QWORD *)(a1 + 192),
          a2,
          a3,
          a4,
          a5,
          v25,
          v26,
          a8,
          a9);
  if ( !(_BYTE)v20 )
    return 0;
  v27 = *(_QWORD *)(a1 + 40);
  if ( !v27 )
    return 0;
  v28 = *(_QWORD *)(a1 + 160);
  if ( v28 != *(_QWORD *)(v27 + 40) )
    return 0;
  v29 = *(_QWORD *)(a1 + 168);
  v108 = *(_QWORD *)(a1 + 192);
  v109 = *(_QWORD *)(a1 + 176);
  v30 = sub_1C74210(v27, v29, v109);
  if ( !v30 )
    return 0;
  sub_1C77080(
    *(_QWORD *)a1,
    &v128,
    &v129,
    1,
    *(__int64 **)(a1 + 8),
    *(_QWORD *)(a1 + 208),
    a2,
    a3,
    a4,
    a5,
    v31,
    v32,
    a8,
    a9,
    *(_QWORD *)(a1 + 16),
    (__int64)v30,
    &v127,
    *(_QWORD *)(a1 + 184),
    *(_QWORD *)(a1 + 152),
    v28,
    v29,
    v109,
    v108,
    *(_QWORD *)(a1 + 200),
    &v115,
    &v116,
    (__int64)&v117,
    &v118,
    &v119,
    (__int64)v120,
    &v121,
    &v122,
    (__int64)&v123,
    &v124,
    &v125,
    &v126);
  v35 = *(unsigned int *)(a1 + 224);
  if ( (unsigned int)v35 >= *(_DWORD *)(a1 + 228) )
  {
    sub_16CD150(a1 + 216, (const void *)(a1 + 232), 0, 8, v33, v34);
    v35 = *(unsigned int *)(a1 + 224);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v35) = v129;
  v36 = *(_QWORD *)(a1 + 160);
  ++*(_DWORD *)(a1 + 224);
  v37 = sub_157EBA0(v36);
  v38 = *(_QWORD *)(v37 - 24);
  if ( *(_QWORD *)(a1 + 168) == v38 )
  {
    v97 = *(_QWORD *)(v37 - 48);
    if ( v97 )
    {
      if ( v38 )
      {
        v98 = *(_QWORD *)(v37 - 16);
        v99 = *(_QWORD *)(v37 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v99 = v98;
        if ( v98 )
          *(_QWORD *)(v98 + 16) = *(_QWORD *)(v98 + 16) & 3LL | v99;
      }
      *(_QWORD *)(v37 - 24) = v97;
      v100 = *(_QWORD *)(v97 + 8);
      *(_QWORD *)(v37 - 16) = v100;
      if ( v100 )
        *(_QWORD *)(v100 + 16) = (v37 - 16) | *(_QWORD *)(v100 + 16) & 3LL;
      v101 = *(_QWORD *)(v37 - 8);
      v102 = v37 - 24;
      *(_QWORD *)(v102 + 16) = (v97 + 8) | v101 & 3;
      *(_QWORD *)(v97 + 8) = v102;
    }
    else if ( v38 )
    {
      v103 = *(_QWORD *)(v37 - 16);
      v104 = *(_QWORD *)(v37 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v104 = v103;
      if ( v103 )
        *(_QWORD *)(v103 + 16) = v104 | *(_QWORD *)(v103 + 16) & 3LL;
      *(_QWORD *)(v37 - 24) = 0;
    }
  }
  else
  {
    if ( *(_QWORD *)(v37 - 48) )
    {
      v39 = *(_QWORD *)(v37 - 40);
      v40 = *(_QWORD *)(v37 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v40 = v39;
      if ( v39 )
        *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
    }
    *(_QWORD *)(v37 - 48) = v38;
    if ( v38 )
    {
      v41 = *(_QWORD *)(v38 + 8);
      *(_QWORD *)(v37 - 40) = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = (v37 - 40) | *(_QWORD *)(v41 + 16) & 3LL;
      v42 = *(_QWORD *)(v37 - 32);
      v43 = v37 - 48;
      *(_QWORD *)(v43 + 16) = (v38 + 8) | v42 & 3;
      *(_QWORD *)(v38 + 8) = v43;
    }
  }
  v44 = *(_QWORD *)(a1 + 168);
  v45 = *(_QWORD *)(v44 + 48);
  for ( i = v44 + 40; i != v45; v45 = *(_QWORD *)(v45 + 8) )
  {
    if ( !v45 )
      BUG();
    if ( *(_BYTE *)(v45 - 8) != 77 )
      break;
    v47 = *(_DWORD *)(v45 - 4) & 0xFFFFFFF;
    if ( v47 )
    {
      v48 = v45 - 24;
      v49 = 0;
      v50 = 24LL * *(unsigned int *)(v45 + 32) + 8;
      while ( 1 )
      {
        v51 = v45 - 24 - 24LL * v47;
        if ( (*(_BYTE *)(v45 - 1) & 0x40) != 0 )
          v51 = *(_QWORD *)(v45 - 32);
        if ( *(_QWORD *)(a1 + 160) == *(_QWORD *)(v51 + v50) )
          break;
        ++v49;
        v50 += 8;
        if ( v47 == v49 )
        {
          v49 = -1;
          break;
        }
      }
    }
    else
    {
      v49 = -1;
      v48 = v45 - 24;
    }
    sub_15F5350(v48, v49, 1);
  }
  v52 = sub_157EBA0(v123);
  v53 = *(_QWORD *)(v52 - 24);
  v54 = *(_QWORD *)(v52 - 48);
  if ( v124 == v53 )
  {
    if ( v54 )
    {
      if ( v53 )
      {
        v92 = *(_QWORD *)(v52 - 16);
        v93 = *(_QWORD *)(v52 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v93 = v92;
        if ( v92 )
          *(_QWORD *)(v92 + 16) = *(_QWORD *)(v92 + 16) & 3LL | v93;
      }
      *(_QWORD *)(v52 - 24) = v54;
      v94 = *(_QWORD *)(v54 + 8);
      *(_QWORD *)(v52 - 16) = v94;
      if ( v94 )
        *(_QWORD *)(v94 + 16) = (v52 - 16) | *(_QWORD *)(v94 + 16) & 3LL;
      v95 = *(_QWORD *)(v52 - 8);
      v96 = v52 - 24;
      *(_QWORD *)(v96 + 16) = (v54 + 8) | v95 & 3;
      *(_QWORD *)(v54 + 8) = v96;
    }
    else if ( v53 )
    {
      v105 = *(_QWORD *)(v52 - 16);
      v106 = *(_QWORD *)(v52 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v106 = v105;
      if ( v105 )
        *(_QWORD *)(v105 + 16) = v106 | *(_QWORD *)(v105 + 16) & 3LL;
      *(_QWORD *)(v52 - 24) = 0;
    }
  }
  else
  {
    if ( v54 )
    {
      v55 = *(_QWORD *)(v52 - 40);
      v56 = *(_QWORD *)(v52 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v56 = v55;
      if ( v55 )
        *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
    }
    *(_QWORD *)(v52 - 48) = v53;
    if ( v53 )
    {
      v57 = *(_QWORD *)(v53 + 8);
      *(_QWORD *)(v52 - 40) = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = (v52 - 40) | *(_QWORD *)(v57 + 16) & 3LL;
      v58 = *(_QWORD *)(v52 - 32);
      v59 = v52 - 48;
      *(_QWORD *)(v59 + 16) = (v53 + 8) | v58 & 3;
      *(_QWORD *)(v53 + 8) = v59;
    }
  }
  v60 = *(_QWORD *)(v124 + 48);
  for ( j = v124 + 40; j != v60; v60 = *(_QWORD *)(v60 + 8) )
  {
    if ( !v60 )
      BUG();
    if ( *(_BYTE *)(v60 - 8) != 77 )
      break;
    v62 = *(_DWORD *)(v60 - 4) & 0xFFFFFFF;
    if ( v62 )
    {
      v63 = v60 - 24;
      v64 = 0;
      v65 = 24LL * *(unsigned int *)(v60 + 32) + 8;
      while ( 1 )
      {
        v66 = v60 - 24 - 24LL * v62;
        if ( (*(_BYTE *)(v60 - 1) & 0x40) != 0 )
          v66 = *(_QWORD *)(v60 - 32);
        if ( v123 == *(_QWORD *)(v66 + v65) )
          break;
        ++v64;
        v65 += 8;
        if ( v62 == v64 )
        {
          v64 = -1;
          break;
        }
      }
    }
    else
    {
      v64 = -1;
      v63 = v60 - 24;
    }
    sub_15F5350(v63, v64, 1);
  }
  v67 = sub_157EBA0(v118);
  v68 = *(_QWORD *)(v67 + 48);
  v69 = (_QWORD *)v67;
  v130 = v68;
  if ( v68 )
    sub_1623A60((__int64)&v130, v68, 2);
  sub_15F20C0(v69);
  v70 = v118;
  v71 = sub_157E9C0(v118);
  v135 = v70;
  v137 = v71;
  v136 = (unsigned __int64 *)(v70 + 40);
  v72 = v119;
  v134 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v133 = 257;
  v73 = sub_1648A60(56, 1u);
  v74 = v73;
  if ( v73 )
    sub_15F8320((__int64)v73, v72, 0);
  if ( v135 )
  {
    v75 = v136;
    sub_157E9D0(v135 + 40, (__int64)v74);
    v76 = v74[3];
    v77 = *v75;
    v74[4] = v75;
    v77 &= 0xFFFFFFFFFFFFFFF8LL;
    v74[3] = v77 | v76 & 7;
    *(_QWORD *)(v77 + 8) = v74 + 3;
    *v75 = *v75 & 7 | (unsigned __int64)(v74 + 3);
  }
  sub_164B780((__int64)v74, v132);
  v78 = v74 + 6;
  if ( v134 )
  {
    v131 = v134;
    sub_1623A60((__int64)&v131, v134, 2);
    v79 = v74[6];
    v78 = v74 + 6;
    if ( v79 )
    {
      sub_161E7C0((__int64)(v74 + 6), v79);
      v78 = v74 + 6;
    }
    v80 = (unsigned __int8 *)v131;
    v74[6] = v131;
    if ( v80 )
    {
      v110 = v78;
      sub_1623210((__int64)&v131, v80, (__int64)v78);
      v78 = v110;
    }
  }
  v132[0] = v130;
  if ( v130 )
  {
    v111 = v78;
    sub_1623A60((__int64)v132, v130, 2);
    v78 = v111;
    if ( v111 == v132 )
    {
      if ( v132[0] )
        sub_161E7C0((__int64)v111, v132[0]);
      goto LABEL_75;
    }
    v90 = v74[6];
    if ( !v90 )
    {
LABEL_99:
      v91 = (unsigned __int8 *)v132[0];
      v74[6] = v132[0];
      if ( v91 )
        sub_1623210((__int64)v132, v91, (__int64)v78);
      goto LABEL_75;
    }
LABEL_98:
    v112 = v78;
    sub_161E7C0((__int64)v78, v90);
    v78 = v112;
    goto LABEL_99;
  }
  if ( v78 != v132 )
  {
    v90 = v74[6];
    if ( v90 )
      goto LABEL_98;
  }
LABEL_75:
  v81 = *(_QWORD *)(v117 + 48);
  for ( k = v117 + 40; k != v81; v81 = *(_QWORD *)(v81 + 8) )
  {
    if ( !v81 )
      BUG();
    if ( *(_BYTE *)(v81 - 8) != 77 )
      break;
    v83 = *(_DWORD *)(v81 - 4) & 0xFFFFFFF;
    if ( v83 )
    {
      v84 = v81 - 24;
      v85 = 0;
      v86 = 24LL * *(unsigned int *)(v81 + 32) + 8;
      while ( 1 )
      {
        v87 = v81 - 24 - 24LL * v83;
        if ( (*(_BYTE *)(v81 - 1) & 0x40) != 0 )
          v87 = *(_QWORD *)(v81 - 32);
        if ( v118 == *(_QWORD *)(v87 + v86) )
          break;
        ++v85;
        v86 += 8;
        if ( v83 == v85 )
        {
          v85 = -1;
          break;
        }
      }
    }
    else
    {
      v85 = -1;
      v84 = v81 - 24;
    }
    sub_15F5350(v84, v85, 1);
  }
  sub_1C75B60(*(_QWORD *)(a1 + 184), *(_QWORD *)(a1 + 168), v115, v121, v113, v114, v107, (__int64)v30, v127);
  if ( v134 )
    sub_161E7C0((__int64)&v134, v134);
  if ( v130 )
    sub_161E7C0((__int64)&v130, v130);
  return v20;
}
