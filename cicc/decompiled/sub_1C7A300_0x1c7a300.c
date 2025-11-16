// Function: sub_1C7A300
// Address: 0x1c7a300
//
__int64 __fastcall sub_1C7A300(
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
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  unsigned int v20; // r15d
  unsigned __int64 v22; // rax
  __int64 v23; // r15
  int v24; // edx
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // r13
  double v31; // xmm4_8
  double v32; // xmm5_8
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rsi
  unsigned __int64 v41; // rcx
  __int64 v42; // rcx
  __int64 v43; // rcx
  unsigned __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r12
  __int64 i; // r14
  unsigned int v48; // edi
  __int64 v49; // r8
  unsigned int v50; // esi
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // rsi
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rdx
  unsigned __int64 v60; // rax
  __int64 v61; // r12
  __int64 j; // r14
  unsigned int v63; // edi
  __int64 v64; // r8
  unsigned int v65; // esi
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 *v69; // r12
  __int64 *v70; // r13
  _BYTE *v71; // r8
  __int64 v72; // rax
  _BYTE *v73; // rsi
  __int64 v74; // rax
  __int64 v75; // r14
  _QWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r13
  _BYTE *v79; // rax
  _QWORD *v80; // rdx
  int v81; // edx
  int v82; // r10d
  __int64 v83; // rdx
  __int64 v84; // rsi
  __int64 *v85; // r13
  __int64 *v86; // r12
  __int64 v87; // rax
  _BYTE *v88; // rdx
  __int64 v89; // rax
  __int64 v90; // r14
  _QWORD *v91; // rax
  __int64 v92; // rax
  __int64 v93; // r13
  _BYTE *v94; // rax
  _QWORD *v95; // rdx
  __int64 v96; // rdx
  __int64 v97; // rsi
  unsigned __int64 v98; // rax
  __int64 v99; // rsi
  _QWORD *v100; // r12
  __int64 v101; // r12
  __int64 v102; // rax
  __int64 v103; // r12
  _QWORD *v104; // rax
  _QWORD *v105; // r13
  unsigned __int64 *v106; // r12
  __int64 v107; // rax
  unsigned __int64 v108; // rcx
  __int64 *v109; // r8
  __int64 v110; // rsi
  unsigned __int8 *v111; // rsi
  __int64 v112; // r12
  __int64 k; // r13
  unsigned int v114; // ecx
  __int64 v115; // r8
  unsigned int v116; // esi
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rsi
  unsigned __int8 *v120; // rsi
  __int64 v121; // rsi
  unsigned __int64 v122; // rdx
  unsigned __int64 v123; // r8
  __int64 v124; // rdx
  __int64 v125; // rsi
  unsigned __int64 v126; // rcx
  unsigned __int64 v127; // r8
  __int64 v128; // rcx
  __int64 v129; // [rsp+0h] [rbp-180h]
  _QWORD *v130; // [rsp+8h] [rbp-178h]
  __int64 v131; // [rsp+10h] [rbp-170h]
  __int64 v132; // [rsp+10h] [rbp-170h]
  __int64 v133; // [rsp+18h] [rbp-168h]
  __int64 v134; // [rsp+18h] [rbp-168h]
  __int64 v135; // [rsp+18h] [rbp-168h]
  __int64 *v136; // [rsp+18h] [rbp-168h]
  __int64 *v137; // [rsp+18h] [rbp-168h]
  __int64 *v138; // [rsp+18h] [rbp-168h]
  __int64 v139; // [rsp+28h] [rbp-158h] BYREF
  __int64 v140; // [rsp+30h] [rbp-150h] BYREF
  __int64 v141; // [rsp+38h] [rbp-148h] BYREF
  __int64 v142; // [rsp+40h] [rbp-140h] BYREF
  __int64 v143; // [rsp+48h] [rbp-138h] BYREF
  __int64 v144; // [rsp+50h] [rbp-130h] BYREF
  __int64 v145; // [rsp+58h] [rbp-128h] BYREF
  char v146[8]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v147; // [rsp+68h] [rbp-118h] BYREF
  __int64 v148; // [rsp+70h] [rbp-110h] BYREF
  __int64 v149; // [rsp+78h] [rbp-108h] BYREF
  __int64 v150; // [rsp+80h] [rbp-100h] BYREF
  __int64 v151; // [rsp+88h] [rbp-F8h] BYREF
  __int64 v152; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v153; // [rsp+98h] [rbp-E8h] BYREF
  __int64 v154; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v155; // [rsp+A8h] [rbp-D8h] BYREF
  __int64 v156; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v157; // [rsp+B8h] [rbp-C8h] BYREF
  _BYTE *v158; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE *v159; // [rsp+C8h] [rbp-B8h]
  _BYTE *v160; // [rsp+D0h] [rbp-B0h]
  __int64 v161[2]; // [rsp+E0h] [rbp-A0h] BYREF
  __int16 v162; // [rsp+F0h] [rbp-90h]
  __int64 v163; // [rsp+100h] [rbp-80h] BYREF
  __int64 v164; // [rsp+108h] [rbp-78h]
  unsigned __int64 *v165; // [rsp+110h] [rbp-70h]
  __int64 v166; // [rsp+118h] [rbp-68h]
  __int64 v167; // [rsp+120h] [rbp-60h]
  int v168; // [rsp+128h] [rbp-58h]
  __int64 v169; // [rsp+130h] [rbp-50h]
  __int64 v170; // [rsp+138h] [rbp-48h]

  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(_QWORD *)(a1 + 40);
  v139 = 0;
  v140 = 0;
  v12 = *(_QWORD *)(a1 + 160);
  v13 = *(unsigned int *)(v10 + 48);
  v14 = *(_QWORD *)a1;
  if ( !(_DWORD)v13 )
    goto LABEL_214;
  v15 = *(_QWORD *)(a1 + 168);
  v16 = *(_QWORD *)(v10 + 32);
  v17 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v18 = (__int64 *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( v15 != *v18 )
  {
    v81 = 1;
    while ( v19 != -8 )
    {
      v82 = v81 + 1;
      v17 = (v13 - 1) & (v81 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v15 == *v18 )
        goto LABEL_3;
      v81 = v82;
    }
LABEL_214:
    BUG();
  }
LABEL_3:
  if ( v18 == (__int64 *)(v16 + 16 * v13) )
    goto LABEL_214;
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
    if ( v15 != *(_QWORD *)(v22 - 24) )
      return 0;
  }
  else if ( v24 != 32 || v15 != *(_QWORD *)(v22 - 48) )
  {
    return 0;
  }
  if ( v11 == *(_QWORD *)(v23 - 48) && sub_13FC1A0(v14, *(_QWORD *)(v23 - 24)) )
  {
    v129 = *(_QWORD *)(v23 - 24);
  }
  else
  {
    if ( v11 != *(_QWORD *)(v23 - 24) || !sub_13FC1A0(v14, *(_QWORD *)(v23 - 48)) )
      return 0;
    v129 = *(_QWORD *)(v23 - 48);
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
                           &v139,
                           &v140) )
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
  v29 = *(_QWORD *)(a1 + 176);
  v30 = *(_QWORD *)(a1 + 168);
  v133 = *(_QWORD *)(a1 + 192);
  v130 = sub_1C74210(v27, v30, v29);
  if ( !v130 )
    return 0;
  sub_1C77080(
    *(_QWORD *)a1,
    &v154,
    &v155,
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
    (__int64)v130,
    &v153,
    *(_QWORD *)(a1 + 184),
    *(_QWORD *)(a1 + 152),
    v28,
    v30,
    v29,
    v133,
    *(_QWORD *)(a1 + 200),
    &v141,
    &v142,
    (__int64)&v143,
    &v144,
    &v145,
    (__int64)v146,
    &v147,
    &v148,
    (__int64)&v149,
    &v150,
    &v151,
    &v152);
  v35 = *(unsigned int *)(a1 + 224);
  if ( (unsigned int)v35 >= *(_DWORD *)(a1 + 228) )
  {
    sub_16CD150(a1 + 216, (const void *)(a1 + 232), 0, 8, v33, v34);
    v35 = *(unsigned int *)(a1 + 224);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v35) = v155;
  v36 = *(_QWORD *)(a1 + 160);
  ++*(_DWORD *)(a1 + 224);
  v37 = sub_157EBA0(v36);
  v38 = *(_QWORD *)(v37 - 24);
  if ( v38 )
  {
    v39 = *(_QWORD *)(a1 + 168);
    if ( v38 == v39 )
    {
      v38 = *(_QWORD *)(v37 - 48);
      if ( !v38 )
      {
        *(_QWORD *)(v37 - 48) = v39;
LABEL_30:
        v42 = *(_QWORD *)(v39 + 8);
        *(_QWORD *)(v37 - 40) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = (v37 - 40) | *(_QWORD *)(v42 + 16) & 3LL;
        v43 = *(_QWORD *)(v37 - 32);
        v44 = v37 - 48;
        *(_QWORD *)(v44 + 16) = (v39 + 8) | v43 & 3;
        *(_QWORD *)(v39 + 8) = v44;
LABEL_33:
        v45 = *(_QWORD *)(a1 + 168);
        goto LABEL_34;
      }
LABEL_27:
      v40 = *(_QWORD *)(v37 - 40);
      v41 = *(_QWORD *)(v37 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v41 = v40;
      if ( v40 )
        *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
      *(_QWORD *)(v37 - 48) = v39;
      if ( !v39 )
        goto LABEL_33;
      goto LABEL_30;
    }
    v125 = *(_QWORD *)(v37 - 16);
    v126 = *(_QWORD *)(v37 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v126 = v125;
    if ( v125 )
      *(_QWORD *)(v125 + 16) = *(_QWORD *)(v125 + 16) & 3LL | v126;
    *(_QWORD *)(v37 - 24) = v39;
    if ( !v39 )
      goto LABEL_33;
    v127 = v37 - 24;
    goto LABEL_181;
  }
  v39 = *(_QWORD *)(a1 + 168);
  v45 = v39;
  if ( v39 )
  {
    *(_QWORD *)(v37 - 24) = v39;
    v127 = v37 - 24;
LABEL_181:
    v128 = *(_QWORD *)(v39 + 8);
    *(_QWORD *)(v37 - 16) = v128;
    if ( v128 )
      *(_QWORD *)(v128 + 16) = (v37 - 16) | *(_QWORD *)(v128 + 16) & 3LL;
    *(_QWORD *)(v37 - 8) = (v39 + 8) | *(_QWORD *)(v37 - 8) & 3LL;
    *(_QWORD *)(v39 + 8) = v127;
    v45 = *(_QWORD *)(a1 + 168);
    goto LABEL_34;
  }
  v38 = *(_QWORD *)(v37 - 48);
  if ( v38 )
    goto LABEL_27;
LABEL_34:
  v46 = *(_QWORD *)(v45 + 48);
  for ( i = v45 + 40; i != v46; v46 = *(_QWORD *)(v46 + 8) )
  {
    if ( !v46 )
      BUG();
    if ( *(_BYTE *)(v46 - 8) != 77 )
      break;
    v48 = *(_DWORD *)(v46 - 4) & 0xFFFFFFF;
    if ( v48 )
    {
      v49 = v46 - 24;
      v50 = 0;
      v51 = 24LL * *(unsigned int *)(v46 + 32) + 8;
      while ( 1 )
      {
        v52 = v46 - 24 - 24LL * v48;
        if ( (*(_BYTE *)(v46 - 1) & 0x40) != 0 )
          v52 = *(_QWORD *)(v46 - 32);
        if ( v38 == *(_QWORD *)(v52 + v51) )
          break;
        ++v50;
        v51 += 8;
        if ( v48 == v50 )
        {
          v50 = -1;
          break;
        }
      }
    }
    else
    {
      v50 = -1;
      v49 = v46 - 24;
    }
    sub_15F5350(v49, v50, 1);
  }
  v53 = sub_157EBA0(v149);
  v54 = v150;
  v55 = *(_QWORD *)(v53 - 24);
  if ( v55 )
  {
    if ( v55 == v150 )
    {
      v55 = *(_QWORD *)(v53 - 48);
      if ( !v55 )
      {
        *(_QWORD *)(v53 - 48) = v150;
LABEL_50:
        v58 = *(_QWORD *)(v54 + 8);
        *(_QWORD *)(v53 - 40) = v58;
        if ( v58 )
          *(_QWORD *)(v58 + 16) = (v53 - 40) | *(_QWORD *)(v58 + 16) & 3LL;
        v59 = *(_QWORD *)(v53 - 32);
        v60 = v53 - 48;
        *(_QWORD *)(v60 + 16) = (v54 + 8) | v59 & 3;
        *(_QWORD *)(v54 + 8) = v60;
LABEL_53:
        v54 = v150;
        goto LABEL_54;
      }
LABEL_47:
      v56 = *(_QWORD *)(v53 - 40);
      v57 = *(_QWORD *)(v53 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v57 = v56;
      if ( v56 )
        *(_QWORD *)(v56 + 16) = *(_QWORD *)(v56 + 16) & 3LL | v57;
      *(_QWORD *)(v53 - 48) = v54;
      if ( !v54 )
        goto LABEL_53;
      goto LABEL_50;
    }
    v121 = *(_QWORD *)(v53 - 16);
    v122 = *(_QWORD *)(v53 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v122 = v121;
    if ( v121 )
      *(_QWORD *)(v121 + 16) = *(_QWORD *)(v121 + 16) & 3LL | v122;
    *(_QWORD *)(v53 - 24) = v54;
    if ( !v54 )
      goto LABEL_53;
    v123 = v53 - 24;
    goto LABEL_174;
  }
  if ( v150 )
  {
    *(_QWORD *)(v53 - 24) = v150;
    v123 = v53 - 24;
LABEL_174:
    v124 = *(_QWORD *)(v54 + 8);
    *(_QWORD *)(v53 - 16) = v124;
    if ( v124 )
      *(_QWORD *)(v124 + 16) = (v53 - 16) | *(_QWORD *)(v124 + 16) & 3LL;
    *(_QWORD *)(v53 - 8) = (v54 + 8) | *(_QWORD *)(v53 - 8) & 3LL;
    *(_QWORD *)(v54 + 8) = v123;
    v54 = v150;
    goto LABEL_54;
  }
  v55 = *(_QWORD *)(v53 - 48);
  if ( v55 )
    goto LABEL_47;
LABEL_54:
  v61 = *(_QWORD *)(v54 + 48);
  for ( j = v54 + 40; j != v61; v61 = *(_QWORD *)(v61 + 8) )
  {
    if ( !v61 )
      BUG();
    if ( *(_BYTE *)(v61 - 8) != 77 )
      break;
    v63 = *(_DWORD *)(v61 - 4) & 0xFFFFFFF;
    if ( v63 )
    {
      v64 = v61 - 24;
      v65 = 0;
      v66 = 24LL * *(unsigned int *)(v61 + 32) + 8;
      while ( 1 )
      {
        v67 = v61 - 24 - 24LL * v63;
        if ( (*(_BYTE *)(v61 - 1) & 0x40) != 0 )
          v67 = *(_QWORD *)(v61 - 32);
        if ( v55 == *(_QWORD *)(v67 + v66) )
          break;
        ++v65;
        v66 += 8;
        if ( v63 == v65 )
        {
          v65 = -1;
          break;
        }
      }
    }
    else
    {
      v65 = -1;
      v64 = v61 - 24;
    }
    sub_15F5350(v64, v65, 1);
  }
  v158 = 0;
  v68 = *(_QWORD *)a1;
  v159 = 0;
  v69 = *(__int64 **)(v68 + 32);
  v70 = *(__int64 **)(v68 + 40);
  v160 = 0;
  if ( v70 == v69 )
  {
    v85 = *(__int64 **)(v155 + 40);
    v86 = *(__int64 **)(v155 + 32);
    if ( v85 == v86 )
      goto LABEL_131;
    v71 = 0;
  }
  else
  {
    v71 = 0;
    do
    {
      v72 = *v69;
      if ( *v69 != *(_QWORD *)(a1 + 184)
        && v72 != *(_QWORD *)(a1 + 152)
        && v72 != *(_QWORD *)(a1 + 160)
        && v72 != *(_QWORD *)(a1 + 168)
        && v72 != *(_QWORD *)(a1 + 176)
        && v72 != *(_QWORD *)(a1 + 192) )
      {
        if ( v160 == v71 )
        {
          sub_1292090((__int64)&v158, v71, v69);
          v71 = v159;
        }
        else
        {
          if ( v71 )
          {
            *(_QWORD *)v71 = v72;
            v71 = v159;
          }
          v71 += 8;
          v159 = v71;
        }
      }
      ++v69;
    }
    while ( v69 != v70 );
    v73 = v158;
    v74 = (v71 - v158) >> 3;
    if ( (_DWORD)v74 )
    {
      v75 = 0;
      v134 = 8LL * (unsigned int)(v74 - 1) + 8;
      while ( 1 )
      {
        v78 = *(_QWORD *)a1;
        v163 = *(_QWORD *)&v73[v75];
        v79 = sub_1C73C60(*(_QWORD **)(v78 + 32), *(_QWORD *)(v78 + 40), &v163);
        sub_1977D00(v78 + 32, v79);
        v76 = *(_QWORD **)(v78 + 64);
        if ( *(_QWORD **)(v78 + 72) == v76 )
        {
          v80 = &v76[*(unsigned int *)(v78 + 84)];
          if ( v76 == v80 )
          {
LABEL_96:
            v76 = v80;
          }
          else
          {
            while ( v163 != *v76 )
            {
              if ( v80 == ++v76 )
                goto LABEL_96;
            }
          }
          goto LABEL_87;
        }
        v131 = v163;
        v76 = sub_16CC9F0(v78 + 56, v163);
        if ( v131 == *v76 )
          break;
        v77 = *(_QWORD *)(v78 + 72);
        if ( v77 == *(_QWORD *)(v78 + 64) )
        {
          v76 = (_QWORD *)(v77 + 8LL * *(unsigned int *)(v78 + 84));
          v80 = v76;
          goto LABEL_87;
        }
LABEL_81:
        v73 = v158;
        v75 += 8;
        if ( v134 == v75 )
        {
          v71 = v159;
          goto LABEL_99;
        }
      }
      v83 = *(_QWORD *)(v78 + 72);
      if ( v83 == *(_QWORD *)(v78 + 64) )
        v84 = *(unsigned int *)(v78 + 84);
      else
        v84 = *(unsigned int *)(v78 + 80);
      v80 = (_QWORD *)(v83 + 8 * v84);
LABEL_87:
      if ( v80 != v76 )
      {
        *v76 = -2;
        ++*(_DWORD *)(v78 + 88);
      }
      goto LABEL_81;
    }
LABEL_99:
    v85 = *(__int64 **)(v155 + 40);
    v86 = *(__int64 **)(v155 + 32);
    if ( v71 != v73 )
    {
      v159 = v73;
      if ( v86 == v85 )
        goto LABEL_131;
      goto LABEL_101;
    }
    if ( v86 == v85 )
    {
      v88 = v73;
      goto LABEL_113;
    }
  }
  v73 = v71;
  do
  {
LABEL_101:
    v87 = *v86;
    if ( *v86 != v147 && v87 != v148 && v87 != v149 && v87 != v150 && v87 != v151 && v87 != v152 )
    {
      if ( v160 == v73 )
      {
        sub_1292090((__int64)&v158, v73, v86);
        v73 = v159;
      }
      else
      {
        if ( v73 )
        {
          *(_QWORD *)v73 = v87;
          v73 = v159;
        }
        v73 += 8;
        v159 = v73;
      }
    }
    ++v86;
  }
  while ( v86 != v85 );
  v88 = v158;
LABEL_113:
  v89 = (v73 - v88) >> 3;
  if ( (_DWORD)v89 )
  {
    v90 = 0;
    v135 = 8LL * (unsigned int)(v89 - 1);
    while ( 1 )
    {
      v93 = v155;
      v163 = *(_QWORD *)&v88[v90];
      v94 = sub_1C73C60(*(_QWORD **)(v155 + 32), *(_QWORD *)(v155 + 40), &v163);
      sub_1977D00(v93 + 32, v94);
      v91 = *(_QWORD **)(v93 + 64);
      if ( *(_QWORD **)(v93 + 72) == v91 )
      {
        v95 = &v91[*(unsigned int *)(v93 + 84)];
        if ( v91 == v95 )
        {
LABEL_129:
          v91 = v95;
        }
        else
        {
          while ( v163 != *v91 )
          {
            if ( v95 == ++v91 )
              goto LABEL_129;
          }
        }
      }
      else
      {
        v132 = v163;
        v91 = sub_16CC9F0(v93 + 56, v163);
        if ( v132 == *v91 )
        {
          v96 = *(_QWORD *)(v93 + 72);
          if ( v96 == *(_QWORD *)(v93 + 64) )
            v97 = *(unsigned int *)(v93 + 84);
          else
            v97 = *(unsigned int *)(v93 + 80);
          v95 = (_QWORD *)(v96 + 8 * v97);
        }
        else
        {
          v92 = *(_QWORD *)(v93 + 72);
          if ( v92 != *(_QWORD *)(v93 + 64) )
            goto LABEL_117;
          v91 = (_QWORD *)(v92 + 8LL * *(unsigned int *)(v93 + 84));
          v95 = v91;
        }
      }
      if ( v91 != v95 )
      {
        *v91 = -2;
        ++*(_DWORD *)(v93 + 88);
      }
LABEL_117:
      if ( v90 == v135 )
        break;
      v88 = v158;
      v90 += 8;
    }
  }
LABEL_131:
  v98 = sub_157EBA0(v144);
  v99 = *(_QWORD *)(v98 + 48);
  v100 = (_QWORD *)v98;
  v156 = v99;
  if ( v99 )
    sub_1623A60((__int64)&v156, v99, 2);
  sub_15F20C0(v100);
  v101 = v144;
  v102 = sub_157E9C0(v144);
  v164 = v101;
  v166 = v102;
  v165 = (unsigned __int64 *)(v101 + 40);
  v103 = v145;
  v163 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v162 = 257;
  v104 = sub_1648A60(56, 1u);
  v105 = v104;
  if ( v104 )
    sub_15F8320((__int64)v104, v103, 0);
  if ( v164 )
  {
    v106 = v165;
    sub_157E9D0(v164 + 40, (__int64)v105);
    v107 = v105[3];
    v108 = *v106;
    v105[4] = v106;
    v108 &= 0xFFFFFFFFFFFFFFF8LL;
    v105[3] = v108 | v107 & 7;
    *(_QWORD *)(v108 + 8) = v105 + 3;
    *v106 = *v106 & 7 | (unsigned __int64)(v105 + 3);
  }
  sub_164B780((__int64)v105, v161);
  v109 = v105 + 6;
  if ( v163 )
  {
    v157 = v163;
    sub_1623A60((__int64)&v157, v163, 2);
    v110 = v105[6];
    v109 = v105 + 6;
    if ( v110 )
    {
      sub_161E7C0((__int64)(v105 + 6), v110);
      v109 = v105 + 6;
    }
    v111 = (unsigned __int8 *)v157;
    v105[6] = v157;
    if ( v111 )
    {
      v136 = v109;
      sub_1623210((__int64)&v157, v111, (__int64)v109);
      v109 = v136;
    }
  }
  v161[0] = v156;
  if ( v156 )
  {
    v137 = v109;
    sub_1623A60((__int64)v161, v156, 2);
    v109 = v137;
    if ( v137 == v161 )
    {
      if ( v161[0] )
        sub_161E7C0((__int64)v137, v161[0]);
      goto LABEL_146;
    }
    v119 = v105[6];
    if ( !v119 )
    {
LABEL_168:
      v120 = (unsigned __int8 *)v161[0];
      v105[6] = v161[0];
      if ( v120 )
        sub_1623210((__int64)v161, v120, (__int64)v109);
      goto LABEL_146;
    }
LABEL_167:
    v138 = v109;
    sub_161E7C0((__int64)v109, v119);
    v109 = v138;
    goto LABEL_168;
  }
  if ( v109 != v161 )
  {
    v119 = v105[6];
    if ( v119 )
      goto LABEL_167;
  }
LABEL_146:
  v112 = *(_QWORD *)(v143 + 48);
  for ( k = v143 + 40; k != v112; v112 = *(_QWORD *)(v112 + 8) )
  {
    if ( !v112 )
      BUG();
    if ( *(_BYTE *)(v112 - 8) != 77 )
      break;
    v114 = *(_DWORD *)(v112 - 4) & 0xFFFFFFF;
    if ( v114 )
    {
      v115 = v112 - 24;
      v116 = 0;
      v117 = 24LL * *(unsigned int *)(v112 + 32) + 8;
      while ( 1 )
      {
        v118 = v112 - 24 - 24LL * v114;
        if ( (*(_BYTE *)(v112 - 1) & 0x40) != 0 )
          v118 = *(_QWORD *)(v112 - 32);
        if ( v144 == *(_QWORD *)(v118 + v117) )
          break;
        ++v116;
        v117 += 8;
        if ( v114 == v116 )
        {
          v116 = -1;
          break;
        }
      }
    }
    else
    {
      v116 = -1;
      v115 = v112 - 24;
    }
    sub_15F5350(v115, v116, 1);
  }
  sub_1C75B60(*(_QWORD *)(a1 + 184), *(_QWORD *)(a1 + 168), v141, v147, v139, v140, v129, (__int64)v130, v153);
  if ( v163 )
    sub_161E7C0((__int64)&v163, v163);
  if ( v156 )
    sub_161E7C0((__int64)&v156, v156);
  if ( v158 )
    j_j___libc_free_0(v158, v160 - v158);
  return v20;
}
