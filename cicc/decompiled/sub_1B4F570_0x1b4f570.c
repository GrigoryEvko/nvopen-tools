// Function: sub_1B4F570
// Address: 0x1b4f570
//
__int64 __fastcall sub_1B4F570(
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
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // r15
  __int64 v19; // rbx
  char v20; // cl
  __int64 v21; // r14
  __int64 v22; // r12
  int v23; // edx
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int64 *v26; // r13
  unsigned __int64 *v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // r10
  __int64 v35; // rcx
  char v36; // al
  char v37; // di
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned int v42; // r13d
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdx
  __int64 v48; // rax
  char v49; // al
  __int64 v50; // rsi
  __int64 v51; // rax
  _QWORD *v52; // r13
  __int64 v53; // rdx
  __int64 v54; // rdi
  _QWORD *v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rax
  _QWORD *v58; // rax
  _QWORD *v59; // r14
  unsigned __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // rbx
  __int64 v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rbx
  __int64 v72; // rax
  char v73; // di
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rsi
  __int64 v78; // r14
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r12
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // rax
  _QWORD *v85; // r13
  double v86; // xmm4_8
  double v87; // xmm5_8
  __int64 v88; // rcx
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rbx
  unsigned __int64 v94; // r12
  unsigned __int64 v95; // r11
  unsigned __int64 *v96; // rax
  unsigned __int64 *v97; // r10
  unsigned __int64 *v98; // rax
  unsigned __int64 *v99; // rax
  __int64 v100; // rdx
  unsigned __int64 v101; // r11
  _BOOL8 v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rcx
  __int64 v107; // rcx
  unsigned __int64 v108; // rcx
  unsigned __int64 *v109; // rax
  unsigned __int64 v110; // r8
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // rdi
  unsigned int v113; // r12d
  int v114; // r13d
  __int64 v115; // r14
  unsigned int v116; // esi
  __int64 v117; // rax
  unsigned __int64 *v118; // rdi
  __int64 v119; // rsi
  _QWORD *v120; // rax
  double v121; // xmm4_8
  double v122; // xmm5_8
  __int64 v123; // [rsp+8h] [rbp-168h]
  __int64 v124; // [rsp+10h] [rbp-160h]
  int v125; // [rsp+10h] [rbp-160h]
  unsigned __int8 v126; // [rsp+1Fh] [rbp-151h]
  __int64 v127; // [rsp+20h] [rbp-150h]
  __int64 v128; // [rsp+20h] [rbp-150h]
  int v129; // [rsp+28h] [rbp-148h]
  unsigned __int64 *v130; // [rsp+28h] [rbp-148h]
  __int64 v132; // [rsp+38h] [rbp-138h]
  unsigned __int64 v133; // [rsp+38h] [rbp-138h]
  unsigned __int64 *v134; // [rsp+38h] [rbp-138h]
  __int64 v135; // [rsp+40h] [rbp-130h]
  unsigned __int8 v136; // [rsp+48h] [rbp-128h]
  unsigned __int64 v137; // [rsp+48h] [rbp-128h]
  unsigned __int64 *v138; // [rsp+48h] [rbp-128h]
  unsigned __int64 v139; // [rsp+48h] [rbp-128h]
  unsigned __int64 *v140; // [rsp+48h] [rbp-128h]
  __int64 v141; // [rsp+50h] [rbp-120h]
  unsigned __int8 v142; // [rsp+58h] [rbp-118h]
  __int64 v143; // [rsp+58h] [rbp-118h]
  __int64 v144; // [rsp+58h] [rbp-118h]
  unsigned __int64 v145; // [rsp+58h] [rbp-118h]
  unsigned int v146; // [rsp+58h] [rbp-118h]
  unsigned int v148; // [rsp+60h] [rbp-110h]
  __int64 v149; // [rsp+60h] [rbp-110h]
  __int64 v150; // [rsp+68h] [rbp-108h]
  __int64 v151; // [rsp+68h] [rbp-108h]
  __int64 v152; // [rsp+68h] [rbp-108h]
  __int64 v153; // [rsp+78h] [rbp-F8h] BYREF
  const char *v154; // [rsp+80h] [rbp-F0h] BYREF
  char v155; // [rsp+90h] [rbp-E0h]
  char v156; // [rsp+91h] [rbp-DFh]
  __int64 v157; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v158; // [rsp+A8h] [rbp-C8h]
  int v159; // [rsp+B8h] [rbp-B8h]
  __int64 v160; // [rsp+C0h] [rbp-B0h] BYREF
  int v161; // [rsp+C8h] [rbp-A8h] BYREF
  unsigned __int64 *v162; // [rsp+D0h] [rbp-A0h]
  int *v163; // [rsp+D8h] [rbp-98h]
  int *v164; // [rsp+E0h] [rbp-90h]
  __int64 v165; // [rsp+E8h] [rbp-88h]
  __int64 v166[16]; // [rsp+F0h] [rbp-80h] BYREF

  v10 = *(_QWORD *)(a1 - 24);
  v11 = *(_QWORD *)(a1 - 48);
  v12 = 0;
  v13 = *(_QWORD *)(v10 + 48);
  v141 = v10;
  v14 = *(_QWORD *)(v11 + 48);
  v132 = v11;
  v15 = *(_BYTE *)(v13 - 8);
  v16 = *(_QWORD *)(v13 + 8);
  v17 = v13 - 24;
  v18 = *(_QWORD *)(v14 + 8);
  v19 = v14 - 24;
  if ( v15 == 78 )
  {
    v64 = *(_QWORD *)(v13 - 48);
    if ( !*(_BYTE *)(v64 + 16) && (*(_BYTE *)(v64 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v64 + 36) - 35) < 4 )
      v12 = v13 - 24;
  }
  v20 = *(_BYTE *)(v14 - 8);
  if ( v20 == 78 )
  {
    v63 = *(_QWORD *)(v14 - 48);
    v151 = v14;
    if ( !*(_BYTE *)(v63 + 16)
      && (*(_BYTE *)(v63 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v63 + 36) - 35) <= 3
      && v12 )
    {
      if ( sub_15F40E0(v12, v14 - 24) )
        goto LABEL_4;
      v20 = *(_BYTE *)(v151 - 8);
      if ( *(_BYTE *)(v13 - 8) != 78 )
      {
        if ( v20 == 78 )
          goto LABEL_38;
LABEL_4:
        if ( *(_BYTE *)(v17 + 16) != 77 )
          goto LABEL_5;
        return 0;
      }
    }
    else
    {
      if ( v15 != 78 )
        goto LABEL_38;
      v20 = 78;
    }
  }
  else if ( v15 != 78 )
  {
    goto LABEL_4;
  }
  do
  {
    v38 = *(_QWORD *)(v17 - 24);
    v39 = v16;
    if ( *(_BYTE *)(v38 + 16) )
      break;
    if ( (*(_BYTE *)(v38 + 33) & 0x20) == 0 )
      break;
    if ( (unsigned int)(*(_DWORD *)(v38 + 36) - 35) > 3 )
      break;
    v16 = *(_QWORD *)(v16 + 8);
    v17 = v39 - 24;
  }
  while ( *(_BYTE *)(v39 - 8) == 78 );
  if ( v20 != 78 )
    goto LABEL_4;
  do
  {
LABEL_38:
    v40 = *(_QWORD *)(v19 - 24);
    v41 = v18;
    if ( *(_BYTE *)(v40 + 16) || (*(_BYTE *)(v40 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v40 + 36) - 35) > 3 )
      goto LABEL_4;
    v18 = *(_QWORD *)(v18 + 8);
    v19 = v41 - 24;
  }
  while ( *(_BYTE *)(v41 - 8) == 78 );
  if ( *(_BYTE *)(v17 + 16) == 77 )
    return 0;
LABEL_5:
  v126 = sub_15F40E0(v17, v19);
  if ( !v126
    || *(_BYTE *)(v17 + 16) == 29 && !(unsigned __int8)sub_1B43FA0(v10, v132, v17, v19)
    || (unsigned __int8)sub_1B42630(v17) )
  {
    return 0;
  }
  v142 = 0;
  v150 = v16;
  v21 = v17;
  v22 = v19;
  v135 = *(_QWORD *)(a1 + 40);
  while ( 1 )
  {
    v23 = *(unsigned __int8 *)(v21 + 16);
    if ( (unsigned int)(v23 - 25) <= 9 )
      break;
    if ( (unsigned __int8)sub_1B42630(v21)
      || *(_BYTE *)(v21 + 16) == 78
      && *(_BYTE *)(v22 + 16) == 78
      && ((*(_WORD *)(v22 + 18) & 3) == 2) != ((*(_WORD *)(v21 + 18) & 3) == 2) )
    {
      return v142;
    }
    if ( !(unsigned __int8)sub_14A2D20(a2) )
      return v142;
    v136 = sub_14A2D20(a2);
    if ( !v136 )
      return v142;
    if ( *(_BYTE *)(v21 + 16) == 78
      && (v51 = *(_QWORD *)(v21 - 24), !*(_BYTE *)(v51 + 16))
      && (*(_BYTE *)(v51 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v51 + 36) - 35) <= 3
      || *(_BYTE *)(v22 + 16) == 78
      && (v62 = *(_QWORD *)(v22 - 24), !*(_BYTE *)(v62 + 16))
      && (*(_BYTE *)(v62 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v62 + 36) - 35) <= 3 )
    {
      v52 = (_QWORD *)(a1 + 24);
      v53 = v21 + 24;
      v54 = v135 + 40;
      v55 = *(_QWORD **)(v21 + 32);
      if ( a1 + 24 != v21 + 24 && v52 != v55 )
      {
        if ( v54 != v141 + 40 )
        {
          v127 = *(_QWORD *)(v21 + 32);
          sub_157EA80(v54, v141 + 40, v53, v127);
          v55 = (_QWORD *)v127;
          v53 = v21 + 24;
          v54 = v135 + 40;
        }
        if ( v52 != v55 && (_QWORD *)v53 != v55 )
        {
          v56 = *v55 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*(_QWORD *)(v21 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v55;
          *v55 = *v55 & 7LL | *(_QWORD *)(v21 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          v57 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(v56 + 8) = v52;
          v57 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v21 + 24) = v57 | *(_QWORD *)(v21 + 24) & 7LL;
          *(_QWORD *)(v57 + 8) = v53;
          *(_QWORD *)(a1 + 24) = v56 | *(_QWORD *)(a1 + 24) & 7LL;
        }
      }
      v58 = *(_QWORD **)(v22 + 32);
      v59 = (_QWORD *)(v22 + 24);
      if ( v52 != v58 && v52 != v59 )
      {
        if ( v54 != v132 + 40 )
        {
          v144 = *(_QWORD *)(v22 + 32);
          sub_157EA80(v54, v132 + 40, v22 + 24, v144);
          v58 = (_QWORD *)v144;
        }
        if ( v52 != v58 && v59 != v58 )
        {
          v60 = *v58 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*(_QWORD *)(v22 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v58;
          *v58 = *v58 & 7LL | *(_QWORD *)(v22 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          v61 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(v60 + 8) = v52;
          v61 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v22 + 24) = v61 | *(_QWORD *)(v22 + 24) & 7LL;
          *(_QWORD *)(v61 + 8) = v59;
          *(_QWORD *)(a1 + 24) = v60 | *(_QWORD *)(a1 + 24) & 7LL;
        }
      }
    }
    else
    {
      v26 = *(unsigned __int64 **)(v21 + 32);
      v27 = (unsigned __int64 *)(a1 + 24);
      v28 = v21 + 24;
      if ( (unsigned __int64 *)(a1 + 24) != v26 && v27 != (unsigned __int64 *)v28 )
      {
        if ( v135 + 40 != v141 + 40 )
        {
          sub_157EA80(v135 + 40, v141 + 40, v28, *(_QWORD *)(v21 + 32));
          v27 = (unsigned __int64 *)(a1 + 24);
          v28 = v21 + 24;
        }
        if ( v27 != v26 && (unsigned __int64 *)v28 != v26 )
        {
          v29 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*(_QWORD *)(v21 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v26;
          *v26 = *v26 & 7 | *(_QWORD *)(v21 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          v30 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(v29 + 8) = v27;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v21 + 24) = v30 | *(_QWORD *)(v21 + 24) & 7LL;
          *(_QWORD *)(v30 + 8) = v28;
          *(_QWORD *)(a1 + 24) = v29 | *(_QWORD *)(a1 + 24) & 7LL;
        }
      }
      if ( *(_QWORD *)(v22 + 8) )
        sub_164D160(v22, v21, a3, a4, a5, a6, v24, v25, a9, a10);
      sub_15F2780((unsigned __int8 *)v21, v22);
      v166[0] = 0x400000001LL;
      v166[1] = 0x600000003LL;
      v166[2] = 0x100000000BLL;
      v166[3] = 0xC00000011LL;
      v166[4] = 0xA0000000DLL;
      sub_1AEC0C0(v21, v22, (unsigned int *)v166, 10);
      v31 = sub_15C70A0(v22 + 48);
      v32 = sub_15C70A0(v21 + 48);
      sub_15AC0B0(v21, v32, v31);
      sub_15F20C0((_QWORD *)v22);
    }
    v33 = *(_QWORD *)(v18 + 8);
    v22 = v18 - 24;
    v34 = 0;
    v35 = *(_QWORD *)(v150 + 8);
    v21 = v150 - 24;
    v36 = *(_BYTE *)(v150 - 8);
    if ( v36 == 78 )
    {
      v50 = *(_QWORD *)(v150 - 48);
      if ( !*(_BYTE *)(v50 + 16) && (*(_BYTE *)(v50 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v50 + 36) - 35) < 4 )
        v34 = v150 - 24;
    }
    v37 = *(_BYTE *)(v18 - 8);
    if ( v37 != 78 )
    {
      if ( v36 != 78 )
        goto LABEL_31;
      goto LABEL_50;
    }
    v44 = *(_QWORD *)(v18 - 48);
    if ( !*(_BYTE *)(v44 + 16)
      && (*(_BYTE *)(v44 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v44 + 36) - 35) <= 3
      && v34 )
    {
      v143 = *(_QWORD *)(v150 + 8);
      v49 = sub_15F40E0(v34, v18 - 24);
      v35 = v143;
      if ( v49 )
        goto LABEL_31;
      v37 = *(_BYTE *)(v18 - 8);
      if ( *(_BYTE *)(v150 - 8) == 78 )
      {
        do
        {
LABEL_50:
          v45 = *(_QWORD *)(v21 - 24);
          v46 = v35;
          if ( *(_BYTE *)(v45 + 16) )
            break;
          if ( (*(_BYTE *)(v45 + 33) & 0x20) == 0 )
            break;
          if ( (unsigned int)(*(_DWORD *)(v45 + 36) - 35) > 3 )
            break;
          v35 = *(_QWORD *)(v35 + 8);
          v21 = v46 - 24;
        }
        while ( *(_BYTE *)(v46 - 8) == 78 );
      }
      if ( v37 != 78 )
        goto LABEL_31;
      goto LABEL_55;
    }
    if ( v36 == 78 )
    {
      v37 = 78;
      goto LABEL_50;
    }
    do
    {
LABEL_55:
      v47 = *(_QWORD *)(v22 - 24);
      v48 = v33;
      if ( *(_BYTE *)(v47 + 16) )
        break;
      if ( (*(_BYTE *)(v47 + 33) & 0x20) == 0 )
        break;
      if ( (unsigned int)(*(_DWORD *)(v47 + 36) - 35) > 3 )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      v22 = v48 - 24;
    }
    while ( *(_BYTE *)(v48 - 8) == 78 );
LABEL_31:
    v150 = v35;
    v142 = sub_15F40E0(v21, v22);
    if ( !v142 )
      return v136;
    v18 = v33;
  }
  v66 = v22;
  v42 = v142;
  v67 = v21;
  if ( (_BYTE)v23 == 29 && !(unsigned __int8)sub_1B43FA0(v141, v132, v21, v66) )
    return v42;
  v68 = sub_157EBA0(v141);
  if ( !v68 || (v129 = sub_15F4D60(v68), v145 = sub_157EBA0(v141), !v129) )
  {
LABEL_140:
    v85 = (_QWORD *)sub_15F4880(v67);
    sub_157E9D0(v135 + 40, (__int64)v85);
    v88 = *(_QWORD *)(a1 + 24);
    v85[4] = a1 + 24;
    v88 &= 0xFFFFFFFFFFFFFFF8LL;
    v85[3] = v88 | v85[3] & 7LL;
    *(_QWORD *)(v88 + 8) = v85 + 3;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a1 + 24) & 7LL | (unsigned __int64)(v85 + 3);
    if ( *(_BYTE *)(*v85 + 8LL) )
    {
      sub_164D160(v67, (__int64)v85, a3, a4, a5, a6, v86, v87, a9, a10);
      sub_164D160(v66, (__int64)v85, a3, a4, a5, a6, v121, v122, a9, a10);
      sub_164B7C0((__int64)v85, v67);
    }
    sub_1B47690((__int64)v166, (__int64)v85, 0, 0, 0);
    v161 = 0;
    v162 = 0;
    v163 = &v161;
    v164 = &v161;
    v165 = 0;
    sub_1B46D30((__int64)&v157, v141);
    v123 = v157;
    v125 = v159;
    v146 = v158;
    if ( v158 == v159 )
    {
LABEL_188:
      sub_1B46D30((__int64)&v157, v141);
      v113 = v158;
      v114 = v159;
      v115 = v157;
      while ( v114 != v113 )
      {
        v116 = v113++;
        v117 = sub_15F4DF0(v115, v116);
        sub_1B44430(v117, v135, v141);
      }
      sub_1B44FE0(a1);
      sub_1B43DD0((__int64)v162);
      sub_17CD270(v166);
      return v126;
    }
    v89 = v132;
    while ( 1 )
    {
      v90 = sub_15F4DF0(v123, v146);
      v91 = sub_157F280(v90);
      v149 = v92;
      v93 = v91;
      v153 = v91;
      if ( v91 != v92 )
        break;
LABEL_187:
      if ( v125 == ++v146 )
        goto LABEL_188;
    }
    while ( 1 )
    {
      v94 = sub_1455EB0(v93, v141);
      v95 = sub_1455EB0(v93, v89);
      if ( v94 != v95 )
      {
        v96 = v162;
        v97 = (unsigned __int64 *)&v161;
        if ( !v162 )
          goto LABEL_165;
        do
        {
          if ( v94 > v96[4] || v94 == v96[4] && v95 > v96[5] )
          {
            v96 = (unsigned __int64 *)v96[3];
          }
          else
          {
            v97 = v96;
            v96 = (unsigned __int64 *)v96[2];
          }
        }
        while ( v96 );
        if ( v97 == (unsigned __int64 *)&v161 || v94 < v97[4] || v94 == v97[4] && v95 < v97[5] )
        {
LABEL_165:
          v130 = v97;
          v137 = v95;
          v98 = (unsigned __int64 *)sub_22077B0(56);
          v98[4] = v94;
          v98[5] = v137;
          v98[6] = 0;
          v133 = v137;
          v138 = v98;
          v99 = sub_1B4F420(&v160, v130, v98 + 4);
          v101 = v133;
          if ( v100 )
          {
            v102 = 1;
            if ( &v161 != (int *)v100 && !v99 && v94 >= *(_QWORD *)(v100 + 32) )
            {
              v102 = 0;
              if ( v94 == *(_QWORD *)(v100 + 32) )
                v102 = v133 < *(_QWORD *)(v100 + 40);
            }
            sub_220F040(v102, v138, v100, &v161);
            ++v165;
            v97 = v138;
            v95 = v133;
          }
          else
          {
            v118 = v138;
            v134 = v99;
            v139 = v101;
            j_j___libc_free_0(v118, 56);
            v95 = v139;
            v97 = v134;
          }
        }
        if ( !v97[6] )
        {
          v140 = v97;
          v119 = *(_QWORD *)(a1 - 72);
          v156 = 1;
          v154 = "hte";
          v155 = 3;
          v120 = sub_1B47760(v166, v119, v94, v95, (__int64 *)&v154, a1);
          v97 = v140;
          v140[6] = (unsigned __int64)v120;
        }
        if ( (*(_DWORD *)(v93 + 20) & 0xFFFFFFF) != 0 )
          break;
      }
LABEL_146:
      sub_1B42F80((__int64)&v153);
      v93 = v153;
      if ( v153 == v149 )
        goto LABEL_187;
    }
    v103 = 0;
    v104 = 8LL * (*(_DWORD *)(v93 + 20) & 0xFFFFFFF);
    while ( 1 )
    {
      v107 = v103 + 24LL * *(unsigned int *)(v93 + 56) + 8;
      if ( (*(_BYTE *)(v93 + 23) & 0x40) != 0 )
      {
        v105 = *(_QWORD *)(v93 - 8);
        v106 = *(_QWORD *)(v105 + v107);
        if ( v141 == v106 )
          goto LABEL_178;
      }
      else
      {
        v105 = v93 - 24LL * (*(_DWORD *)(v93 + 20) & 0xFFFFFFF);
        v106 = *(_QWORD *)(v105 + v107);
        if ( v141 == v106 )
        {
LABEL_178:
          v108 = v97[6];
          v109 = (unsigned __int64 *)(3 * v103 + v105);
          if ( *v109 )
          {
            v110 = v109[1];
            v111 = v109[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v111 = v110;
            if ( v110 )
              *(_QWORD *)(v110 + 16) = *(_QWORD *)(v110 + 16) & 3LL | v111;
          }
          *v109 = v108;
          if ( v108 )
          {
            v112 = *(_QWORD *)(v108 + 8);
            v109[1] = v112;
            if ( v112 )
              *(_QWORD *)(v112 + 16) = (unsigned __int64)(v109 + 1) | *(_QWORD *)(v112 + 16) & 3LL;
            v109[2] = (v108 + 8) | v109[2] & 3;
            *(_QWORD *)(v108 + 8) = v109;
          }
          goto LABEL_175;
        }
      }
      if ( v89 == v106 )
        goto LABEL_178;
LABEL_175:
      v103 += 8;
      if ( v104 == v103 )
        goto LABEL_146;
    }
  }
  v128 = v66;
  v148 = 0;
  v124 = v21;
  v136 = v42;
  while ( 1 )
  {
    v69 = sub_15F4DF0(v145, v148);
    v70 = sub_157F280(v69);
    v152 = v65;
    v71 = v70;
    if ( v70 != v65 )
      break;
LABEL_138:
    if ( v129 == ++v148 )
    {
      v66 = v128;
      v67 = v124;
      goto LABEL_140;
    }
  }
  while ( 1 )
  {
    v72 = 0x17FFFFFFE8LL;
    v73 = *(_BYTE *)(v71 + 23) & 0x40;
    v74 = *(_DWORD *)(v71 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(v71 + 20) & 0xFFFFFFF) != 0 )
    {
      v65 = 24LL * *(unsigned int *)(v71 + 56) + 8;
      v75 = 0;
      do
      {
        v76 = v71 - 24LL * (unsigned int)v74;
        if ( v73 )
          v76 = *(_QWORD *)(v71 - 8);
        if ( v141 == *(_QWORD *)(v76 + v65) )
        {
          v72 = 24 * v75;
          goto LABEL_122;
        }
        ++v75;
        v65 += 8;
      }
      while ( (_DWORD)v74 != (_DWORD)v75 );
      v72 = 0x17FFFFFFE8LL;
    }
LABEL_122:
    if ( v73 )
    {
      v77 = *(_QWORD *)(v71 - 8);
    }
    else
    {
      v65 = 24LL * (unsigned int)v74;
      v77 = v71 - v65;
    }
    v78 = *(_QWORD *)(v77 + v72);
    v79 = 0x17FFFFFFE8LL;
    if ( (_DWORD)v74 )
    {
      v80 = 0;
      v65 = v77 + 24LL * *(unsigned int *)(v71 + 56);
      do
      {
        if ( v132 == *(_QWORD *)(v65 + 8 * v80 + 8) )
        {
          v79 = 24 * v80;
          goto LABEL_129;
        }
        ++v80;
      }
      while ( (_DWORD)v74 != (_DWORD)v80 );
      v79 = 0x17FFFFFFE8LL;
    }
LABEL_129:
    v81 = *(_QWORD *)(v77 + v79);
    if ( v78 != v81
      && (sub_1B43710(v78, v71, v65, v74)
       || sub_1B43710(v81, v71, v82, v83)
       || *(_BYTE *)(v78 + 16) == 5 && !(unsigned __int8)sub_14AF470(v78, 0, 0, 0)
       || *(_BYTE *)(v81 + 16) == 5 && !(unsigned __int8)sub_14AF470(v81, 0, 0, 0)) )
    {
      return v136;
    }
    v84 = *(_QWORD *)(v71 + 32);
    if ( !v84 )
      BUG();
    v71 = 0;
    if ( *(_BYTE *)(v84 - 8) == 77 )
      v71 = v84 - 24;
    if ( v152 == v71 )
      goto LABEL_138;
  }
}
