// Function: sub_2CB3CF0
// Address: 0x2cb3cf0
//
__int64 __fastcall sub_2CB3CF0(__int64 a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // r12
  __int64 *v6; // rbx
  unsigned int **v7; // r13
  unsigned __int8 *v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int8 *v10; // r15
  __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  __int64 v13; // r15
  __int64 v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // r8
  __int64 v19; // r14
  __int64 *v20; // r13
  int v21; // r12d
  _BYTE *v22; // rsi
  int v23; // ecx
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned int v26; // ebx
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rsi
  _QWORD *v30; // rax
  _BYTE **v31; // rcx
  __int64 v32; // r8
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rsi
  unsigned __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // r10
  unsigned int v40; // esi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r10
  _BYTE *v46; // r14
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rcx
  _BYTE **v50; // r14
  __int64 v51; // rdx
  unsigned int **v52; // rax
  _BYTE **v53; // r12
  _QWORD *v54; // r15
  _BYTE **v55; // r13
  __int64 v56; // r8
  unsigned int **v57; // r11
  __int64 v58; // r9
  _BYTE *v59; // rbx
  __int64 v60; // r12
  _QWORD *v61; // rbx
  _QWORD *v62; // r13
  _DWORD *v63; // rax
  __int64 v64; // r10
  __int64 v65; // rcx
  __int64 v66; // r10
  __int64 v67; // r10
  _QWORD *v68; // rcx
  __int64 v69; // rcx
  _QWORD *v70; // r10
  __int64 v71; // r13
  _BYTE *v72; // r14
  __int64 *v73; // r12
  _QWORD *v74; // rdx
  __int64 v75; // rbx
  bool v76; // r11
  unsigned __int64 v77; // r15
  unsigned __int64 *v78; // rcx
  _QWORD *v79; // r9
  _QWORD *v80; // rax
  __int64 v81; // rax
  _QWORD *v82; // r9
  __int64 v83; // rdi
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  unsigned __int64 v87; // rdi
  void *v88; // rsi
  unsigned __int64 v89; // rax
  char *v90; // r11
  size_t v91; // rdx
  _QWORD *v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rax
  _BYTE *v95; // r8
  __int64 v96; // rcx
  void *v97; // r9
  void *v98; // rax
  char v99; // di
  unsigned int **v100; // [rsp+0h] [rbp-1E0h]
  __int64 v101; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 *v102; // [rsp+8h] [rbp-1D8h]
  __int64 v103; // [rsp+10h] [rbp-1D0h]
  _QWORD *v104; // [rsp+10h] [rbp-1D0h]
  __int64 v105; // [rsp+10h] [rbp-1D0h]
  __int64 v106; // [rsp+10h] [rbp-1D0h]
  __int64 v107; // [rsp+18h] [rbp-1C8h]
  __int64 v108; // [rsp+18h] [rbp-1C8h]
  __int64 v109; // [rsp+18h] [rbp-1C8h]
  bool v110; // [rsp+18h] [rbp-1C8h]
  __int64 v111; // [rsp+18h] [rbp-1C8h]
  _QWORD *v112; // [rsp+18h] [rbp-1C8h]
  __int64 v113; // [rsp+20h] [rbp-1C0h]
  __int64 v114; // [rsp+20h] [rbp-1C0h]
  _BYTE *v115; // [rsp+20h] [rbp-1C0h]
  _BYTE *v116; // [rsp+20h] [rbp-1C0h]
  __int64 v117; // [rsp+20h] [rbp-1C0h]
  _QWORD *v118; // [rsp+28h] [rbp-1B8h]
  __int64 v119; // [rsp+28h] [rbp-1B8h]
  __int64 v120; // [rsp+28h] [rbp-1B8h]
  __int64 v121; // [rsp+28h] [rbp-1B8h]
  _QWORD *v122; // [rsp+28h] [rbp-1B8h]
  _BYTE *v123; // [rsp+28h] [rbp-1B8h]
  int v124; // [rsp+30h] [rbp-1B0h]
  _QWORD *v125; // [rsp+30h] [rbp-1B0h]
  __int64 v126; // [rsp+30h] [rbp-1B0h]
  __int64 v127; // [rsp+30h] [rbp-1B0h]
  __int64 v128; // [rsp+30h] [rbp-1B0h]
  __int64 v129; // [rsp+30h] [rbp-1B0h]
  _QWORD *v130; // [rsp+30h] [rbp-1B0h]
  _BYTE *v131; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v132; // [rsp+30h] [rbp-1B0h]
  char *v133; // [rsp+30h] [rbp-1B0h]
  char *v134; // [rsp+38h] [rbp-1A8h]
  __int64 v135; // [rsp+38h] [rbp-1A8h]
  __int64 v136; // [rsp+38h] [rbp-1A8h]
  __int64 v137; // [rsp+38h] [rbp-1A8h]
  _BYTE *v138; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v139; // [rsp+40h] [rbp-1A0h] BYREF
  unsigned int v140; // [rsp+48h] [rbp-198h]
  char v141; // [rsp+60h] [rbp-180h]
  char v142; // [rsp+61h] [rbp-17Fh]
  _BYTE *v143; // [rsp+70h] [rbp-170h] BYREF
  __int64 v144; // [rsp+78h] [rbp-168h]
  _BYTE v145[32]; // [rsp+80h] [rbp-160h] BYREF
  char *v146; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v147; // [rsp+A8h] [rbp-138h]
  _BYTE v148[32]; // [rsp+B0h] [rbp-130h] BYREF
  _QWORD *v149; // [rsp+D0h] [rbp-110h] BYREF
  unsigned __int64 v150; // [rsp+D8h] [rbp-108h]
  _BYTE **v151; // [rsp+E0h] [rbp-100h]
  __int64 v152; // [rsp+E8h] [rbp-F8h]
  __int64 v153; // [rsp+F0h] [rbp-F0h]
  void *src; // [rsp+F8h] [rbp-E8h]
  __int64 v155; // [rsp+100h] [rbp-E0h]
  __int64 v156; // [rsp+108h] [rbp-D8h]
  __int64 v157; // [rsp+110h] [rbp-D0h]
  char *v158; // [rsp+118h] [rbp-C8h]
  unsigned __int64 v159; // [rsp+120h] [rbp-C0h] BYREF
  unsigned int v160; // [rsp+128h] [rbp-B8h]
  char v161; // [rsp+130h] [rbp-B0h] BYREF
  _QWORD *v162; // [rsp+168h] [rbp-78h]
  void *v163; // [rsp+1A0h] [rbp-40h]

  v5 = a3;
  v6 = a2;
  v160 = sub_AE43F0(a3, *(_QWORD *)(a1 + 8));
  if ( v160 > 0x40 )
  {
    v7 = (unsigned int **)&v159;
    sub_C43690((__int64)&v159, 0, 0);
  }
  else
  {
    v159 = 0;
    v7 = (unsigned int **)&v159;
  }
  v8 = sub_BD45C0((unsigned __int8 *)a1, v5, (__int64)&v159, 1, 0, 0, 0, 0);
  v9 = v159;
  v10 = v8;
  if ( v160 > 0x40 )
  {
    *a2 = *(_QWORD *)v159;
    j_j___libc_free_0_0(v9);
    v12 = sub_BD3990(v10, v5);
    v13 = (__int64)v12;
    if ( *v12 != 63 )
      return v13;
  }
  else
  {
    v11 = 0;
    if ( v160 )
      v11 = (__int64)(v159 << (64 - (unsigned __int8)v160)) >> (64 - (unsigned __int8)v160);
    *a2 = v11;
    v12 = sub_BD3990(v8, v5);
    v13 = (__int64)v12;
    if ( *v12 != 63 )
      return v13;
  }
  if ( (unsigned __int8)sub_B4DD90((__int64)v12) )
    return v13;
  v124 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
  v150 = 8;
  v149 = (_QWORD *)sub_22077B0(0x40u);
  v134 = (char *)(v149 + 3);
  v15 = sub_22077B0(0x200u);
  v152 = v15;
  v149[3] = v15;
  src = v134;
  v153 = v15 + 512;
  v158 = v134;
  v156 = v15;
  v157 = v15 + 512;
  v151 = (_BYTE **)v15;
  v155 = v15;
  if ( !v124 )
  {
LABEL_16:
    sub_2CB3C70((unsigned __int64 *)&v149);
    return v13;
  }
  v16 = (unsigned int)(v124 - 1);
  while ( 1 )
  {
    v17 = *(_BYTE **)(v13 + 32 * (v16 - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
    if ( *v17 != 17 )
      break;
    if ( v151 == (_BYTE **)v152 )
    {
      v82 = src;
      v137 = v158 - (_BYTE *)src;
      v83 = (v158 - (_BYTE *)src) >> 3;
      if ( ((v153 - (__int64)v151) >> 3) + ((v155 - v156) >> 3) + ((v83 - 1) << 6) == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( src != v149 )
        goto LABEL_92;
      v86 = v83 + 2;
      v87 = 2 * (v83 + 2);
      if ( v150 <= v87 )
      {
        v93 = 1;
        if ( v150 )
          v93 = v150;
        v132 = v150 + v93 + 2;
        if ( v132 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v87, v93, v86);
        v105 = v16;
        v111 = v86;
        v115 = *(_BYTE **)(v13 + 32 * (v16 - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
        v94 = sub_22077B0(8 * v132);
        v95 = v115;
        v122 = (_QWORD *)v94;
        v96 = v105;
        v97 = (void *)(v94 + 8 * ((v132 - v111) >> 1) + 8);
        if ( v158 + 8 != src )
        {
          v98 = memmove(v97, src, v158 + 8 - (_BYTE *)src);
          v96 = v105;
          v95 = v115;
          v97 = v98;
        }
        v106 = v96;
        v112 = v97;
        v116 = v95;
        j_j___libc_free_0((unsigned __int64)v149);
        v17 = v116;
        v82 = v112;
        v16 = v106;
        v149 = v122;
        v150 += v93 + 2;
        goto LABEL_97;
      }
      v88 = v158 + 8;
      v89 = (v150 - v86) >> 1;
      v90 = (char *)src + 8 * v89 + 8;
      v91 = v158 + 8 - (_BYTE *)src;
      if ( src <= v90 )
      {
        if ( src != v88 )
        {
          v117 = v16;
          v123 = *(_BYTE **)(v13 + 32 * (v16 - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
          v133 = (char *)src + 8 * v89 + 8;
          memmove(v90, src, v91);
          v17 = v123;
          v16 = v117;
          v82 = v133;
          goto LABEL_97;
        }
      }
      else if ( src != v88 )
      {
        v121 = v16;
        v131 = *(_BYTE **)(v13 + 32 * (v16 - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
        v92 = memmove((char *)src + 8 * v89 + 8, src, v91);
        v17 = v131;
        v16 = v121;
        v82 = v92;
LABEL_97:
        src = v82;
        v152 = *v82;
        v153 = v152 + 512;
        v158 = (char *)v82 + v137;
        v156 = *(_QWORD *)((char *)v82 + v137);
        v157 = v156 + 512;
LABEL_92:
        v120 = v16;
        v130 = v82;
        v138 = v17;
        v84 = sub_22077B0(0x200u);
        v16 = v120;
        *(v130 - 1) = v84;
        src = (char *)src - 8;
        v85 = *(_QWORD *)src + 512LL;
        v152 = *(_QWORD *)src;
        v153 = v85;
        v151 = (_BYTE **)(v152 + 504);
        *(_QWORD *)(v152 + 504) = v138;
        goto LABEL_15;
      }
      v82 = (char *)src + 8 * v89 + 8;
      goto LABEL_97;
    }
    *--v151 = v17;
LABEL_15:
    if ( v16-- == 0 )
      goto LABEL_16;
  }
  v143 = v145;
  v144 = 0x400000000LL;
  if ( (_DWORD)v16 )
  {
    v125 = a4;
    v19 = v5;
    v20 = v6;
    v21 = v16;
    v22 = v145;
    v23 = 1;
    v24 = *(_QWORD *)(v13 + 32 * (1LL - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
    v25 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v22[8 * v25] = v24;
      v26 = v23 + 1;
      v25 = (unsigned int)(v144 + 1);
      LODWORD(v144) = v144 + 1;
      if ( v23 == v21 )
        break;
      v24 = *(_QWORD *)(v13 + 32 * (v26 - (unsigned __int64)(*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
      if ( v25 + 1 > (unsigned __int64)HIDWORD(v144) )
      {
        v107 = *(_QWORD *)(v13 + 32 * (v26 - (unsigned __int64)(*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
        sub_C8D5F0((__int64)&v143, v145, v25 + 1, 8u, (__int64)v17, v24);
        v25 = (unsigned int)v144;
        v24 = v107;
      }
      v22 = v143;
      v23 = v26;
    }
    v6 = v20;
    v5 = v19;
    v7 = (unsigned int **)&v159;
    a4 = v125;
  }
  sub_23D0AB0((__int64)&v159, v13, 0, 0, 0);
  v27 = (_QWORD *)a4[2];
  v28 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
  v118 = a4 + 1;
  if ( !v27 )
  {
LABEL_32:
    v31 = (_BYTE **)v143;
    v32 = (unsigned int)v144;
    goto LABEL_33;
  }
  v29 = a4 + 1;
  while ( 2 )
  {
    if ( v28 > v27[4] )
    {
      v27 = (_QWORD *)v27[3];
      goto LABEL_28;
    }
    v30 = (_QWORD *)v27[2];
    if ( v28 < v27[4] )
    {
      v29 = v27;
      v27 = (_QWORD *)v27[2];
LABEL_28:
      if ( !v27 )
        goto LABEL_32;
      continue;
    }
    break;
  }
  v68 = (_QWORD *)v27[3];
  while ( v68 )
  {
    if ( v68[4] <= v28 )
    {
      v68 = (_QWORD *)v68[3];
    }
    else
    {
      v29 = v68;
      v68 = (_QWORD *)v68[2];
    }
  }
  while ( v30 )
  {
    while ( 1 )
    {
      v69 = v30[3];
      if ( v30[4] >= v28 )
        break;
      v30 = (_QWORD *)v30[3];
      if ( !v69 )
        goto LABEL_77;
    }
    v27 = v30;
    v30 = (_QWORD *)v30[2];
  }
LABEL_77:
  v31 = (_BYTE **)v143;
  v32 = (unsigned int)v144;
  if ( v29 != v27 )
  {
    v70 = a4;
    v71 = v5;
    v72 = &v143[8 * (unsigned int)v144];
    v73 = v6;
    v74 = v70;
    v75 = v13;
    v76 = v72 == v143;
    v77 = (unsigned __int64)v143;
    v78 = &v159;
    do
    {
      v39 = v27[5];
      v79 = (_QWORD *)v77;
      v80 = (_QWORD *)(v39 + 32 * (1LL - (*(_DWORD *)(v39 + 4) & 0x7FFFFFF)));
      if ( (_QWORD *)v39 != v80 && !v76 )
      {
        do
        {
          if ( *v80 != *v79 )
            break;
          v80 += 4;
          ++v79;
          if ( (_QWORD *)v39 == v80 )
            break;
        }
        while ( v72 != (_BYTE *)v79 );
      }
      if ( v80 == (_QWORD *)v39 && v72 == (_BYTE *)v79 )
      {
        v6 = v73;
        v5 = v71;
        v7 = (unsigned int **)v78;
        goto LABEL_42;
      }
      v102 = v78;
      v104 = v74;
      v110 = v76;
      v114 = v32;
      v81 = sub_220EEE0((__int64)v27);
      v32 = v114;
      v76 = v110;
      v74 = v104;
      v27 = (_QWORD *)v81;
      v78 = v102;
    }
    while ( (_QWORD *)v81 != v29 );
    a4 = v104;
    v31 = (_BYTE **)v77;
    v13 = v75;
    v6 = v73;
    v5 = v71;
    v7 = (unsigned int **)v102;
  }
LABEL_33:
  v148[17] = 1;
  v146 = "splitGEPI.base";
  v148[16] = 3;
  v113 = sub_921130(
           v7,
           *(_QWORD *)(v13 + 72),
           *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)),
           v31,
           v32,
           (__int64)&v146,
           0);
  v33 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
  v34 = sub_22077B0(0x30u);
  v35 = (_QWORD *)a4[2];
  *(_QWORD *)(v34 + 32) = v33;
  v36 = v34;
  *(_QWORD *)(v34 + 40) = v113;
  if ( v35 )
  {
    while ( 1 )
    {
      v37 = v35[4];
      v38 = (_QWORD *)v35[3];
      if ( v33 < v37 )
        v38 = (_QWORD *)v35[2];
      if ( !v38 )
        break;
      v35 = v38;
    }
    v99 = 1;
    if ( v35 != v118 )
      v99 = v33 < v37;
  }
  else
  {
    v35 = v118;
    v99 = 1;
  }
  sub_220F040(v99, v36, v35, v118);
  ++a4[5];
  v39 = v113;
LABEL_42:
  v108 = v39;
  v119 = *(_QWORD *)(v39 + 8);
  v40 = sub_AE2980(v5, *(_DWORD *)(v119 + 8) >> 8)[1];
  v146 = v148;
  v147 = 0x400000000LL;
  v41 = sub_BCCE00(v162, v40);
  v42 = sub_ACD640(v41, 0, 0);
  v45 = v108;
  v46 = (_BYTE *)v42;
  v47 = (unsigned int)v147;
  v48 = (unsigned int)v147 + 1LL;
  if ( v48 > HIDWORD(v147) )
  {
    sub_C8D5F0((__int64)&v146, v148, v48, 8u, v43, v44);
    v47 = (unsigned int)v147;
    v45 = v108;
  }
  v49 = v5;
  *(_QWORD *)&v146[8 * v47] = v46;
  v50 = v151;
  LODWORD(v147) = v147 + 1;
  v51 = (unsigned int)v147;
  v52 = v7;
  v53 = (_BYTE **)v153;
  v54 = src;
  v55 = (_BYTE **)v155;
  v56 = (__int64)v6;
  v57 = v52;
  v58 = v49;
  while ( v55 != v50 )
  {
    v59 = *v50;
    if ( v51 + 1 > (unsigned __int64)HIDWORD(v147) )
    {
      v100 = v57;
      v101 = v58;
      v103 = v56;
      v109 = v45;
      sub_C8D5F0((__int64)&v146, v148, v51 + 1, 8u, v56, v58);
      v51 = (unsigned int)v147;
      v57 = v100;
      v58 = v101;
      v56 = v103;
      v45 = v109;
    }
    ++v50;
    *(_QWORD *)&v146[8 * v51] = v59;
    v51 = (unsigned int)(v147 + 1);
    LODWORD(v147) = v147 + 1;
    if ( v53 == v50 )
    {
      v50 = (_BYTE **)v54[1];
      ++v54;
      v53 = v50 + 64;
    }
  }
  v60 = v58;
  v142 = 1;
  v139 = (unsigned __int64)"splitGEPI.replace";
  v61 = (_QWORD *)v56;
  v141 = 3;
  v126 = v45;
  v62 = (_QWORD *)sub_921130(v57, *(_QWORD *)(v45 + 80), v45, (_BYTE **)v146, v51, (__int64)&v139, 0);
  v63 = sub_AE2980(v60, *(_DWORD *)(v119 + 8) >> 8);
  v64 = v126;
  v140 = v63[1];
  if ( v140 > 0x40 )
  {
    sub_C43690((__int64)&v139, 0, 1);
    v64 = v126;
  }
  else
  {
    v139 = 0;
  }
  v127 = v64;
  sub_B4DE60((__int64)v62, v60, (__int64)&v139);
  if ( v140 > 0x40 )
  {
    v65 = *(_QWORD *)v139;
  }
  else
  {
    v65 = 0;
    if ( v140 )
      v65 = (__int64)(v139 << (64 - (unsigned __int8)v140)) >> (64 - (unsigned __int8)v140);
  }
  *v61 += v65;
  sub_B43D60(v62);
  v66 = v127;
  if ( v140 > 0x40 && v139 )
  {
    j_j___libc_free_0_0(v139);
    v66 = v127;
  }
  if ( v146 != v148 )
  {
    v128 = v66;
    _libc_free((unsigned __int64)v146);
    v66 = v128;
  }
  v129 = v66;
  nullsub_61();
  v163 = &unk_49DA100;
  nullsub_63();
  v67 = v129;
  if ( (char *)v159 != &v161 )
  {
    _libc_free(v159);
    v67 = v129;
  }
  if ( v143 != v145 )
  {
    v135 = v67;
    _libc_free((unsigned __int64)v143);
    v67 = v135;
  }
  v136 = v67;
  sub_2CB3C70((unsigned __int64 *)&v149);
  return v136;
}
