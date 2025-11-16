// Function: sub_F894B0
// Address: 0xf894b0
//
__int64 __fastcall sub_F894B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  bool v3; // zf
  char v4; // al
  __int64 v5; // r12
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r13d
  unsigned int v10; // eax
  _QWORD *v11; // r8
  __int64 v12; // rcx
  _QWORD *v13; // rsi
  __int64 v14; // rdi
  unsigned __int16 v15; // dx
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // ecx
  __int64 v22; // r8
  int v23; // ecx
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r10
  __int64 v27; // r15
  char v28; // r13
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  int v31; // edx
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int16 v34; // dx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // r13
  __int64 *v40; // rbx
  __int64 v41; // r12
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int v45; // r15d
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  int v48; // esi
  __int64 v49; // rdi
  unsigned int v50; // ecx
  __int64 v51; // r8
  __int64 v52; // r15
  int v53; // ecx
  __int64 v54; // rsi
  bool v55; // dl
  int v56; // edi
  int v57; // edi
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 v60; // r8
  char v61; // cl
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  int v65; // r9d
  unsigned int i; // eax
  _QWORD *v67; // rsi
  unsigned int v68; // eax
  __int64 v69; // rdx
  int v70; // r10d
  __int64 v71; // r15
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  int v75; // r9d
  __int64 v76; // rax
  __int16 v77; // ax
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rax
  unsigned __int64 v81; // rcx
  __int16 v82; // ax
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  unsigned __int8 **v86; // r12
  unsigned __int64 v87; // r13
  unsigned __int8 *v88; // r15
  __int64 v89; // rsi
  unsigned __int64 v90; // rax
  __int64 v91; // rsi
  __int64 v92; // r14
  unsigned __int8 *v93; // rax
  unsigned __int16 v94; // ax
  __int64 *v95; // rsi
  __int64 *v96; // rax
  __int64 *v97; // r12
  __int64 v98; // rax
  int v99; // esi
  int v100; // edx
  __int64 v101; // rdx
  int v102; // eax
  int v103; // r9d
  __int64 v104; // rax
  char v105; // dl
  char v106; // di
  char v107; // dh
  char v108; // si
  unsigned __int64 v109; // [rsp+8h] [rbp-198h]
  __int64 v110; // [rsp+18h] [rbp-188h]
  __int64 v112; // [rsp+20h] [rbp-180h]
  __int64 v113; // [rsp+28h] [rbp-178h]
  char v114; // [rsp+38h] [rbp-168h]
  __int64 v115; // [rsp+38h] [rbp-168h]
  char v116; // [rsp+38h] [rbp-168h]
  char v117; // [rsp+40h] [rbp-160h]
  __int64 v118; // [rsp+40h] [rbp-160h]
  char v119; // [rsp+48h] [rbp-158h]
  __int64 *v120; // [rsp+48h] [rbp-158h]
  __int64 *v121; // [rsp+50h] [rbp-150h] BYREF
  __int64 *v122; // [rsp+58h] [rbp-148h] BYREF
  __int64 v123; // [rsp+60h] [rbp-140h] BYREF
  __int64 v124; // [rsp+68h] [rbp-138h]
  __int64 v125; // [rsp+70h] [rbp-130h] BYREF
  __int64 v126; // [rsp+78h] [rbp-128h] BYREF
  __int64 v127; // [rsp+80h] [rbp-120h]
  __int64 v128; // [rsp+88h] [rbp-118h]
  __int64 v129; // [rsp+90h] [rbp-110h]
  __int16 v130; // [rsp+98h] [rbp-108h]
  _QWORD v131[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 *v132; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v133; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 v134; // [rsp+C0h] [rbp-E0h] BYREF
  _QWORD v135[8]; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 v136; // [rsp+108h] [rbp-98h] BYREF
  __int64 *v137; // [rsp+110h] [rbp-90h]
  __int64 v138; // [rsp+118h] [rbp-88h]
  int v139; // [rsp+120h] [rbp-80h]
  char v140; // [rsp+124h] [rbp-7Ch]
  __int64 v141; // [rsp+128h] [rbp-78h] BYREF

  v2 = a2;
  v3 = *(_WORD *)(a2 + 24) == 7;
  v4 = *(_BYTE *)(a1 + 584);
  LOBYTE(v125) = 0;
  v139 = 0;
  v5 = *(_QWORD *)(a1 + 576);
  v117 = v4;
  v6 = *(_BYTE *)(a1 + 585);
  v140 = 1;
  v119 = v6;
  v141 = a2;
  v132 = &v125;
  v133 = (__int64)v135;
  v134 = 0x800000000LL;
  v137 = &v141;
  v138 = 0x100000008LL;
  v136 = 1;
  if ( !v3 )
    goto LABEL_116;
  v7 = *(_QWORD *)(a2 + 40);
  if ( !*(_WORD *)(v7 + 24) )
  {
    v8 = *(_QWORD *)(v7 + 32);
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      if ( !*(_QWORD *)(v8 + 24) )
        goto LABEL_5;
    }
    else if ( v9 == (unsigned int)sub_C444A0(v8 + 24) )
    {
      goto LABEL_5;
    }
LABEL_116:
    v135[0] = a2;
    v10 = 1;
    LODWORD(v134) = 1;
    goto LABEL_6;
  }
LABEL_5:
  LOBYTE(v125) = 1;
  v10 = 0;
LABEL_6:
  v113 = v5;
  v11 = v135;
  v12 = (__int64)&v125;
  while ( 1 )
  {
    v13 = &v11[v10];
    if ( !v10 )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)v12 )
        goto LABEL_12;
      v14 = *(v13 - 1);
      LODWORD(v134) = --v10;
      v15 = *(_WORD *)(v14 + 24);
      if ( v15 > 0xEu )
      {
        if ( v15 != 15 )
          BUG();
        goto LABEL_11;
      }
      if ( v15 > 1u )
        break;
LABEL_11:
      --v13;
      if ( !v10 )
        goto LABEL_12;
    }
    v35 = sub_D960E0(v14);
    v39 = (__int64 *)(v35 + 8 * v36);
    if ( (__int64 *)v35 != v39 )
    {
      v40 = (__int64 *)v35;
      while ( 2 )
      {
        v41 = *v40;
        if ( !v140 )
          goto LABEL_43;
        v42 = v137;
        v12 = HIDWORD(v138);
        v36 = (__int64)&v137[HIDWORD(v138)];
        if ( v137 != (__int64 *)v36 )
        {
          while ( v41 != *v42 )
          {
            if ( (__int64 *)v36 == ++v42 )
              goto LABEL_49;
          }
          goto LABEL_40;
        }
LABEL_49:
        if ( HIDWORD(v138) < (unsigned int)v138 )
        {
          ++HIDWORD(v138);
          *(_QWORD *)v36 = v41;
          ++v136;
          if ( *(_WORD *)(v41 + 24) != 7 )
            goto LABEL_51;
LABEL_45:
          v43 = *(_QWORD *)(v41 + 40);
          if ( *(_WORD *)(v43 + 24) )
          {
LABEL_48:
            *(_BYTE *)v132 = 1;
          }
          else
          {
            v44 = *(_QWORD *)(v43 + 32);
            v45 = *(_DWORD *)(v44 + 32);
            if ( v45 > 0x40 )
            {
              if ( v45 != (unsigned int)sub_C444A0(v44 + 24) )
                goto LABEL_51;
              goto LABEL_48;
            }
            if ( !*(_QWORD *)(v44 + 24) )
              goto LABEL_48;
LABEL_51:
            v46 = (unsigned int)v134;
            v47 = (unsigned int)v134 + 1LL;
            if ( v47 > HIDWORD(v134) )
            {
              sub_C8D5F0((__int64)&v133, v135, v47, 8u, v37, v38);
              v46 = (unsigned int)v134;
            }
            v36 = v133;
            *(_QWORD *)(v133 + 8 * v46) = v41;
            LODWORD(v134) = v134 + 1;
          }
        }
        else
        {
LABEL_43:
          sub_C8CC70((__int64)&v136, *v40, v36, v12, v37, v38);
          if ( (_BYTE)v36 )
          {
            if ( *(_WORD *)(v41 + 24) != 7 )
              goto LABEL_51;
            goto LABEL_45;
          }
        }
LABEL_40:
        v12 = (__int64)v132;
        if ( *(_BYTE *)v132 )
          goto LABEL_42;
        if ( v39 == ++v40 )
          goto LABEL_42;
        continue;
      }
    }
    v12 = (__int64)v132;
LABEL_42:
    v11 = (_QWORD *)v133;
    v10 = v134;
  }
LABEL_12:
  v16 = v113;
  v17 = a1;
  if ( !v140 )
  {
    _libc_free(v137, v13);
    v11 = (_QWORD *)v133;
  }
  if ( v11 != v135 )
    _libc_free(v11, v13);
  if ( (_BYTE)v125 )
    goto LABEL_81;
  v18 = *(_QWORD *)a1;
  v19 = *(_QWORD *)(a1 + 568);
  v20 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
  v21 = *(_DWORD *)(v20 + 24);
  v22 = *(_QWORD *)(v20 + 8);
  if ( v21 )
  {
    v23 = v21 - 1;
    v24 = v23 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v19 == *v25 )
    {
LABEL_19:
      v27 = v25[1];
      v28 = v117;
      goto LABEL_26;
    }
    v102 = 1;
    while ( v26 != -4096 )
    {
      v103 = v102 + 1;
      v24 = v23 & (v102 + v24);
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v19 == *v25 )
        goto LABEL_19;
      v102 = v103;
    }
  }
  v27 = 0;
  v28 = v117;
LABEL_26:
  while ( sub_DADE90(v18, v2, v27) )
  {
    if ( !v27 )
    {
      v117 = v28;
      goto LABEL_81;
    }
    v33 = sub_D4B130(v27);
    if ( v33 )
    {
      v29 = v33 + 48;
      v30 = *(_QWORD *)(v33 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v30 == v29 )
      {
        v32 = 0;
      }
      else
      {
        if ( !v30 )
          BUG();
        v31 = *(unsigned __int8 *)(v30 - 24);
        v32 = v30 - 24;
        if ( (unsigned int)(v31 - 30) >= 0xB )
          v32 = 0;
      }
      v119 = 0;
      v16 = v32 + 24;
      v28 = 0;
    }
    else
    {
      v16 = sub_AA5190(**(_QWORD **)(v27 + 32));
      if ( v16 )
      {
        v28 = v34;
        v119 = HIBYTE(v34);
      }
      else
      {
        v119 = 0;
        v28 = 0;
      }
    }
    v27 = *(_QWORD *)v27;
    v18 = *(_QWORD *)a1;
  }
  v117 = v28;
  if ( v27 && sub_DAE0A0(*(_QWORD *)a1, v2, v27) )
  {
    if ( !*(_BYTE *)(a1 + 444) )
    {
      if ( sub_C8CA60(a1 + 416, v27) )
        goto LABEL_55;
      goto LABEL_169;
    }
    v73 = *(_QWORD **)(a1 + 424);
    v74 = &v73[*(unsigned int *)(a1 + 436)];
    if ( v73 == v74 )
    {
LABEL_169:
      v16 = sub_AA5190(**(_QWORD **)(v27 + 32));
      v106 = v105;
      v108 = v107;
      if ( !v16 )
      {
        v108 = 0;
        v106 = 0;
      }
      v119 = v108;
      v117 = v106;
      goto LABEL_55;
    }
    while ( v27 != *v73 )
    {
      if ( v74 == ++v73 )
        goto LABEL_169;
    }
  }
LABEL_55:
  while ( *(_QWORD *)(a1 + 576) != v16 )
  {
    if ( v16 )
    {
      v132 = 0;
      v52 = v16 - 24;
      v133 = 0;
      v134 = v16 - 24;
      if ( v16 != -4072 && v16 != -8168 )
        sub_BD73F0((__int64)&v132);
    }
    else
    {
      v132 = 0;
      v52 = 0;
      v133 = 0;
      v134 = 0;
    }
    v53 = *(_DWORD *)(a1 + 88);
    if ( v53 )
    {
      v48 = v53 - 1;
      v49 = *(_QWORD *)(a1 + 72);
      v50 = (v53 - 1) & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
      v51 = *(_QWORD *)(v49 + 24LL * v50 + 16);
      if ( v51 == v134 )
      {
LABEL_58:
        if ( v134 && v134 != -8192 && v134 != -4096 )
          sub_BD60C0(&v132);
        goto LABEL_62;
      }
      v75 = 1;
      while ( v51 != -4096 )
      {
        v50 = v48 & (v75 + v50);
        v51 = *(_QWORD *)(v49 + 24LL * v50 + 16);
        if ( v134 == v51 )
          goto LABEL_58;
        ++v75;
      }
    }
    v127 = v52;
    v54 = v52;
    v125 = 0;
    v126 = 0;
    v55 = v52 != -8192 && v52 != -4096 && v52 != 0;
    if ( v55 )
    {
      sub_BD73F0((__int64)&v125);
      v54 = v127;
      v55 = v127 != -8192 && v127 != 0 && v127 != -4096;
    }
    v56 = *(_DWORD *)(a1 + 120);
    if ( v56 )
    {
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a1 + 104);
      LODWORD(v59) = v57 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v60 = *(_QWORD *)(v58 + 24LL * (unsigned int)v59 + 16);
      if ( v54 == v60 )
      {
LABEL_72:
        v61 = 1;
        goto LABEL_73;
      }
      v70 = 1;
      while ( v60 != -4096 )
      {
        v59 = v57 & (unsigned int)(v59 + v70);
        v60 = *(_QWORD *)(v58 + 24 * v59 + 16);
        if ( v60 == v54 )
          goto LABEL_72;
        ++v70;
      }
    }
    v61 = 0;
LABEL_73:
    if ( v55 )
    {
      v114 = v61;
      sub_BD60C0(&v125);
      v61 = v114;
    }
    if ( v134 && v134 != -8192 && v134 != -4096 )
    {
      v116 = v61;
      sub_BD60C0(&v132);
      v61 = v116;
    }
    if ( !v61 )
    {
      if ( *(_BYTE *)v52 != 85 )
        break;
      v69 = *(_QWORD *)(v52 - 32);
      if ( !v69
        || *(_BYTE *)v69
        || *(_QWORD *)(v69 + 24) != *(_QWORD *)(v52 + 80)
        || (*(_BYTE *)(v69 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v69 + 36) - 68) > 3 )
      {
        break;
      }
    }
LABEL_62:
    v117 = 0;
    v16 = *(_QWORD *)(v16 + 8);
    v119 = 0;
  }
LABEL_81:
  v62 = 0;
  v63 = *(unsigned int *)(a1 + 56);
  if ( v16 )
    v62 = v16 - 24;
  v64 = *(_QWORD *)(a1 + 40);
  v115 = v62;
  if ( (_DWORD)v63 )
  {
    v65 = 1;
    for ( i = (v63 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4)
                | ((unsigned __int64)(((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4)))); ; i = (v63 - 1) & v68 )
    {
      v67 = (_QWORD *)(v64 + 40LL * i);
      if ( v2 == *v67 && v115 == v67[1] )
        break;
      if ( *v67 == -4096 && v67[1] == -4096 )
        goto LABEL_122;
      v68 = v65 + i;
      ++v65;
    }
    if ( v67 != (_QWORD *)(v64 + 40 * v63) )
      return v67[4];
  }
LABEL_122:
  v76 = *(_QWORD *)(a1 + 568);
  v126 = 0;
  v125 = a1 + 520;
  v128 = v76;
  v127 = 0;
  if ( v76 != 0 && v76 != -4096 && v76 != -8192 )
    sub_BD73F0((__int64)&v126);
  v77 = *(_WORD *)(a1 + 584);
  v129 = *(_QWORD *)(a1 + 576);
  v130 = v77;
  sub_B33910(v131, (__int64 *)(a1 + 520));
  v80 = *(unsigned int *)(a1 + 792);
  v81 = *(unsigned int *)(a1 + 796);
  v131[1] = a1;
  if ( v80 + 1 > v81 )
  {
    sub_C8D5F0(a1 + 784, (const void *)(a1 + 800), v80 + 1, 8u, v78, v79);
    v80 = *(unsigned int *)(a1 + 792);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 784) + 8 * v80) = &v125;
  ++*(_DWORD *)(a1 + 792);
  LOBYTE(v82) = v117;
  HIBYTE(v82) = v119;
  sub_A88F30(a1 + 520, *(_QWORD *)(v115 + 40), v16, v82);
  v132 = &v134;
  v133 = 0x600000000LL;
  v71 = (__int64)sub_F7DFE0(a1, v2, v115, (__int64)&v132);
  if ( v71 )
  {
    v86 = (unsigned __int8 **)v132;
    v120 = &v132[(unsigned int)v133];
    if ( v120 == v132 )
      goto LABEL_141;
    v112 = v71;
    v87 = v109;
    v110 = v2;
    while ( 1 )
    {
      v88 = *v86;
      v89 = (__int64)*v86;
      sub_F83EF0(v17, *v86);
      sub_B44F30(v88);
      sub_B44B50((__int64 *)v88, v89);
      sub_B44A60((__int64)v88);
      v90 = *v88;
      if ( (unsigned __int8)v90 > 0x36u )
        goto LABEL_136;
      v91 = 0x40540000000000LL;
      if ( _bittest64(&v91, v90) )
        break;
LABEL_130:
      if ( v120 == (__int64 *)++v86 )
      {
        v71 = v112;
        v2 = v110;
        goto LABEL_141;
      }
    }
    v89 = (__int64)v88;
    v123 = sub_DDD3C0(*(__int64 **)v17, v88);
    if ( BYTE4(v123) )
    {
      sub_B447F0(v88, (v123 & 2) != 0);
      v89 = ((unsigned int)v123 >> 2) & 1;
      sub_B44850(v88, (v123 & 4) != 0);
    }
    LOBYTE(v90) = *v88;
LABEL_136:
    if ( (((_BYTE)v90 - 68) & 0xFB) == 0 )
    {
      v92 = *((_QWORD *)v88 - 4);
      v118 = *(_QWORD *)(v17 + 8);
      v93 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v92 + 8), v89);
      v87 = v87 & 0xFFFFFF0000000000LL | 0x27;
      v94 = sub_9A1D50(v87, v92, v93, (__int64)v88, v118);
      if ( HIBYTE(v94) )
      {
        if ( (_BYTE)v94 )
          sub_B448D0((__int64)v88, 1);
      }
    }
    goto LABEL_130;
  }
  v104 = sub_F89410(a1, v2, v83, v84, v85);
  v71 = sub_F87310(a1, v104);
LABEL_141:
  v123 = v2;
  v95 = &v123;
  v124 = v115;
  v3 = (unsigned __int8)sub_F81C20(v17 + 32, &v123, &v121) == 0;
  v96 = v121;
  if ( v3 )
  {
    v99 = *(_DWORD *)(v17 + 48);
    ++*(_QWORD *)(v17 + 32);
    v122 = v96;
    v100 = v99 + 1;
    v95 = (__int64 *)*(unsigned int *)(v17 + 56);
    if ( 4 * v100 >= (unsigned int)(3 * (_DWORD)v95) )
    {
      LODWORD(v95) = 2 * (_DWORD)v95;
    }
    else if ( (int)v95 - *(_DWORD *)(v17 + 52) - v100 > (unsigned int)v95 >> 3 )
    {
LABEL_154:
      *(_DWORD *)(v17 + 48) = v100;
      if ( *v96 != -4096 || v96[1] != -4096 )
        --*(_DWORD *)(v17 + 52);
      v97 = v96 + 2;
      *v96 = v123;
      v101 = v124;
      v96[2] = 6;
      v96[1] = v101;
      v96[3] = 0;
      v96[4] = 0;
      if ( !v71 )
        goto LABEL_149;
      goto LABEL_146;
    }
    sub_F835C0(v17 + 32, (int)v95);
    v95 = &v123;
    sub_F81C20(v17 + 32, &v123, &v122);
    v100 = *(_DWORD *)(v17 + 48) + 1;
    v96 = v122;
    goto LABEL_154;
  }
  v97 = v121 + 2;
  v98 = v121[4];
  if ( v71 != v98 )
  {
    if ( v98 != 0 && v98 != -4096 && v98 != -8192 )
      sub_BD60C0(v97);
LABEL_146:
    v97[2] = v71;
    if ( v71 != 0 && v71 != -4096 && v71 != -8192 )
      sub_BD73F0((__int64)v97);
  }
LABEL_149:
  if ( v132 != &v134 )
    _libc_free(v132, v95);
  sub_F80960((__int64)&v125);
  return v71;
}
