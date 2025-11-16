// Function: sub_11B39B0
// Address: 0x11b39b0
//
unsigned __int8 *__fastcall sub_11B39B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  char v4; // al
  unsigned int v5; // ebx
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 *v10; // r8
  int v11; // r9d
  int v12; // r10d
  __int64 v13; // r11
  _BYTE *v14; // rdx
  __int64 *v15; // r8
  __int64 *v16; // rcx
  unsigned __int8 *v17; // r14
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 *v21; // rdx
  int v22; // r10d
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int8 *v26; // rax
  __int64 v27; // rdx
  __int64 *i; // rax
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 *v35; // r11
  __int64 *v36; // rax
  __int64 *v37; // r12
  __int64 *v38; // r9
  int v39; // r11d
  __int64 *v40; // r10
  unsigned int v41; // edx
  __int64 *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r14
  _QWORD *v45; // r15
  __int64 *v46; // r9
  __int64 v47; // rbx
  __int64 *v48; // rbx
  __int64 v49; // r13
  __int64 v50; // rax
  char v51; // di
  __int64 v52; // rdx
  unsigned __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // r15
  __int64 *v56; // rdx
  __int64 v57; // rax
  __int64 *v58; // r14
  __int64 *v59; // r13
  __int64 *v60; // r14
  __int64 *v61; // rbx
  __int64 *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // r14
  __int64 *v66; // rax
  __int16 v67; // ax
  __int64 v68; // r14
  char v69; // dh
  __int64 v70; // r8
  char v71; // al
  __int16 v72; // cx
  __int64 *v73; // rbx
  const char *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r12
  __int64 *v77; // r13
  __int64 *v78; // rsi
  int v79; // ecx
  int v80; // r9d
  __int64 *v81; // rdi
  __int64 v82; // rax
  __int64 *v83; // rdx
  __int64 v84; // r10
  __int64 v85; // rdx
  int v86; // eax
  int v87; // eax
  unsigned int v88; // ecx
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rcx
  __int64 v92; // rbx
  __int64 v93; // rbx
  __int64 *v94; // rbx
  _QWORD *v95; // rax
  __int64 v96; // rdx
  unsigned __int64 v97; // rax
  __int64 v98; // r13
  __int64 v99; // r15
  __int64 v100; // r12
  __int64 v101; // r13
  __int64 v102; // rdi
  __int64 v103; // rax
  _QWORD *v104; // rax
  __int64 v105; // rbx
  __int64 *v106; // [rsp+8h] [rbp-208h]
  __int64 v107; // [rsp+10h] [rbp-200h]
  __int64 v108; // [rsp+20h] [rbp-1F0h]
  __int64 v109; // [rsp+28h] [rbp-1E8h]
  __int64 *v110; // [rsp+28h] [rbp-1E8h]
  __int64 *v111; // [rsp+30h] [rbp-1E0h]
  __int64 v112; // [rsp+38h] [rbp-1D8h]
  __int64 v113; // [rsp+38h] [rbp-1D8h]
  __int64 *v114; // [rsp+40h] [rbp-1D0h]
  char v115; // [rsp+48h] [rbp-1C8h]
  __int64 **v116; // [rsp+48h] [rbp-1C8h]
  char v117; // [rsp+50h] [rbp-1C0h]
  int v118; // [rsp+58h] [rbp-1B8h]
  __int64 *v119; // [rsp+58h] [rbp-1B8h]
  char v120; // [rsp+60h] [rbp-1B0h]
  __int64 *v121; // [rsp+60h] [rbp-1B0h]
  __int64 v122; // [rsp+60h] [rbp-1B0h]
  int v123; // [rsp+60h] [rbp-1B0h]
  __int64 *v124; // [rsp+70h] [rbp-1A0h]
  __int64 *v125; // [rsp+70h] [rbp-1A0h]
  __int64 v126; // [rsp+70h] [rbp-1A0h]
  __int64 v127; // [rsp+80h] [rbp-190h]
  __int64 *v128; // [rsp+80h] [rbp-190h]
  int v130; // [rsp+9Ch] [rbp-174h] BYREF
  __int64 v131; // [rsp+A0h] [rbp-170h] BYREF
  __int64 *v132; // [rsp+B0h] [rbp-160h]
  __int64 *v133; // [rsp+B8h] [rbp-158h]
  _QWORD v134[4]; // [rsp+C0h] [rbp-150h] BYREF
  __int16 v135; // [rsp+E0h] [rbp-130h]
  __int64 *v136; // [rsp+F0h] [rbp-120h]
  __int64 v137; // [rsp+F8h] [rbp-118h]
  _BYTE v138[32]; // [rsp+100h] [rbp-110h] BYREF
  __int64 *v139; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v140; // [rsp+128h] [rbp-E8h]
  _BYTE v141[32]; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v142; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v143; // [rsp+158h] [rbp-B8h] BYREF
  __int64 *v144; // [rsp+160h] [rbp-B0h]
  __int64 *v145; // [rsp+168h] [rbp-A8h]
  __int64 v146; // [rsp+170h] [rbp-A0h]
  __int16 v147; // [rsp+178h] [rbp-98h]
  __int64 v148; // [rsp+180h] [rbp-90h] BYREF
  __int64 v149; // [rsp+190h] [rbp-80h] BYREF
  __int64 v150; // [rsp+198h] [rbp-78h]
  __int64 *v151; // [rsp+1A0h] [rbp-70h] BYREF
  unsigned int v152; // [rsp+1A8h] [rbp-68h]
  char v153; // [rsp+1E0h] [rbp-30h] BYREF

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_BYTE *)(v3 + 8);
  if ( v4 != 15 )
  {
    if ( v4 != 16 )
      BUG();
    v5 = *(_DWORD *)(v3 + 32);
    if ( v5 <= 2 )
      goto LABEL_3;
    return 0;
  }
  v5 = *(_DWORD *)(v3 + 12);
  if ( v5 > 2 )
    return 0;
LABEL_3:
  v6 = v5;
  v136 = (__int64 *)v138;
  v137 = 0x200000000LL;
  if ( v5 )
  {
    v7 = v138;
    while ( 1 )
    {
      v7[8] = 0;
      v7 += 16;
      if ( v6 == 1 )
        break;
      v6 = 1;
    }
  }
  LODWORD(v137) = v5;
  if ( !byte_4F90C68 && (unsigned int)sub_2207590(&byte_4F90C68) )
  {
    dword_4F90C70 = 2 * v5;
    sub_2207640(&byte_4F90C68);
  }
  if ( dword_4F90C70 <= 0 )
  {
LABEL_15:
    v10 = v136;
    v16 = &v136[2 * (unsigned int)v137];
  }
  else
  {
    while ( 1 )
    {
      v8 = (__int64)&v136[2 * (unsigned int)v137];
      v9 = (__int64 *)sub_11AEEF0(v136, v8);
      if ( (__int64 *)v8 == v9 )
        break;
      v14 = *(_BYTE **)(v13 - 32);
      if ( *v14 <= 0x1Cu || *(_DWORD *)(v13 + 80) != 1 )
        goto LABEL_23;
      v15 = &v10[2 * **(unsigned int **)(v13 + 72)];
      if ( *((_BYTE *)v15 + 8) )
        v14 = (_BYTE *)*v15;
      *v15 = (__int64)v14;
      *((_BYTE *)v15 + 8) = 1;
      if ( **(_BYTE **)(v13 - 64) != 94 || v12 + 1 == v11 )
        goto LABEL_15;
    }
    v16 = v9;
  }
  v8 = (__int64)v16;
  v17 = 0;
  v18 = (__int64 *)sub_11AEEF0(v10, (__int64)v16);
  if ( v19 != v18 || v10 == v19 )
    goto LABEL_17;
  v21 = v10;
  v22 = 0;
  v23 = 0;
  v8 = 0;
  while ( 1 )
  {
    v24 = *v21;
    if ( !*v21 || *(_BYTE *)v24 != 93 )
      break;
    v25 = *(_QWORD *)(v24 - 32);
    if ( v3 != *(_QWORD *)(v25 + 8) || *(_DWORD *)(v24 + 80) != 1 || **(_DWORD **)(v24 + 72) != v22 )
      goto LABEL_23;
    if ( (_BYTE)v8 )
    {
      if ( !v23 )
LABEL_202:
        BUG();
      if ( v25 != v23 )
        goto LABEL_23;
    }
    else
    {
      v23 = *(_QWORD *)(v24 - 32);
      v8 = 1;
    }
    v21 += 2;
    ++v22;
    if ( v19 == v21 )
    {
      v8 = a2;
      v26 = sub_F162A0(a1, a2, v23);
      v10 = v136;
      v17 = v26;
      goto LABEL_17;
    }
  }
  v27 = *(_QWORD *)(*v10 + 40);
  for ( i = v10 + 2; i != v19; i += 2 )
  {
    v8 = *(_QWORD *)(*i + 40);
    if ( v27 )
    {
      if ( v8 != v27 )
        goto LABEL_23;
    }
    else
    {
      v27 = *(_QWORD *)(*i + 40);
    }
  }
  v127 = v27;
  if ( !v27 )
  {
LABEL_23:
    v17 = 0;
    if ( v10 != (__int64 *)v138 )
      goto LABEL_18;
    return v17;
  }
  v17 = *(unsigned __int8 **)(v27 + 16);
  if ( !v17 )
    goto LABEL_17;
  while ( (unsigned __int8)(**((_BYTE **)v17 + 3) - 30) > 0xAu )
  {
    v17 = (unsigned __int8 *)*((_QWORD *)v17 + 1);
    if ( !v17 )
      goto LABEL_17;
  }
  v139 = (__int64 *)v141;
  v140 = 0x400000000LL;
  v29 = *(_QWORD *)(v27 + 16);
  if ( v29 )
  {
    while ( 1 )
    {
      v30 = *(_QWORD *)(v29 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
        break;
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        goto LABEL_76;
    }
    LODWORD(v31) = 0;
LABEL_56:
    v32 = *(_QWORD *)(v30 + 40);
    if ( (unsigned int)v31 > 0x3F )
    {
      v17 = 0;
      goto LABEL_93;
    }
    v33 = (unsigned int)v31;
    if ( (unsigned int)v31 >= (unsigned __int64)HIDWORD(v140) )
    {
      if ( HIDWORD(v140) < (unsigned __int64)(unsigned int)v31 + 1 )
      {
        v8 = (__int64)v141;
        v126 = *(_QWORD *)(v30 + 40);
        sub_C8D5F0((__int64)&v139, v141, (unsigned int)v31 + 1LL, 8u, v32, (unsigned int)v31 + 1LL);
        v33 = (unsigned int)v140;
        v32 = v126;
      }
      v139[v33] = v32;
      v31 = (unsigned int)(v140 + 1);
      LODWORD(v140) = v140 + 1;
    }
    else
    {
      v34 = &v139[(unsigned int)v31];
      if ( v34 )
      {
        *v34 = v32;
        LODWORD(v31) = v140;
      }
      v31 = (unsigned int)(v31 + 1);
      LODWORD(v140) = v31;
    }
    while ( 1 )
    {
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        break;
      v30 = *(_QWORD *)(v29 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
        goto LABEL_56;
    }
    v35 = v139;
    v124 = &v139[v31];
  }
  else
  {
LABEL_76:
    v35 = (__int64 *)v141;
    v124 = (__int64 *)v141;
  }
  v36 = (__int64 *)&v151;
  v149 = 0;
  v150 = 1;
  do
  {
    *v36 = -4096;
    v36 += 2;
  }
  while ( v36 != (__int64 *)&v153 );
  v115 = 0;
  v117 = 0;
  if ( v124 == v35 )
    goto LABEL_90;
  v109 = v3;
  v37 = v35;
  v107 = a2;
  do
  {
    v44 = *v37;
    v143 = 0;
    v142 = v44;
    if ( (v150 & 1) != 0 )
    {
      v38 = (__int64 *)&v151;
      v8 = 3;
    }
    else
    {
      v38 = v151;
      if ( !v152 )
      {
        v40 = 0;
        goto LABEL_78;
      }
      v8 = v152 - 1;
    }
    v39 = 1;
    v40 = 0;
    v41 = v8 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    v42 = &v38[2 * v41];
    v43 = *v42;
    if ( v44 == *v42 )
      goto LABEL_69;
    while ( v43 != -4096 )
    {
      if ( !v40 && v43 == -8192 )
        v40 = v42;
      v41 = v8 & (v39 + v41);
      v42 = &v38[2 * v41];
      v43 = *v42;
      if ( v44 == *v42 )
        goto LABEL_69;
      ++v39;
    }
    if ( !v40 )
      v40 = v42;
LABEL_78:
    v8 = (__int64)&v142;
    v45 = sub_11B3790((__int64)&v149, &v142, v40);
    *v45 = v142;
    v46 = v136;
    v45[1] = v143;
    v47 = 2LL * (unsigned int)v137;
    v114 = &v46[v47];
    if ( v46 == &v46[v47] )
    {
LABEL_85:
      v53 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v53 == v44 + 48 )
        goto LABEL_201;
      if ( !v53 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v53 - 24) - 30 > 0xA )
LABEL_201:
        BUG();
      if ( *(_BYTE *)(v53 - 24) != 31 || (*(_DWORD *)(v53 - 20) & 0x7FFFFFF) != 1 )
        goto LABEL_90;
    }
    else
    {
      v118 = 0;
      v48 = v46;
      v49 = 0;
      v120 = 0;
      do
      {
        v8 = v127;
        v50 = sub_BD5BF0(*v48, v127, v44);
        if ( *(_BYTE *)v50 <= 0x1Cu )
          goto LABEL_85;
        v51 = v115;
        if ( v127 == *(_QWORD *)(v50 + 40) )
          v51 = 1;
        v115 = v51;
        if ( *(_BYTE *)v50 != 93 )
          goto LABEL_85;
        v52 = *(_QWORD *)(v50 - 32);
        if ( v109 != *(_QWORD *)(v52 + 8) || *(_DWORD *)(v50 + 80) != 1 || **(_DWORD **)(v50 + 72) != v118 )
          goto LABEL_85;
        if ( v120 )
        {
          if ( !v49 )
            goto LABEL_202;
          if ( v52 != v49 )
            goto LABEL_85;
        }
        else
        {
          v120 = 1;
          v49 = *(_QWORD *)(v50 - 32);
        }
        ++v118;
        v48 += 2;
      }
      while ( v114 != v48 );
      v45[1] = v49;
      v117 = 1;
    }
LABEL_69:
    ++v37;
  }
  while ( v124 != v37 );
  v54 = v109;
  v55 = v107;
  if ( v117 )
  {
    v112 = *(_QWORD *)(v107 + 40);
    v8 = (__int64)&v149;
    sub_11B1A30(&v142, &v149);
    if ( (v150 & 1) != 0 )
    {
      v56 = (__int64 *)&v151;
      v57 = 4;
    }
    else
    {
      v56 = v151;
      v57 = v152;
    }
    v121 = &v56[2 * v57];
    if ( v121 != v144 )
    {
      v58 = v145;
      v59 = v144;
      while ( 1 )
      {
        if ( !v59[1] )
        {
          if ( v112 != v127 || v115 || (v93 = 2LL * (unsigned int)v137, v119 = &v136[v93], &v136[v93] == v136) )
          {
LABEL_168:
            v17 = 0;
            goto LABEL_91;
          }
          v94 = v136;
          while ( 1 )
          {
            v8 = v127;
            if ( *(_BYTE *)sub_BD5BF0(*v94, v127, *v59) > 0x15u )
              break;
            v94 += 2;
            if ( v119 == v94 )
              goto LABEL_168;
          }
        }
        for ( v59 += 2; v58 != v59; v59 += 2 )
        {
          if ( *v59 != -4096 && *v59 != -8192 )
            break;
        }
        if ( v121 == v59 )
        {
          v55 = v107;
          break;
        }
      }
    }
    sub_11B1A30(&v131, &v149);
    v60 = v132;
    v61 = v133;
    if ( (v150 & 1) != 0 )
    {
      v62 = (__int64 *)&v151;
      v63 = 4;
    }
    else
    {
      v62 = v151;
      v63 = v152;
    }
    v110 = &v62[2 * v63];
    if ( v110 != v132 )
    {
      v116 = (__int64 **)v54;
      v108 = v55;
      do
      {
        v64 = v60[1];
        if ( !v64 )
        {
          v113 = *v60;
          v97 = *(_QWORD *)(*v60 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v97 != *v60 + 48 )
          {
            if ( !v97 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v97 - 24) - 30 <= 0xA )
              v64 = v97 - 24;
          }
          sub_D5F1F0(*(_QWORD *)(a1 + 32), v64);
          v98 = sub_ACADE0(v116);
          v125 = v136;
          v111 = &v136[2 * (unsigned int)v137];
          if ( v111 != v136 )
          {
            v106 = v61;
            v99 = v98;
            v123 = 0;
            do
            {
              v100 = sub_BD5BF0(*v125, v127, v113);
              v101 = *(_QWORD *)(a1 + 32);
              v135 = 257;
              v102 = *(_QWORD *)(v101 + 80);
              v130 = v123;
              v103 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, int *, __int64))(*(_QWORD *)v102 + 88LL))(
                       v102,
                       v99,
                       v100,
                       &v130,
                       1);
              if ( v103 )
              {
                v99 = v103;
              }
              else
              {
                LOWORD(v146) = 257;
                v104 = sub_BD2C40(104, unk_3F148BC);
                v105 = (__int64)v104;
                if ( v104 )
                {
                  sub_B44260((__int64)v104, *(_QWORD *)(v99 + 8), 65, 2u, 0, 0);
                  *(_QWORD *)(v105 + 72) = v105 + 88;
                  *(_QWORD *)(v105 + 80) = 0x400000000LL;
                  sub_B4FD20(v105, v99, v100, &v130, 1, (__int64)&v142);
                }
                v99 = v105;
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v101 + 88) + 16LL))(
                  *(_QWORD *)(v101 + 88),
                  v105,
                  v134,
                  *(_QWORD *)(v101 + 56),
                  *(_QWORD *)(v101 + 64));
                sub_94AAF0((unsigned int **)v101, v105);
              }
              v125 += 2;
              ++v123;
            }
            while ( v111 != v125 );
            v61 = v106;
            v98 = v99;
          }
          v60[1] = v98;
        }
        do
          v60 += 2;
        while ( v60 != v61 && (*v60 == -8192 || *v60 == -4096) );
      }
      while ( v60 != v110 );
      v54 = (__int64)v116;
      v55 = v108;
    }
    v65 = *(_QWORD *)(a1 + 32);
    v142 = v65;
    v66 = *(__int64 **)(v65 + 48);
    v143 = 0;
    v145 = v66;
    v144 = 0;
    if ( v66 + 512 != 0 && v66 != 0 && v66 != (__int64 *)-8192LL )
      sub_BD73F0((__int64)&v143);
    v67 = *(_WORD *)(v65 + 64);
    v146 = *(_QWORD *)(v65 + 56);
    v147 = v67;
    sub_B33910(&v148, (__int64 *)v65);
    v68 = *(_QWORD *)(a1 + 32);
    v70 = sub_AA4FF0(v127);
    if ( v70 )
      v71 = v69;
    else
      v71 = 0;
    LOBYTE(v72) = 1;
    HIBYTE(v72) = v71;
    sub_A88F30(v68, v127, v70, v72);
    v73 = *(__int64 **)(a1 + 32);
    v74 = sub_BD5D20(v55);
    v135 = 773;
    v134[1] = v75;
    v134[0] = v74;
    v134[2] = ".merged";
    v76 = sub_D5C860(v73, v54, v140, (__int64)v134);
    v128 = &v139[(unsigned int)v140];
    if ( v128 == v139 )
    {
LABEL_172:
      v8 = v55;
      v17 = sub_F162A0(a1, v55, v76);
      sub_F11320((__int64)&v142);
      goto LABEL_91;
    }
    v77 = v139;
    while ( 2 )
    {
      v92 = *v77;
      v134[0] = *v77;
      if ( (v150 & 1) != 0 )
      {
        v78 = (__int64 *)&v151;
        v79 = 3;
        goto LABEL_147;
      }
      v78 = v151;
      if ( v152 )
      {
        v79 = v152 - 1;
LABEL_147:
        v80 = 1;
        v81 = 0;
        LODWORD(v82) = v79 & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
        v83 = &v78[2 * (unsigned int)v82];
        v84 = *v83;
        if ( v92 == *v83 )
        {
LABEL_148:
          v85 = v83[1];
          goto LABEL_149;
        }
        while ( v84 != -4096 )
        {
          if ( !v81 && v84 == -8192 )
            v81 = v83;
          v82 = v79 & (unsigned int)(v82 + v80);
          v83 = &v78[2 * v82];
          v84 = *v83;
          if ( v92 == *v83 )
            goto LABEL_148;
          ++v80;
        }
        if ( v81 )
          v83 = v81;
      }
      else
      {
        v83 = 0;
      }
      v95 = sub_11B3790((__int64)&v149, v134, v83);
      v96 = v134[0];
      v95[1] = 0;
      *v95 = v96;
      v85 = 0;
LABEL_149:
      v86 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
      if ( v86 == *(_DWORD *)(v76 + 72) )
      {
        v122 = v85;
        sub_B48D90(v76);
        v85 = v122;
        v86 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
      }
      v87 = (v86 + 1) & 0x7FFFFFF;
      v88 = v87 | *(_DWORD *)(v76 + 4) & 0xF8000000;
      v89 = *(_QWORD *)(v76 - 8) + 32LL * (unsigned int)(v87 - 1);
      *(_DWORD *)(v76 + 4) = v88;
      if ( *(_QWORD *)v89 )
      {
        v90 = *(_QWORD *)(v89 + 8);
        **(_QWORD **)(v89 + 16) = v90;
        if ( v90 )
          *(_QWORD *)(v90 + 16) = *(_QWORD *)(v89 + 16);
      }
      *(_QWORD *)v89 = v85;
      if ( v85 )
      {
        v91 = *(_QWORD *)(v85 + 16);
        *(_QWORD *)(v89 + 8) = v91;
        if ( v91 )
          *(_QWORD *)(v91 + 16) = v89 + 8;
        *(_QWORD *)(v89 + 16) = v85 + 16;
        *(_QWORD *)(v85 + 16) = v89;
      }
      ++v77;
      *(_QWORD *)(*(_QWORD *)(v76 - 8)
                + 32LL * *(unsigned int *)(v76 + 72)
                + 8LL * ((*(_DWORD *)(v76 + 4) & 0x7FFFFFFu) - 1)) = v92;
      if ( v128 == v77 )
        goto LABEL_172;
      continue;
    }
  }
LABEL_90:
  v17 = 0;
LABEL_91:
  if ( (v150 & 1) == 0 )
  {
    v8 = 16LL * v152;
    sub_C7D6A0((__int64)v151, v8, 8);
  }
LABEL_93:
  if ( v139 != (__int64 *)v141 )
    _libc_free(v139, v8);
  v10 = v136;
LABEL_17:
  if ( v10 != (__int64 *)v138 )
LABEL_18:
    _libc_free(v10, v8);
  return v17;
}
