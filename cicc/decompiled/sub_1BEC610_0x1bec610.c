// Function: sub_1BEC610
// Address: 0x1bec610
//
__int64 __fastcall sub_1BEC610(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rbx
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rbx
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi
  _QWORD *v21; // rax
  _QWORD *i; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  char *v25; // rax
  _BYTE *v26; // r13
  char *v27; // r12
  __int128 v28; // rdi
  char *v29; // rbx
  unsigned int v30; // r9d
  _QWORD *v31; // rax
  _QWORD *v32; // r8
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // r12
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned int v44; // ecx
  int v45; // eax
  _QWORD *v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rax
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r12
  unsigned int v55; // ebx
  unsigned int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // r10
  __int64 *v59; // rbx
  unsigned int v60; // edi
  __int64 **v61; // rax
  __int64 *v62; // rcx
  __int64 *v63; // r13
  __int64 v64; // rax
  __int64 *v65; // r9
  __int64 *j; // r15
  __int64 v67; // rax
  int v68; // r8d
  int v69; // r9d
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // r12
  int v74; // r10d
  __int64 **v75; // rdx
  int v76; // ecx
  signed __int64 v77; // rsi
  unsigned int v78; // ecx
  __int64 v79; // rsi
  unsigned int v80; // eax
  __int64 *v81; // r8
  int v82; // r10d
  __int64 **v83; // r9
  __int64 **v84; // r8
  unsigned int v85; // r12d
  int v86; // r9d
  __int64 *v87; // rsi
  int v88; // r11d
  __int64 *v89; // r8
  int v90; // edx
  unsigned int v91; // eax
  __int64 v92; // rdi
  int v93; // esi
  __int64 *v94; // rcx
  _QWORD *v95; // r15
  __int64 *v96; // r9
  unsigned int v97; // ebx
  int v98; // eax
  __int64 v99; // rcx
  __int64 v100; // [rsp+18h] [rbp-168h]
  __int64 v101; // [rsp+20h] [rbp-160h]
  unsigned int v102; // [rsp+20h] [rbp-160h]
  __int64 v103; // [rsp+28h] [rbp-158h]
  __int64 **v104; // [rsp+28h] [rbp-158h]
  int v105; // [rsp+28h] [rbp-158h]
  __int64 v106; // [rsp+28h] [rbp-158h]
  __int64 v107; // [rsp+30h] [rbp-150h]
  __int64 v108; // [rsp+30h] [rbp-150h]
  char *v109; // [rsp+38h] [rbp-148h]
  __int64 v110; // [rsp+40h] [rbp-140h]
  __int64 v111; // [rsp+48h] [rbp-138h]
  __int64 **v112; // [rsp+48h] [rbp-138h]
  _QWORD *v113; // [rsp+50h] [rbp-130h] BYREF
  __int64 v114; // [rsp+58h] [rbp-128h]
  _QWORD v115[2]; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v116; // [rsp+70h] [rbp-110h]
  void *src; // [rsp+78h] [rbp-108h]
  char *v118; // [rsp+80h] [rbp-100h]
  char *v119; // [rsp+88h] [rbp-F8h]
  __int64 v120; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v121; // [rsp+98h] [rbp-E8h]
  __int64 v122; // [rsp+A0h] [rbp-E0h]
  __int64 v123; // [rsp+A8h] [rbp-D8h]
  __int64 v124; // [rsp+B0h] [rbp-D0h]
  __int64 v125; // [rsp+B8h] [rbp-C8h]
  __int64 v126; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v127; // [rsp+C8h] [rbp-B8h]
  __int64 v128; // [rsp+D0h] [rbp-B0h]
  unsigned int v129; // [rsp+D8h] [rbp-A8h]
  __int64 v130; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v131; // [rsp+E8h] [rbp-98h]
  __int64 v132; // [rsp+F0h] [rbp-90h]
  unsigned int v133; // [rsp+F8h] [rbp-88h]
  __int64 **v134; // [rsp+100h] [rbp-80h]
  __int64 v135; // [rsp+108h] [rbp-78h]
  _BYTE v136[112]; // [rsp+110h] [rbp-70h] BYREF

  v1 = a1[2];
  v2 = a1[1];
  v123 = 0;
  v3 = *a1;
  v124 = 0;
  v122 = v1;
  v134 = (__int64 **)v136;
  v135 = 0x800000000LL;
  v120 = v3;
  v121 = v2;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v113 = v115;
  strcpy((char *)v115, "TopRegion");
  v114 = 9;
  v4 = sub_22077B0(136);
  v5 = v4;
  if ( v4 )
  {
    v6 = v113;
    *(_BYTE *)(v4 + 8) = 1;
    v7 = v114;
    *(_QWORD *)v4 = &unk_49F6D50;
    *(_QWORD *)(v4 + 16) = v4 + 32;
    sub_1BEAFD0((__int64 *)(v4 + 16), v6, (__int64)&v6[v7]);
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = v5 + 72;
    *(_QWORD *)(v5 + 64) = 0x100000000LL;
    *(_QWORD *)(v5 + 88) = 0x100000000LL;
    *(_QWORD *)(v5 + 80) = v5 + 96;
    *(_QWORD *)(v5 + 104) = 0;
    *(_QWORD *)v5 = &unk_49F7138;
    *(_QWORD *)(v5 + 112) = 0;
    *(_QWORD *)(v5 + 120) = 0;
    *(_BYTE *)(v5 + 128) = 0;
  }
  v123 = v5;
  if ( v113 != v115 )
    j_j___libc_free_0(v113, v115[0] + 1LL);
  v8 = sub_13FC520(v120);
  v100 = sub_1BEC0C0((__int64)&v120, v8);
  sub_1BEB840((__int64)&v120, v100, v8);
  v9 = sub_1BEC0C0((__int64)&v120, **(_QWORD **)(v120 + 32));
  v12 = *(unsigned int *)(v100 + 88);
  if ( (unsigned int)v12 >= *(_DWORD *)(v100 + 92) )
  {
    sub_16CD150(v100 + 80, (const void *)(v100 + 96), 0, 8, v10, v11);
    v12 = *(unsigned int *)(v100 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v100 + 80) + 8 * v12) = v9;
  v13 = v120;
  ++*(_DWORD *)(v100 + 88);
  v113 = (_QWORD *)v13;
  v14 = *(_QWORD *)(v13 + 40) - *(_QWORD *)(v13 + 32);
  v114 = 0;
  v15 = (unsigned int)(v14 >> 3) | ((unsigned __int64)(unsigned int)(v14 >> 3) >> 1);
  v16 = (((v15 >> 2) | v15) >> 4) | (v15 >> 2) | v15;
  v17 = (((v16 >> 8) | v16) >> 16) | (v16 >> 8) | v16;
  if ( (_DWORD)v17 == -1 )
  {
    v115[0] = 0;
    v115[1] = 0;
    v116 = 0;
  }
  else
  {
    v18 = 4 * (v17 + 1);
    v19 = (((((((v18 / 3 + 1) | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 2)
            | (v18 / 3 + 1)
            | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 4)
          | (((v18 / 3 + 1) | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 2)
          | (v18 / 3 + 1)
          | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 8)
        | (((((v18 / 3 + 1) | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 2)
          | (v18 / 3 + 1)
          | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 4)
        | (((v18 / 3 + 1) | ((unsigned __int64)(v18 / 3 + 1) >> 1)) >> 2)
        | (v18 / 3 + 1)
        | ((unsigned __int64)(v18 / 3 + 1) >> 1);
    v20 = ((v19 >> 16) | v19) + 1;
    v116 = v20;
    v21 = (_QWORD *)sub_22077B0(16 * v20);
    v115[1] = 0;
    v115[0] = v21;
    for ( i = &v21[2 * v116]; i != v21; v21 += 2 )
    {
      if ( v21 )
        *v21 = -8;
    }
  }
  src = 0;
  v118 = 0;
  v119 = 0;
  v23 = (__int64)(*(_QWORD *)(v13 + 40) - *(_QWORD *)(v13 + 32)) >> 3;
  if ( (_DWORD)v23 )
  {
    v24 = 8LL * (unsigned int)v23;
    v25 = (char *)sub_22077B0(v24);
    v26 = src;
    v27 = v25;
    if ( v118 - (_BYTE *)src > 0 )
    {
      memmove(v25, src, v118 - (_BYTE *)src);
      v77 = v119 - v26;
    }
    else
    {
      if ( !src )
      {
LABEL_15:
        src = v27;
        v118 = v27;
        v119 = &v27[v24];
        goto LABEL_16;
      }
      v77 = v119 - (_BYTE *)src;
    }
    j_j___libc_free_0(v26, v77);
    goto LABEL_15;
  }
LABEL_16:
  *((_QWORD *)&v28 + 1) = v121;
  *(_QWORD *)&v28 = &v113;
  sub_13FF3D0(v28);
  v29 = v118;
  v109 = (char *)src;
  if ( v118 != src )
  {
    while ( 1 )
    {
      v37 = *((_QWORD *)v29 - 1);
      v38 = sub_1BEC0C0((__int64)&v120, v37);
      sub_1BEB840((__int64)&v120, v38, v37);
      v39 = sub_157EBA0(v37);
      if ( (unsigned int)sub_15F4D60(v39) != 1 )
        break;
      v48 = sub_15F4DF0(v39, 0);
      v51 = sub_1BEC0C0((__int64)&v120, v48);
      v52 = *(unsigned int *)(v38 + 88);
      if ( (unsigned int)v52 >= *(_DWORD *)(v38 + 92) )
      {
        sub_16CD150(v38 + 80, (const void *)(v38 + 96), 0, 8, v49, v50);
        v52 = *(unsigned int *)(v38 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(v38 + 80) + 8 * v52) = v51;
      ++*(_DWORD *)(v38 + 88);
LABEL_25:
      v29 -= 8;
      sub_1BEC480((__int64)&v120, v38, v37);
      if ( v109 == v29 )
        goto LABEL_37;
    }
    v40 = sub_15F4DF0(v39, 0);
    v111 = sub_1BEC0C0((__int64)&v120, v40);
    v41 = sub_15F4DF0(v39, 1u);
    v42 = sub_1BEC0C0((__int64)&v120, v41);
    v43 = *(_QWORD *)(v39 - 72);
    v110 = v42;
    if ( v133 )
    {
      v30 = (v133 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v31 = (_QWORD *)(v131 + 16LL * v30);
      v32 = (_QWORD *)*v31;
      if ( v43 == *v31 )
      {
        v33 = v31[1];
        goto LABEL_20;
      }
      v105 = 1;
      v46 = 0;
      while ( v32 != (_QWORD *)-8LL )
      {
        if ( v46 || v32 != (_QWORD *)-16LL )
          v31 = v46;
        v30 = (v133 - 1) & (v105 + v30);
        v95 = (_QWORD *)(v131 + 16LL * v30);
        v32 = (_QWORD *)*v95;
        if ( v43 == *v95 )
        {
          v33 = v95[1];
          goto LABEL_20;
        }
        ++v105;
        v46 = v31;
        v31 = (_QWORD *)(v131 + 16LL * v30);
      }
      if ( !v46 )
        v46 = v31;
      ++v130;
      v45 = v132 + 1;
      if ( 4 * ((int)v132 + 1) < 3 * v133 )
      {
        LODWORD(v32) = v133 - HIDWORD(v132) - v45;
        v30 = v133 >> 3;
        if ( (unsigned int)v32 > v133 >> 3 )
          goto LABEL_31;
        v102 = ((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4);
        v106 = v43;
        sub_1BA21E0((__int64)&v130, v133);
        if ( !v133 )
        {
LABEL_172:
          LODWORD(v132) = v132 + 1;
          BUG();
        }
        v32 = 0;
        v43 = v106;
        v30 = 1;
        v78 = (v133 - 1) & v102;
        v45 = v132 + 1;
        v46 = (_QWORD *)(v131 + 16LL * v78);
        v79 = *v46;
        if ( v106 == *v46 )
          goto LABEL_31;
        while ( v79 != -8 )
        {
          if ( v79 == -16 && !v32 )
            v32 = v46;
          v78 = (v133 - 1) & (v30 + v78);
          v46 = (_QWORD *)(v131 + 16LL * v78);
          v79 = *v46;
          if ( v106 == *v46 )
            goto LABEL_31;
          ++v30;
        }
        goto LABEL_117;
      }
    }
    else
    {
      ++v130;
    }
    v103 = v43;
    sub_1BA21E0((__int64)&v130, 2 * v133);
    if ( !v133 )
      goto LABEL_172;
    v43 = v103;
    v44 = (v133 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
    v45 = v132 + 1;
    v46 = (_QWORD *)(v131 + 16LL * v44);
    v47 = *v46;
    if ( v103 == *v46 )
      goto LABEL_31;
    v30 = 1;
    v32 = 0;
    while ( v47 != -8 )
    {
      if ( !v32 && v47 == -16 )
        v32 = v46;
      v44 = (v133 - 1) & (v30 + v44);
      v46 = (_QWORD *)(v131 + 16LL * v44);
      v47 = *v46;
      if ( v103 == *v46 )
        goto LABEL_31;
      ++v30;
    }
LABEL_117:
    if ( v32 )
      v46 = v32;
LABEL_31:
    LODWORD(v132) = v45;
    if ( *v46 != -8 )
      --HIDWORD(v132);
    *v46 = v43;
    v33 = 0;
    v46[1] = 0;
LABEL_20:
    *(_QWORD *)(v38 + 104) = v33;
    v34 = v38 + 80;
    v35 = *(unsigned int *)(v38 + 88);
    if ( (unsigned int)v35 >= *(_DWORD *)(v38 + 92) )
    {
      sub_16CD150(v34, (const void *)(v38 + 96), 0, 8, (int)v32, v30);
      v35 = *(unsigned int *)(v38 + 88);
      v34 = v38 + 80;
    }
    *(_QWORD *)(*(_QWORD *)(v38 + 80) + 8 * v35) = v111;
    v36 = (unsigned int)(*(_DWORD *)(v38 + 88) + 1);
    *(_DWORD *)(v38 + 88) = v36;
    if ( *(_DWORD *)(v38 + 92) <= (unsigned int)v36 )
    {
      sub_16CD150(v34, (const void *)(v38 + 96), 0, 8, (int)v32, v30);
      v36 = *(unsigned int *)(v38 + 88);
    }
    *(_QWORD *)(*(_QWORD *)(v38 + 80) + 8 * v36) = v110;
    ++*(_DWORD *)(v38 + 88);
    goto LABEL_25;
  }
LABEL_37:
  v53 = sub_13FA560(v120);
  v54 = v53;
  if ( !v129 )
  {
    ++v126;
    goto LABEL_121;
  }
  v55 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
  v56 = (v129 - 1) & v55;
  v57 = (__int64 *)(v127 + 16LL * v56);
  v58 = *v57;
  if ( v54 == *v57 )
  {
LABEL_39:
    v101 = v57[1];
    goto LABEL_40;
  }
  v88 = 1;
  v89 = 0;
  while ( v58 != -8 )
  {
    if ( !v89 && v58 == -16 )
      v89 = v57;
    v56 = (v129 - 1) & (v88 + v56);
    v57 = (__int64 *)(v127 + 16LL * v56);
    v58 = *v57;
    if ( v54 == *v57 )
      goto LABEL_39;
    ++v88;
  }
  if ( !v89 )
    v89 = v57;
  ++v126;
  v90 = v128 + 1;
  if ( 4 * ((int)v128 + 1) >= 3 * v129 )
  {
LABEL_121:
    sub_1BEBF00((__int64)&v126, 2 * v129);
    if ( v129 )
    {
      v90 = v128 + 1;
      v91 = (v129 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v89 = (__int64 *)(v127 + 16LL * v91);
      v92 = *v89;
      if ( v54 != *v89 )
      {
        v93 = 1;
        v94 = 0;
        while ( v92 != -8 )
        {
          if ( !v94 && v92 == -16 )
            v94 = v89;
          v91 = (v129 - 1) & (v93 + v91);
          v89 = (__int64 *)(v127 + 16LL * v91);
          v92 = *v89;
          if ( v54 == *v89 )
            goto LABEL_111;
          ++v93;
        }
        if ( v94 )
          v89 = v94;
      }
      goto LABEL_111;
    }
    goto LABEL_170;
  }
  if ( v129 - HIDWORD(v128) - v90 <= v129 >> 3 )
  {
    sub_1BEBF00((__int64)&v126, v129);
    if ( v129 )
    {
      v96 = 0;
      v97 = (v129 - 1) & v55;
      v90 = v128 + 1;
      v98 = 1;
      v89 = (__int64 *)(v127 + 16LL * v97);
      v99 = *v89;
      if ( v54 != *v89 )
      {
        while ( v99 != -8 )
        {
          if ( !v96 && v99 == -16 )
            v96 = v89;
          v97 = (v129 - 1) & (v98 + v97);
          v89 = (__int64 *)(v127 + 16LL * v97);
          v99 = *v89;
          if ( v54 == *v89 )
            goto LABEL_111;
          ++v98;
        }
        if ( v96 )
          v89 = v96;
      }
      goto LABEL_111;
    }
LABEL_170:
    LODWORD(v128) = v128 + 1;
    BUG();
  }
LABEL_111:
  LODWORD(v128) = v90;
  if ( *v89 != -8 )
    --HIDWORD(v128);
  *v89 = v54;
  v89[1] = 0;
  v101 = 0;
LABEL_40:
  sub_1BEB840((__int64)&v120, v101, v54);
  sub_1BEC480((__int64)&v120, v101, v54);
  v112 = v134;
  v104 = &v134[(unsigned int)v135];
  if ( v134 != v104 )
  {
    while ( 1 )
    {
      v59 = *v112;
      if ( !v133 )
        break;
      v60 = (v133 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v61 = (__int64 **)(v131 + 16LL * v60);
      v62 = *v61;
      if ( v59 != *v61 )
      {
        v74 = 1;
        v75 = 0;
        while ( v62 != (__int64 *)-8LL )
        {
          if ( v62 == (__int64 *)-16LL && !v75 )
            v75 = v61;
          v60 = (v133 - 1) & (v74 + v60);
          v61 = (__int64 **)(v131 + 16LL * v60);
          v62 = *v61;
          if ( v59 == *v61 )
            goto LABEL_43;
          ++v74;
        }
        if ( !v75 )
          v75 = v61;
        ++v130;
        v76 = v132 + 1;
        if ( 4 * ((int)v132 + 1) < 3 * v133 )
        {
          if ( v133 - HIDWORD(v132) - v76 <= v133 >> 3 )
          {
            sub_1BA21E0((__int64)&v130, v133);
            if ( !v133 )
            {
LABEL_171:
              LODWORD(v132) = v132 + 1;
              BUG();
            }
            v84 = 0;
            v85 = (v133 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
            v86 = 1;
            v76 = v132 + 1;
            v75 = (__int64 **)(v131 + 16LL * v85);
            v87 = *v75;
            if ( *v75 != v59 )
            {
              while ( v87 != (__int64 *)-8LL )
              {
                if ( v87 == (__int64 *)-16LL && !v84 )
                  v84 = v75;
                v85 = (v133 - 1) & (v86 + v85);
                v75 = (__int64 **)(v131 + 16LL * v85);
                v87 = *v75;
                if ( v59 == *v75 )
                  goto LABEL_67;
                ++v86;
              }
              if ( v84 )
                v75 = v84;
            }
          }
          goto LABEL_67;
        }
LABEL_87:
        sub_1BA21E0((__int64)&v130, 2 * v133);
        if ( !v133 )
          goto LABEL_171;
        v76 = v132 + 1;
        v80 = (v133 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
        v75 = (__int64 **)(v131 + 16LL * v80);
        v81 = *v75;
        if ( v59 != *v75 )
        {
          v82 = 1;
          v83 = 0;
          while ( v81 != (__int64 *)-8LL )
          {
            if ( !v83 && v81 == (__int64 *)-16LL )
              v83 = v75;
            v80 = (v133 - 1) & (v82 + v80);
            v75 = (__int64 **)(v131 + 16LL * v80);
            v81 = *v75;
            if ( v59 == *v75 )
              goto LABEL_67;
            ++v82;
          }
          if ( v83 )
            v75 = v83;
        }
LABEL_67:
        LODWORD(v132) = v76;
        if ( *v75 != (__int64 *)-8LL )
          --HIDWORD(v132);
        *v75 = v59;
        v63 = 0;
        v75[1] = 0;
        goto LABEL_45;
      }
LABEL_43:
      v63 = v61[1];
      if ( v63 )
        v63 -= 5;
LABEL_45:
      v64 = 24LL * (*((_DWORD *)v59 + 5) & 0xFFFFFFF);
      if ( (*((_BYTE *)v59 + 23) & 0x40) != 0 )
      {
        v65 = (__int64 *)*(v59 - 1);
        v59 = &v65[(unsigned __int64)v64 / 8];
      }
      else
      {
        v65 = &v59[v64 / 0xFFFFFFFFFFFFFFF8LL];
      }
      for ( j = v65; v59 != j; ++*(_DWORD *)(v67 + 16) )
      {
        v67 = sub_1BEB4C0((__int64)&v120, *j);
        v70 = *((unsigned int *)v63 + 22);
        if ( (unsigned int)v70 >= *((_DWORD *)v63 + 23) )
        {
          v108 = v67;
          sub_16CD150((__int64)(v63 + 10), v63 + 12, 0, 8, v68, v69);
          v70 = *((unsigned int *)v63 + 22);
          v67 = v108;
        }
        *(_QWORD *)(v63[10] + 8 * v70) = v67;
        ++*((_DWORD *)v63 + 22);
        v71 = *(unsigned int *)(v67 + 16);
        if ( (unsigned int)v71 >= *(_DWORD *)(v67 + 20) )
        {
          v107 = v67;
          sub_16CD150(v67 + 8, (const void *)(v67 + 24), 0, 8, v68, v69);
          v67 = v107;
          v71 = *(unsigned int *)(v107 + 16);
        }
        j += 3;
        *(_QWORD *)(*(_QWORD *)(v67 + 8) + 8 * v71) = v63 + 5;
      }
      if ( v104 == ++v112 )
        goto LABEL_54;
    }
    ++v130;
    goto LABEL_87;
  }
LABEL_54:
  v72 = v123;
  *(_QWORD *)(v123 + 112) = v100;
  *(_QWORD *)(v100 + 48) = v72;
  *(_QWORD *)(v72 + 120) = v101;
  *(_QWORD *)(v101 + 48) = v72;
  if ( src )
    j_j___libc_free_0(src, v119 - (_BYTE *)src);
  j___libc_free_0(v115[0]);
  if ( v134 != (__int64 **)v136 )
    _libc_free((unsigned __int64)v134);
  j___libc_free_0(v131);
  j___libc_free_0(v127);
  return v72;
}
