// Function: sub_38653C0
// Address: 0x38653c0
//
void __fastcall sub_38653C0(__int64 a1, __int64 a2, char a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r14
  __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  int v11; // eax
  unsigned int v12; // edx
  __int64 v13; // r15
  unsigned int v14; // r14d
  char *v15; // rdi
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  __int64 **v18; // rax
  __int64 *v19; // rdx
  unsigned __int64 v20; // rdi
  unsigned int v21; // esi
  unsigned __int64 v22; // rdx
  int v23; // ebx
  __int64 v24; // rax
  unsigned int v25; // edi
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  unsigned int v28; // edx
  __int64 v29; // r12
  __int64 v30; // rcx
  int v31; // edx
  __int64 v32; // r8
  int v33; // edi
  unsigned __int64 v34; // rsi
  int v35; // r15d
  unsigned int v36; // ebx
  __int64 v37; // rdi
  unsigned int *v38; // rax
  unsigned int *v39; // rdx
  unsigned __int64 v40; // rax
  __int64 *v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // rax
  signed __int64 v47; // rcx
  _QWORD *v48; // rdx
  _QWORD *v49; // r12
  _QWORD *v50; // r13
  _QWORD *v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rsi
  signed __int64 v54; // rsi
  __int64 v55; // rcx
  _QWORD *v56; // rsi
  _QWORD *v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // rax
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // r13
  unsigned __int64 *v62; // r9
  unsigned int v63; // r8d
  unsigned __int64 *v64; // rax
  unsigned __int64 v65; // rdi
  unsigned int v66; // r8d
  __int64 v67; // rdx
  unsigned __int64 v68; // rsi
  unsigned int *v69; // rcx
  unsigned int *v70; // rax
  unsigned int v71; // eax
  __int64 *v72; // r13
  __int64 v73; // r14
  unsigned int v74; // esi
  unsigned __int64 v75; // rdx
  signed __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 *v78; // rdx
  __int64 *v79; // r13
  int v80; // r8d
  unsigned int v81; // eax
  _QWORD *v82; // r13
  __int64 v83; // rbx
  _QWORD *v84; // r12
  void *v85; // rdi
  __int64 v86; // rax
  unsigned int v87; // r9d
  size_t v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rdx
  _QWORD *v91; // r13
  char v92; // r14
  __int64 v93; // rax
  char v94; // r12
  __int64 v95; // rax
  bool v96; // zf
  unsigned int v97; // eax
  unsigned int *v98; // rbx
  __int64 v99; // rax
  __int64 v100; // rdx
  _QWORD *v101; // r14
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 *v104; // r12
  unsigned __int64 v105; // rdi
  unsigned __int64 *v106; // rcx
  int v107; // r11d
  int v108; // eax
  __int64 v109; // rdx
  unsigned __int64 v110; // rdi
  int v111; // r11d
  int v112; // r11d
  __int64 v113; // rdx
  unsigned __int64 v114; // rdi
  int v115; // r11d
  unsigned __int64 v116; // r9
  _QWORD *v117; // r9
  unsigned __int64 v118; // r9
  __int64 v119; // r13
  int v120; // ecx
  __int64 v121; // rsi
  __int64 *v122; // r14
  _QWORD *v123; // rdx
  _QWORD *v124; // r8
  _BYTE *v125; // rdi
  _BYTE *v126; // rax
  _QWORD *v127; // [rsp+0h] [rbp-1A0h]
  _QWORD *v128; // [rsp+8h] [rbp-198h]
  __int64 v129; // [rsp+20h] [rbp-180h]
  unsigned int v131; // [rsp+30h] [rbp-170h]
  _QWORD *v132; // [rsp+30h] [rbp-170h]
  __int64 v133; // [rsp+38h] [rbp-168h]
  unsigned int v134; // [rsp+38h] [rbp-168h]
  unsigned int v135; // [rsp+38h] [rbp-168h]
  unsigned int v136; // [rsp+48h] [rbp-158h]
  unsigned int v137; // [rsp+48h] [rbp-158h]
  unsigned __int64 v138; // [rsp+48h] [rbp-158h]
  __int64 v139; // [rsp+48h] [rbp-158h]
  unsigned int v140; // [rsp+5Ch] [rbp-144h] BYREF
  __int64 v141; // [rsp+60h] [rbp-140h] BYREF
  unsigned __int64 v142; // [rsp+68h] [rbp-138h]
  __int64 v143; // [rsp+70h] [rbp-130h]
  unsigned int v144; // [rsp+78h] [rbp-128h]
  __int64 v145; // [rsp+80h] [rbp-120h] BYREF
  __int64 v146; // [rsp+88h] [rbp-118h]
  signed __int64 v147; // [rsp+90h] [rbp-110h]
  char *v148; // [rsp+98h] [rbp-108h] BYREF
  __int64 v149; // [rsp+A0h] [rbp-100h]
  unsigned int v150; // [rsp+A8h] [rbp-F8h] BYREF
  unsigned int *v151; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v152; // [rsp+B8h] [rbp-E8h]
  char v153[8]; // [rsp+C0h] [rbp-E0h] BYREF
  char v154[8]; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 v155; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned __int64 v156; // [rsp+D8h] [rbp-C8h]
  __int64 *v157; // [rsp+E0h] [rbp-C0h]
  __int64 *v158; // [rsp+E8h] [rbp-B8h]
  __int64 v159; // [rsp+F0h] [rbp-B0h]
  __int64 *v160; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v161; // [rsp+108h] [rbp-98h]
  __int64 *v162; // [rsp+110h] [rbp-90h] BYREF
  char *v163; // [rsp+118h] [rbp-88h] BYREF
  __int64 v164; // [rsp+120h] [rbp-80h]
  _DWORD v165[30]; // [rsp+128h] [rbp-78h] BYREF

  v6 = a1;
  v129 = a1 + 152;
  v8 = *(_QWORD *)(a1 + 152);
  v9 = v8 + 48LL * *(unsigned int *)(a1 + 160);
  while ( v8 != v9 )
  {
    while ( 1 )
    {
      v9 -= 48;
      v10 = *(_QWORD *)(v9 + 24);
      if ( v10 == v9 + 40 )
        break;
      _libc_free(v10);
      if ( v8 == v9 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(v6 + 160) = 0;
  v11 = *(_DWORD *)(v6 + 16);
  if ( !a3 )
  {
    if ( v11 )
    {
      v12 = 0;
      v13 = v6;
      v14 = 0;
      while ( 1 )
      {
        v160 = (__int64 *)v13;
        v16 = *(_QWORD *)(v13 + 8) + ((unsigned __int64)v14 << 6);
        v161 = *(_QWORD *)(v16 + 32);
        v17 = *(__int64 **)(v16 + 24);
        v163 = (char *)v165;
        v162 = v17;
        v165[0] = v14;
        v164 = 0x200000001LL;
        if ( *(_DWORD *)(v13 + 164) <= v12 )
        {
          sub_38630E0(v129, 0);
          v12 = *(_DWORD *)(v13 + 160);
        }
        v18 = (__int64 **)(*(_QWORD *)(v13 + 152) + 48LL * v12);
        if ( v18 )
        {
          *v18 = v160;
          v18[1] = (__int64 *)v161;
          v19 = v162;
          v18[4] = (__int64 *)0x200000000LL;
          v18[2] = v19;
          v18[3] = (__int64 *)(v18 + 5);
          if ( (_DWORD)v164 )
            sub_385B820((__int64)(v18 + 3), &v163, (unsigned int)v164, a4, a5, a6);
          v12 = *(_DWORD *)(v13 + 160);
        }
        v15 = v163;
        *(_DWORD *)(v13 + 160) = v12 + 1;
        if ( v15 != (char *)v165 )
          _libc_free((unsigned __int64)v15);
        if ( ++v14 >= *(_DWORD *)(v13 + 16) )
          break;
        v12 = *(_DWORD *)(v13 + 160);
      }
    }
    return;
  }
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  if ( !v11 )
  {
    v20 = 0;
    LODWORD(v155) = 0;
    v151 = (unsigned int *)v153;
    v152 = 0x200000000LL;
    v156 = 0;
    v157 = &v155;
    v158 = &v155;
    v159 = 0;
    goto LABEL_19;
  }
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  while ( 1 )
  {
    v29 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + (v24 << 6) + 16);
    if ( !v21 )
    {
      ++v141;
LABEL_30:
      sub_177C7D0((__int64)&v141, 2 * v21);
      if ( !v144 )
        goto LABEL_224;
      LODWORD(v30) = (v144 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v31 = v143 + 1;
      v26 = v142 + 16LL * (unsigned int)v30;
      v32 = *(_QWORD *)v26;
      if ( v29 != *(_QWORD *)v26 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v34 )
            v34 = v26;
          v30 = (v144 - 1) & ((_DWORD)v30 + v33);
          v26 = v142 + 16 * v30;
          v32 = *(_QWORD *)v26;
          if ( v29 == *(_QWORD *)v26 )
            goto LABEL_174;
          ++v33;
        }
        if ( v34 )
          v26 = v34;
      }
      goto LABEL_174;
    }
    v25 = (v21 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
    v26 = v22 + 16LL * v25;
    v27 = *(_QWORD *)v26;
    if ( v29 == *(_QWORD *)v26 )
      goto LABEL_26;
    v115 = 1;
    v116 = 0;
    while ( v27 != -8 )
    {
      if ( v116 || v27 != -16 )
        v26 = v116;
      v25 = (v21 - 1) & (v115 + v25);
      v27 = *(_QWORD *)(v22 + 16LL * v25);
      if ( v29 == v27 )
      {
        v26 = v22 + 16LL * v25;
        goto LABEL_26;
      }
      ++v115;
      v116 = v26;
      v26 = v22 + 16LL * v25;
    }
    if ( v116 )
      v26 = v116;
    ++v141;
    v31 = v143 + 1;
    if ( 4 * ((int)v143 + 1) >= 3 * v21 )
      goto LABEL_30;
    if ( v21 - (v31 + HIDWORD(v143)) <= v21 >> 3 )
    {
      sub_177C7D0((__int64)&v141, v21);
      if ( !v144 )
      {
LABEL_224:
        LODWORD(v143) = v143 + 1;
        BUG();
      }
      v118 = 0;
      LODWORD(v119) = (v144 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v31 = v143 + 1;
      v120 = 1;
      v26 = v142 + 16LL * (unsigned int)v119;
      v121 = *(_QWORD *)v26;
      if ( v29 != *(_QWORD *)v26 )
      {
        while ( v121 != -8 )
        {
          if ( v121 == -16 && !v118 )
            v118 = v26;
          v119 = (v144 - 1) & ((_DWORD)v119 + v120);
          v26 = v142 + 16 * v119;
          v121 = *(_QWORD *)v26;
          if ( v29 == *(_QWORD *)v26 )
            goto LABEL_174;
          ++v120;
        }
        if ( v118 )
          v26 = v118;
      }
    }
LABEL_174:
    LODWORD(v143) = v31;
    if ( *(_QWORD *)v26 != -8 )
      --HIDWORD(v143);
    *(_QWORD *)v26 = v29;
    *(_DWORD *)(v26 + 8) = 0;
LABEL_26:
    *(_DWORD *)(v26 + 8) = v23;
    v28 = *(_DWORD *)(v6 + 16);
    v24 = (unsigned int)(v23 + 1);
    v23 = v24;
    if ( (unsigned int)v24 >= v28 )
      break;
    v22 = v142;
    v21 = v144;
  }
  LODWORD(v155) = 0;
  v151 = (unsigned int *)v153;
  v152 = 0x200000000LL;
  v156 = 0;
  v157 = &v155;
  v158 = &v155;
  v159 = 0;
  if ( !v28 )
  {
    v20 = 0;
    goto LABEL_19;
  }
  v35 = 0;
  v36 = 0;
  v37 = 0;
LABEL_39:
  v38 = v151;
  v39 = &v151[(unsigned int)v152];
  if ( v151 != v39 )
  {
    while ( v35 != *v38 )
    {
      if ( v39 == ++v38 )
        goto LABEL_53;
    }
    if ( v39 != v38 )
      goto LABEL_44;
  }
LABEL_53:
  v44 = *(_QWORD *)(v6 + 8) + (v37 << 6);
  v45 = *(_QWORD *)(v44 + 16);
  v46 = *(unsigned __int8 *)(v44 + 40);
  v146 = 1;
  v47 = (4 * v46) | v45 & 0xFFFFFFFFFFFFFFFBLL;
  v160 = (__int64 *)&v162;
  v161 = 0x200000000LL;
  v145 = (__int64)&v145;
  v147 = v47;
  v48 = *(_QWORD **)(a2 + 16);
  v49 = (_QWORD *)(a2 + 8);
  if ( !v48 )
    goto LABEL_225;
  v50 = (_QWORD *)(a2 + 8);
  v51 = *(_QWORD **)(a2 + 16);
  do
  {
    while ( 1 )
    {
      v52 = v51[2];
      v53 = v51[3];
      if ( v47 <= v51[6] )
        break;
      v51 = (_QWORD *)v51[3];
      if ( !v53 )
        goto LABEL_58;
    }
    v50 = v51;
    v51 = (_QWORD *)v51[2];
  }
  while ( v52 );
LABEL_58:
  if ( v49 == v50 || (v54 = v50[6], v47 < v54) )
LABEL_225:
    BUG();
  if ( (v50[5] & 1) != 0 )
  {
LABEL_64:
    v57 = (_QWORD *)(a2 + 8);
    do
    {
      while ( 1 )
      {
        v58 = v48[2];
        v59 = v48[3];
        if ( v48[6] >= v54 )
          break;
        v48 = (_QWORD *)v48[3];
        if ( !v59 )
          goto LABEL_68;
      }
      v57 = v48;
      v48 = (_QWORD *)v48[2];
    }
    while ( v58 );
LABEL_68:
    if ( v57 != v49 && v57[6] <= v54 )
      v49 = v57;
    goto LABEL_71;
  }
  v55 = v50[4];
  if ( (*(_BYTE *)(v55 + 8) & 1) != 0 )
  {
    v54 = *(_QWORD *)(v55 + 16);
    goto LABEL_64;
  }
  v56 = *(_QWORD **)v55;
  if ( (*(_BYTE *)(*(_QWORD *)v55 + 8LL) & 1) == 0 )
  {
    v117 = (_QWORD *)*v56;
    if ( (*(_BYTE *)(*v56 + 8LL) & 1) != 0 )
    {
      v56 = (_QWORD *)*v56;
    }
    else
    {
      v123 = (_QWORD *)*v117;
      if ( (*(_BYTE *)(*v117 + 8LL) & 1) == 0 )
      {
        v124 = (_QWORD *)*v123;
        if ( (*(_BYTE *)(*v123 + 8LL) & 1) == 0 )
        {
          v125 = (_BYTE *)*v124;
          v127 = (_QWORD *)*v123;
          if ( (*(_BYTE *)(*v124 + 8LL) & 1) == 0 )
          {
            v128 = (_QWORD *)*v117;
            v132 = (_QWORD *)*v56;
            v139 = v50[4];
            v126 = sub_3863620(v125);
            v123 = v128;
            v117 = v132;
            v125 = v126;
            *v127 = v126;
            v55 = v139;
          }
          *v123 = v125;
          v124 = v125;
        }
        *v117 = v124;
        v123 = v124;
      }
      *v56 = v123;
      v56 = v123;
    }
    *(_QWORD *)v55 = v56;
  }
  v50[4] = v56;
  v54 = v56[2];
  v48 = *(_QWORD **)(a2 + 16);
  if ( v48 )
    goto LABEL_64;
LABEL_71:
  if ( (v49[5] & 1) != 0 )
  {
    v133 = v6;
    v60 = (unsigned __int64)(v49 + 4);
    while ( 1 )
    {
      v61 = *(_QWORD *)(v60 + 16) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v144 )
      {
        LODWORD(v62) = v142;
        v63 = (v144 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
        v64 = (unsigned __int64 *)(v142 + 16LL * v63);
        v65 = *v64;
        if ( v61 == *v64 )
        {
          v66 = *((_DWORD *)v64 + 2);
          goto LABEL_76;
        }
        v106 = 0;
        v107 = 1;
        while ( v65 != -8 )
        {
          if ( v65 != -16 || v106 )
            v64 = v106;
          v63 = (v144 - 1) & (v107 + v63);
          v122 = (__int64 *)(v142 + 16LL * v63);
          v65 = *v122;
          if ( v61 == *v122 )
          {
            v66 = *((_DWORD *)v122 + 2);
            goto LABEL_76;
          }
          ++v107;
          v106 = v64;
          v64 = (unsigned __int64 *)(v142 + 16LL * v63);
        }
        if ( !v106 )
          v106 = v64;
        ++v141;
        v108 = v143 + 1;
        if ( 4 * ((int)v143 + 1) < 3 * v144 )
        {
          if ( v144 - HIDWORD(v143) - v108 <= v144 >> 3 )
          {
            sub_177C7D0((__int64)&v141, v144);
            if ( !v144 )
            {
LABEL_226:
              LODWORD(v143) = v143 + 1;
              BUG();
            }
            v62 = 0;
            v112 = 1;
            LODWORD(v113) = (v144 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
            v108 = v143 + 1;
            v106 = (unsigned __int64 *)(v142 + 16LL * (unsigned int)v113);
            v114 = *v106;
            if ( v61 != *v106 )
            {
              while ( v114 != -8 )
              {
                if ( v114 == -16 && !v62 )
                  v62 = v106;
                v113 = (v144 - 1) & ((_DWORD)v113 + v112);
                v106 = (unsigned __int64 *)(v142 + 16 * v113);
                v114 = *v106;
                if ( v61 == *v106 )
                  goto LABEL_143;
                ++v112;
              }
LABEL_154:
              if ( v62 )
                v106 = v62;
              goto LABEL_143;
            }
          }
          goto LABEL_143;
        }
      }
      else
      {
        ++v141;
      }
      sub_177C7D0((__int64)&v141, 2 * v144);
      if ( !v144 )
        goto LABEL_226;
      LODWORD(v109) = (v144 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v108 = v143 + 1;
      v106 = (unsigned __int64 *)(v142 + 16LL * (unsigned int)v109);
      v110 = *v106;
      if ( v61 != *v106 )
      {
        v62 = 0;
        v111 = 1;
        while ( v110 != -8 )
        {
          if ( !v62 && v110 == -16 )
            v62 = v106;
          v109 = (v144 - 1) & ((_DWORD)v109 + v111);
          v106 = (unsigned __int64 *)(v142 + 16 * v109);
          v110 = *v106;
          if ( v61 == *v106 )
            goto LABEL_143;
          ++v111;
        }
        goto LABEL_154;
      }
LABEL_143:
      LODWORD(v143) = v108;
      if ( *v106 != -8 )
        --HIDWORD(v143);
      *v106 = v61;
      v66 = 0;
      *((_DWORD *)v106 + 2) = 0;
LABEL_76:
      v140 = v66;
      if ( !v159 )
      {
        v67 = (unsigned int)v152;
        v68 = (unsigned __int64)v151;
        v69 = &v151[(unsigned int)v152];
        if ( v151 != v69 )
        {
          v70 = v151;
          while ( v66 != *v70 )
          {
            if ( v69 == ++v70 )
              goto LABEL_116;
          }
          if ( v69 != v70 )
            goto LABEL_82;
        }
LABEL_116:
        if ( (unsigned int)v152 <= 1uLL )
        {
          if ( (unsigned int)v152 >= HIDWORD(v152) )
          {
            sub_16CD150((__int64)&v151, v153, 0, 4, v66, (int)v62);
            v66 = v140;
            v69 = &v151[(unsigned int)v152];
          }
          *v69 = v66;
          v66 = v140;
          LODWORD(v152) = v152 + 1;
          goto LABEL_82;
        }
        v138 = v60;
        v131 = v36;
        while ( 1 )
        {
          v98 = (unsigned int *)(v68 + 4 * v67 - 4);
          v99 = sub_B996D0((__int64)v154, v98);
          v101 = (_QWORD *)v100;
          if ( v100 )
          {
            v94 = v99 || (__int64 *)v100 == &v155 || *v98 < *(_DWORD *)(v100 + 32);
            v95 = sub_22077B0(0x28u);
            *(_DWORD *)(v95 + 32) = *v98;
            sub_220F040(v94, v95, v101, &v155);
            ++v159;
          }
          v96 = (_DWORD)v152 == 1;
          v97 = v152 - 1;
          LODWORD(v152) = v152 - 1;
          if ( v96 )
            break;
          v68 = (unsigned __int64)v151;
          v67 = v97;
        }
        v60 = v138;
        v36 = v131;
        v102 = sub_B996D0((__int64)v154, &v140);
        v91 = (_QWORD *)v103;
        if ( v103 )
        {
          if ( !v102 && (__int64 *)v103 != &v155 )
          {
            v92 = v140 < *(_DWORD *)(v103 + 32);
            goto LABEL_114;
          }
LABEL_113:
          v92 = 1;
          goto LABEL_114;
        }
LABEL_115:
        v66 = v140;
        goto LABEL_82;
      }
      v137 = v66;
      v89 = sub_B996D0((__int64)v154, &v140);
      v66 = v137;
      v91 = (_QWORD *)v90;
      if ( v90 )
      {
        if ( v89 || (__int64 *)v90 == &v155 )
          goto LABEL_113;
        v92 = v137 < *(_DWORD *)(v90 + 32);
LABEL_114:
        v93 = sub_22077B0(0x28u);
        *(_DWORD *)(v93 + 32) = v140;
        sub_220F040(v92, v93, v91, &v155);
        ++v159;
        goto LABEL_115;
      }
LABEL_82:
      v71 = v161;
      v72 = &v160[6 * (unsigned int)v161];
      if ( v72 != v160 )
      {
        v73 = (__int64)v160;
        v74 = v66;
        do
        {
          if ( dword_50522E0 < v36 )
            break;
          ++v36;
          if ( (unsigned __int8)sub_385DAA0(v73, v74) )
            goto LABEL_96;
          v73 += 48;
          v74 = v140;
        }
        while ( v72 != (__int64 *)v73 );
        v71 = v161;
        v66 = v74;
      }
      v75 = *(_QWORD *)(v133 + 8) + ((unsigned __int64)v66 << 6);
      v145 = v133;
      v146 = *(_QWORD *)(v75 + 32);
      v76 = *(_QWORD *)(v75 + 24);
      v77 = 0x200000001LL;
      v148 = (char *)&v150;
      v147 = v76;
      v150 = v66;
      v149 = 0x200000001LL;
      if ( v71 >= HIDWORD(v161) )
      {
        sub_38630E0((__int64)&v160, 0);
        v71 = v161;
      }
      v78 = &v160[6 * v71];
      if ( v78 )
      {
        *v78 = v145;
        v78[1] = v146;
        v78[2] = v147;
        v78[3] = (__int64)(v78 + 5);
        v78[4] = 0x200000000LL;
        if ( (_DWORD)v149 )
          sub_385B820((__int64)(v78 + 3), &v148, (__int64)v78, v77, v66, (int)v62);
        v71 = v161;
      }
      LODWORD(v161) = v71 + 1;
      if ( v148 != (char *)&v150 )
        _libc_free((unsigned __int64)v148);
LABEL_96:
      v60 = *(_QWORD *)(v60 + 8) & 0xFFFFFFFFFFFFFFFELL;
      if ( !v60 )
      {
        v6 = v133;
        break;
      }
    }
  }
  v79 = v160;
  v80 = v161;
  if ( 3LL * (unsigned int)v161 )
  {
    v136 = v36;
    v81 = *(_DWORD *)(v6 + 160);
    v82 = v160 + 3;
    v83 = (unsigned int)v161;
    do
    {
      if ( v81 >= *(_DWORD *)(v6 + 164) )
      {
        sub_38630E0(v129, 0);
        v81 = *(_DWORD *)(v6 + 160);
      }
      v84 = (_QWORD *)(*(_QWORD *)(v6 + 152) + 48LL * v81);
      if ( v84 )
      {
        v85 = v84 + 5;
        *v84 = *(v82 - 3);
        v84[1] = *(v82 - 2);
        v86 = *(v82 - 1);
        v84[3] = v84 + 5;
        v84[2] = v86;
        v84[4] = 0x200000000LL;
        v87 = *((_DWORD *)v82 + 2);
        if ( v87 && v84 + 3 != v82 )
        {
          v88 = 4LL * v87;
          if ( v87 <= 2
            || (v135 = *((_DWORD *)v82 + 2),
                sub_16CD150((__int64)(v84 + 3), v84 + 5, v87, 4, v80, v87),
                v85 = (void *)v84[3],
                v87 = v135,
                (v88 = 4LL * *((unsigned int *)v82 + 2)) != 0) )
          {
            v134 = v87;
            memcpy(v85, (const void *)*v82, v88);
            v87 = v134;
          }
          *((_DWORD *)v84 + 8) = v87;
        }
        v81 = *(_DWORD *)(v6 + 160);
      }
      ++v81;
      v82 += 6;
      *(_DWORD *)(v6 + 160) = v81;
      --v83;
    }
    while ( v83 );
    v79 = v160;
    v36 = v136;
    v104 = &v160[6 * (unsigned int)v161];
    if ( v160 != v104 )
    {
      do
      {
        v104 -= 6;
        v105 = v104[3];
        if ( (__int64 *)v105 != v104 + 5 )
          _libc_free(v105);
      }
      while ( v79 != v104 );
      v79 = v160;
    }
  }
  if ( v79 != (__int64 *)&v162 )
    _libc_free((unsigned __int64)v79);
LABEL_44:
  while ( 1 )
  {
    v37 = (unsigned int)(v35 + 1);
    v35 = v37;
    if ( (unsigned int)v37 >= *(_DWORD *)(v6 + 16) )
      break;
    if ( !v159 )
      goto LABEL_39;
    v40 = v156;
    if ( v156 )
    {
      v41 = &v155;
      do
      {
        while ( 1 )
        {
          v42 = *(_QWORD *)(v40 + 16);
          v43 = *(_QWORD *)(v40 + 24);
          if ( (unsigned int)v37 <= *(_DWORD *)(v40 + 32) )
            break;
          v40 = *(_QWORD *)(v40 + 24);
          if ( !v43 )
            goto LABEL_51;
        }
        v41 = (__int64 *)v40;
        v40 = *(_QWORD *)(v40 + 16);
      }
      while ( v42 );
LABEL_51:
      if ( v41 != &v155 && (unsigned int)v37 >= *((_DWORD *)v41 + 8) )
        continue;
    }
    goto LABEL_53;
  }
  v20 = v156;
LABEL_19:
  sub_385BC70(v20);
  if ( v151 != (unsigned int *)v153 )
    _libc_free((unsigned __int64)v151);
  j___libc_free_0(v142);
}
