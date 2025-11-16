// Function: sub_351C710
// Address: 0x351c710
//
__int64 __fastcall sub_351C710(
        __int64 a1,
        __int64 a2,
        void *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8)
{
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned __int8 v11; // al
  char v12; // al
  _BYTE **v13; // rax
  __int64 v14; // rdx
  void **v15; // r13
  __int64 *v16; // rax
  _QWORD *v17; // r9
  __int64 v18; // r8
  _QWORD *v19; // rdi
  _QWORD *v20; // rsi
  __int64 *v21; // r12
  __int64 v22; // rax
  __int64 *v23; // r13
  _QWORD *v24; // rdi
  _QWORD *v25; // rsi
  unsigned int v26; // esi
  __int64 v27; // r8
  int v28; // r10d
  __int64 *v29; // rdx
  unsigned int v30; // edi
  __int64 *v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  bool v34; // dl
  __int64 v35; // rbx
  int v36; // edx
  __int64 v37; // rcx
  int v38; // edx
  unsigned int v39; // eax
  __int64 v40; // rsi
  int v41; // edi
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  const void *v45; // r14
  __int64 v46; // rax
  unsigned __int64 v47; // rax
  size_t v48; // r12
  int v49; // ebx
  unsigned __int64 v50; // rbx
  void *v51; // r8
  __int64 v52; // r14
  __int64 *v53; // r12
  __int64 *v54; // r9
  __int64 v55; // rbx
  __int64 *v56; // r14
  __int64 *v57; // r15
  __int64 v58; // r12
  char *v59; // rax
  __int64 v60; // rcx
  __int64 *v61; // r12
  unsigned __int64 v62; // r10
  __int64 *v63; // r12
  __int64 v64; // rax
  __int64 *v65; // r9
  __int64 *v66; // r15
  __int64 v67; // r12
  char *v68; // rax
  unsigned __int64 v69; // rbx
  __int64 v70; // rcx
  __int64 *v71; // r12
  unsigned __int64 v72; // rdx
  __int64 *v73; // r14
  __int64 *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rbx
  __int64 v77; // rdi
  __int64 v78; // rsi
  int v79; // eax
  int v80; // eax
  unsigned int v81; // r12d
  int v83; // eax
  __int64 v84; // rcx
  int v85; // edx
  unsigned int v86; // eax
  void *v87; // rsi
  int v88; // edi
  int v89; // esi
  int v90; // esi
  __int64 v91; // r8
  unsigned int v92; // ecx
  __int64 v93; // rdi
  int v94; // r10d
  __int64 *v95; // r11
  int v96; // ecx
  int v97; // ecx
  __int64 v98; // rdi
  __int64 *v99; // r9
  __int64 v100; // r14
  int v101; // r10d
  __int64 v102; // rsi
  char *v103; // rdi
  unsigned __int64 v104; // r13
  unsigned __int64 v105; // rax
  __int64 v106; // r8
  __int64 v107; // r9
  unsigned __int64 v108; // rcx
  bool v109; // cf
  unsigned __int64 v110; // rax
  unsigned int v111; // eax
  unsigned __int64 v112; // rax
  __int64 v113; // rax
  unsigned __int64 v114; // rdx
  __int64 *v115; // rax
  __int64 v116; // r12
  _QWORD *v117; // rdi
  _QWORD *v118; // rsi
  __int64 *v119; // r8
  __int64 *v120; // rax
  __int64 v121; // rsi
  __int64 v122; // rax
  __int64 **v123; // rcx
  __int64 **v124; // r13
  _QWORD *v125; // rdi
  _QWORD *v126; // rsi
  __int64 ***v127; // rax
  __int64 v128; // rdx
  __int64 *v129; // rax
  int v130; // edx
  __int64 v131; // rdi
  int v132; // esi
  unsigned int v133; // edx
  __int64 *v134; // r9
  int v135; // r10d
  unsigned int v136; // eax
  __int64 *v137; // rdi
  unsigned int v138; // r13d
  __int64 v139; // rdi
  __int64 *v140; // rax
  __int64 v141; // rdx
  unsigned __int64 v142; // rsi
  __int64 *v143; // rax
  __int64 v144; // rsi
  int v145; // eax
  int v146; // edx
  unsigned int v147; // eax
  __int64 v148; // rcx
  int i; // edi
  __int64 v151; // [rsp+28h] [rbp-268h]
  unsigned __int64 v152; // [rsp+38h] [rbp-258h]
  unsigned __int64 v153; // [rsp+58h] [rbp-238h]
  __int64 **v154; // [rsp+58h] [rbp-238h]
  void **v155; // [rsp+60h] [rbp-230h]
  unsigned int v156; // [rsp+60h] [rbp-230h]
  void **v157; // [rsp+68h] [rbp-228h]
  __int64 *v158; // [rsp+68h] [rbp-228h]
  __int64 *v159; // [rsp+68h] [rbp-228h]
  __int64 v160; // [rsp+68h] [rbp-228h]
  __int64 v161; // [rsp+70h] [rbp-220h]
  __int64 *v162; // [rsp+70h] [rbp-220h]
  unsigned int v163; // [rsp+78h] [rbp-218h]
  char v164; // [rsp+7Eh] [rbp-212h]
  char v165; // [rsp+7Fh] [rbp-211h]
  void *src; // [rsp+88h] [rbp-208h]
  char *srcc; // [rsp+88h] [rbp-208h]
  void *srca; // [rsp+88h] [rbp-208h]
  __int64 *srcb; // [rsp+88h] [rbp-208h]
  void *srcd; // [rsp+88h] [rbp-208h]
  __int64 *v173; // [rsp+A0h] [rbp-1F0h]
  __int64 v174; // [rsp+A0h] [rbp-1F0h]
  __int64 v175; // [rsp+A8h] [rbp-1E8h] BYREF
  unsigned __int8 v176; // [rsp+B7h] [rbp-1D9h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-1D8h] BYREF
  __int64 *v178; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-1C8h] BYREF
  _QWORD v180[2]; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 *v181; // [rsp+E0h] [rbp-1B0h] BYREF
  __int64 v182; // [rsp+E8h] [rbp-1A8h]
  _QWORD v183[6]; // [rsp+F0h] [rbp-1A0h] BYREF
  void **v184; // [rsp+120h] [rbp-170h] BYREF
  __int64 v185; // [rsp+128h] [rbp-168h]
  _BYTE v186[64]; // [rsp+130h] [rbp-160h] BYREF
  _BYTE *v187; // [rsp+170h] [rbp-120h] BYREF
  __int64 v188; // [rsp+178h] [rbp-118h]
  _BYTE v189[64]; // [rsp+180h] [rbp-110h] BYREF
  void *v190; // [rsp+1C0h] [rbp-D0h] BYREF
  __int64 v191; // [rsp+1C8h] [rbp-C8h]
  _BYTE v192[64]; // [rsp+1D0h] [rbp-C0h] BYREF
  __int64 *v193; // [rsp+210h] [rbp-80h] BYREF
  __int64 v194; // [rsp+218h] [rbp-78h]
  _BYTE v195[112]; // [rsp+220h] [rbp-70h] BYREF

  v8 = a1;
  v9 = a2;
  *a8 = 0;
  v175 = a5;
  v11 = sub_2FD62C0(a2);
  if ( *(_DWORD *)(a2 + 120) == 1 )
    return 0;
  v173 = (__int64 *)(a1 + 600);
  v165 = sub_2FD64C0((__int64 *)(a1 + 600), v11, (__int64 *)a2);
  if ( !v165 )
    return 0;
  v183[2] = a6;
  v183[0] = &v176;
  v183[3] = &v175;
  v176 = 0;
  v183[4] = a7;
  v180[0] = sub_3517D50;
  v180[1] = v183;
  v183[1] = a1;
  v184 = (void **)v186;
  v185 = 0x800000000LL;
  v12 = sub_2FD62C0(a2);
  v188 = 0x800000000LL;
  v164 = v12;
  v187 = v189;
  sub_B2EE70((__int64)&v193, **(_QWORD **)(a1 + 520), 0);
  if ( !v195[0] )
  {
LABEL_4:
    v13 = 0;
LABEL_5:
    sub_2FDA680((__int64)v173, v164, v9, (unsigned __int64)a3, (__int64)&v184, (__int64)v180, (__int64)v13);
    v14 = (unsigned int)v185;
    *a8 = 0;
    v15 = v184;
    v161 = v8 + 888;
    v157 = &v184[v14];
    if ( v157 != v184 )
    {
      do
      {
        while ( 1 )
        {
          v190 = *v15;
          v16 = sub_3515040(v161, (__int64 *)&v190);
          v17 = v190;
          v174 = *v16;
          if ( v190 != a3 )
            break;
          ++v15;
          *a8 = 1;
          if ( v157 == v15 )
            goto LABEL_68;
        }
        v18 = v175;
        if ( v175 )
        {
          if ( *(_DWORD *)(v175 + 16) )
          {
            v83 = *(_DWORD *)(v175 + 24);
            v84 = *(_QWORD *)(v175 + 8);
            if ( !v83 )
              goto LABEL_67;
            v85 = v83 - 1;
            v86 = (v83 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
            v87 = *(void **)(v84 + 8LL * v86);
            if ( v190 != v87 )
            {
              v88 = 1;
              while ( v87 != (void *)-4096LL )
              {
                v86 = v85 & (v88 + v86);
                v87 = *(void **)(v84 + 8LL * v86);
                if ( v190 == v87 )
                  goto LABEL_10;
                ++v88;
              }
              goto LABEL_67;
            }
          }
          else
          {
            v19 = *(_QWORD **)(v175 + 32);
            v20 = &v19[*(unsigned int *)(v175 + 40)];
            if ( v20 == sub_3510810(v19, (__int64)v20, (__int64 *)&v190) )
              goto LABEL_67;
          }
        }
LABEL_10:
        if ( v174 != a4 )
        {
          v21 = (__int64 *)v17[14];
          v22 = *((unsigned int *)v17 + 30);
          if ( v21 != &v21[v22] )
          {
            v155 = v15;
            v23 = &v21[v22];
            while ( 1 )
            {
              v35 = *v21;
              v193 = (__int64 *)*v21;
              if ( !v18 )
                goto LABEL_14;
              if ( !*(_DWORD *)(v18 + 16) )
                break;
              v36 = *(_DWORD *)(v18 + 24);
              v37 = *(_QWORD *)(v18 + 8);
              if ( !v36 )
                goto LABEL_20;
              v38 = v36 - 1;
              v39 = v38 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
              v40 = *(_QWORD *)(v37 + 8LL * v39);
              if ( v35 == v40 )
                goto LABEL_14;
              v41 = 1;
              while ( v40 != -4096 )
              {
                v39 = v38 & (v41 + v39);
                v40 = *(_QWORD *)(v37 + 8LL * v39);
                if ( v35 == v40 )
                  goto LABEL_14;
                ++v41;
              }
LABEL_20:
              if ( v23 == ++v21 )
              {
                v15 = v155;
                goto LABEL_67;
              }
              v18 = v175;
            }
            v24 = *(_QWORD **)(v18 + 32);
            v25 = &v24[*(unsigned int *)(v18 + 40)];
            if ( v25 == sub_3510810(v24, (__int64)v25, (__int64 *)&v193) )
              goto LABEL_20;
LABEL_14:
            v26 = *(_DWORD *)(v8 + 912);
            if ( v26 )
            {
              v27 = *(_QWORD *)(v8 + 896);
              v28 = 1;
              v29 = 0;
              v30 = (v26 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
              v31 = (__int64 *)(v27 + 16LL * v30);
              v32 = *v31;
              if ( v35 == *v31 )
              {
LABEL_16:
                v33 = v31[1];
                v34 = a4 != v33;
LABEL_17:
                if ( v174 != v33 && v34 )
                  ++*(_DWORD *)(v33 + 56);
                goto LABEL_20;
              }
              while ( v32 != -4096 )
              {
                if ( v32 == -8192 && !v29 )
                  v29 = v31;
                v30 = (v26 - 1) & (v28 + v30);
                v31 = (__int64 *)(v27 + 16LL * v30);
                v32 = *v31;
                if ( v35 == *v31 )
                  goto LABEL_16;
                ++v28;
              }
              if ( !v29 )
                v29 = v31;
              v79 = *(_DWORD *)(v8 + 904);
              ++*(_QWORD *)(v8 + 888);
              v80 = v79 + 1;
              if ( 4 * v80 < 3 * v26 )
              {
                if ( v26 - *(_DWORD *)(v8 + 908) - v80 <= v26 >> 3 )
                {
                  sub_3512300(v161, v26);
                  v96 = *(_DWORD *)(v8 + 912);
                  if ( !v96 )
                  {
LABEL_189:
                    ++*(_DWORD *)(v8 + 904);
                    BUG();
                  }
                  v97 = v96 - 1;
                  v98 = *(_QWORD *)(v8 + 896);
                  v99 = 0;
                  LODWORD(v100) = v97 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
                  v101 = 1;
                  v80 = *(_DWORD *)(v8 + 904) + 1;
                  v29 = (__int64 *)(v98 + 16LL * (unsigned int)v100);
                  v102 = *v29;
                  if ( v35 != *v29 )
                  {
                    while ( v102 != -4096 )
                    {
                      if ( !v99 && v102 == -8192 )
                        v99 = v29;
                      v100 = v97 & (unsigned int)(v100 + v101);
                      v29 = (__int64 *)(v98 + 16 * v100);
                      v102 = *v29;
                      if ( v35 == *v29 )
                        goto LABEL_63;
                      ++v101;
                    }
                    if ( v99 )
                      v29 = v99;
                  }
                }
                goto LABEL_63;
              }
            }
            else
            {
              ++*(_QWORD *)(v8 + 888);
            }
            sub_3512300(v161, 2 * v26);
            v89 = *(_DWORD *)(v8 + 912);
            if ( !v89 )
              goto LABEL_189;
            v90 = v89 - 1;
            v91 = *(_QWORD *)(v8 + 896);
            v92 = v90 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v80 = *(_DWORD *)(v8 + 904) + 1;
            v29 = (__int64 *)(v91 + 16LL * v92);
            v93 = *v29;
            if ( v35 != *v29 )
            {
              v94 = 1;
              v95 = 0;
              while ( v93 != -4096 )
              {
                if ( !v95 && v93 == -8192 )
                  v95 = v29;
                v92 = v90 & (v94 + v92);
                v29 = (__int64 *)(v91 + 16LL * v92);
                v93 = *v29;
                if ( v35 == *v29 )
                  goto LABEL_63;
                ++v94;
              }
              if ( v95 )
                v29 = v95;
            }
LABEL_63:
            *(_DWORD *)(v8 + 904) = v80;
            if ( *v29 != -4096 )
              --*(_DWORD *)(v8 + 908);
            *v29 = v35;
            v33 = 0;
            v29[1] = 0;
            v34 = v165;
            goto LABEL_17;
          }
        }
LABEL_67:
        ++v15;
      }
      while ( v157 != v15 );
    }
LABEL_68:
    v81 = v176;
    goto LABEL_69;
  }
  v177 = a2;
  v156 = 0;
  v151 = v175;
  v42 = sub_35108D0(a2);
  v191 = 0x800000000LL;
  v45 = *(const void **)(a2 + 64);
  v46 = *(_QWORD *)(a1 + 776) * v42;
  v190 = v192;
  v152 = v46;
  v47 = *(unsigned int *)(a2 + 72);
  v48 = 8 * v47;
  v49 = *(_DWORD *)(a2 + 72);
  if ( v47 > 8 )
  {
    sub_C8D5F0((__int64)&v190, v192, *(unsigned int *)(a2 + 72), 8u, v43, v44);
    v103 = (char *)v190 + 8 * (unsigned int)v191;
  }
  else
  {
    if ( !v48 )
      goto LABEL_32;
    v103 = v192;
  }
  memcpy(v103, v45, v48);
  LODWORD(v48) = v191;
LABEL_32:
  LODWORD(v191) = v48 + v49;
  v50 = *(unsigned int *)(v177 + 120);
  v51 = *(void **)(v177 + 112);
  v193 = (__int64 *)v195;
  v52 = 8 * v50;
  v194 = 0x800000000LL;
  if ( v50 > 8 )
  {
    srcd = v51;
    sub_C8D5F0((__int64)&v193, v195, v50, 8u, (__int64)v51, v44);
    v51 = srcd;
    v137 = &v193[(unsigned int)v194];
  }
  else
  {
    v53 = (__int64 *)v195;
    if ( !v52 )
      goto LABEL_34;
    v137 = (__int64 *)v195;
  }
  memcpy(v137, v51, 8 * v50);
  v53 = v193;
  LODWORD(v52) = v194;
LABEL_34:
  v54 = &v177;
  LODWORD(v194) = v52 + v50;
  v55 = (unsigned int)(v52 + v50);
  v56 = &v53[v55];
  if ( 8 * v55 )
  {
    src = (void *)v8;
    v57 = v53;
    v58 = (8 * v55) >> 3;
    do
    {
      v158 = v54;
      v59 = (char *)sub_2207800(8 * v58);
      v54 = v158;
      if ( v59 )
      {
        v60 = v58;
        v61 = v57;
        v8 = (__int64)src;
        srcc = v59;
        sub_351C190(v61, v56, v59, v60, v8, v158);
        v62 = (unsigned __int64)srcc;
        goto LABEL_38;
      }
      v58 >>= 1;
    }
    while ( v58 );
    v53 = v57;
    v8 = (__int64)src;
  }
  sub_35132F0(v53, v56, v8, v54);
  v62 = 0;
LABEL_38:
  j_j___libc_free_0(v62);
  v63 = (__int64 *)v190;
  v64 = 8LL * (unsigned int)v191;
  v65 = (__int64 *)((char *)v190 + v64);
  if ( v64 )
  {
    srca = (void *)v8;
    v66 = (__int64 *)v190;
    v67 = v64 >> 3;
    do
    {
      v159 = v65;
      v68 = (char *)sub_2207800(8 * v67);
      v65 = v159;
      v69 = (unsigned __int64)v68;
      if ( v68 )
      {
        v70 = v67;
        v71 = v66;
        v8 = (__int64)srca;
        sub_351C610(v71, v159, v68, v70, (__int64)srca);
        goto LABEL_42;
      }
      v67 >>= 1;
    }
    while ( v67 );
    v63 = v66;
    v8 = (__int64)srca;
  }
  sub_3513070(v63, v65, v8);
  v69 = 0;
LABEL_42:
  j_j___libc_free_0(v69);
  srcb = v193;
  if ( (_DWORD)v194 )
    v156 = 0x80000000 - sub_2E441D0(*(_QWORD *)(v8 + 528), v177, *v193);
  v72 = (unsigned int)v191;
  v162 = (__int64 *)((char *)v190 + 8 * (unsigned int)v191);
  if ( v190 == v162 )
    goto LABEL_150;
  v160 = 0;
  v73 = (__int64 *)v190;
LABEL_51:
  while ( 2 )
  {
    v76 = *v73;
    v77 = *(_QWORD *)(v8 + 536);
    v78 = *v73;
    if ( *(_BYTE *)(v8 + 788) )
    {
      v181 = sub_2F06D30(v77, v78);
      v74 = 0;
      v182 = v75;
      if ( (_BYTE)v75 )
        v74 = v181;
    }
    else
    {
      v74 = (__int64 *)sub_2F06CB0(v77, v78);
    }
    v178 = v74;
    if ( !(unsigned __int8)sub_2FD7360(v173, v177, v76) )
    {
      if ( v160 )
        goto LABEL_50;
      v116 = v177;
      v179 = v76;
      if ( v76 == v177 )
        goto LABEL_50;
      if ( v151 )
      {
        if ( !*(_DWORD *)(v151 + 16) )
        {
          v117 = *(_QWORD **)(v151 + 32);
          v118 = &v117[*(unsigned int *)(v151 + 40)];
          if ( v118 == sub_3510810(v117, (__int64)v118, &v179) )
            goto LABEL_50;
          goto LABEL_125;
        }
        v144 = *(_QWORD *)(v151 + 8);
        v145 = *(_DWORD *)(v151 + 24);
        if ( !v145 )
          goto LABEL_50;
        v146 = v145 - 1;
        v119 = &v179;
        v147 = (v145 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
        v148 = *(_QWORD *)(v144 + 8LL * v147);
        if ( v76 == v148 )
          goto LABEL_125;
        for ( i = 1; ; ++i )
        {
          if ( v148 == -4096 )
            goto LABEL_50;
          v147 = v146 & (i + v147);
          v148 = *(_QWORD *)(v144 + 8LL * v147);
          if ( v76 == v148 )
            break;
        }
      }
      v119 = &v179;
LABEL_125:
      v120 = sub_3515040(v8 + 888, v119);
      v121 = v179;
      v122 = *v120;
      if ( v122 && *(_QWORD *)(*(_QWORD *)v122 + 8LL * *(unsigned int *)(v122 + 8) - 8) != v179 )
        goto LABEL_50;
      v123 = *(__int64 ***)(v179 + 112);
      v154 = &v123[*(unsigned int *)(v179 + 120)];
      if ( v123 == v154 )
      {
        v163 = 0;
        goto LABEL_160;
      }
      v124 = *(__int64 ***)(v179 + 112);
      v163 = 0;
      while ( 1 )
      {
        v129 = *v124;
        v181 = v129;
        if ( (__int64 *)v116 == v129 )
          goto LABEL_132;
        if ( v151 )
        {
          if ( *(_DWORD *)(v151 + 16) )
          {
            v130 = *(_DWORD *)(v151 + 24);
            v131 = *(_QWORD *)(v151 + 8);
            if ( !v130 )
              goto LABEL_132;
            v132 = v130 - 1;
            v133 = (v130 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
            v134 = *(__int64 **)(v131 + 8LL * v133);
            if ( v129 != v134 )
            {
              v135 = 1;
              while ( v134 != (__int64 *)-4096LL )
              {
                v133 = v132 & (v135 + v133);
                v134 = *(__int64 **)(v131 + 8LL * v133);
                if ( v129 == v134 )
                  goto LABEL_130;
                ++v135;
              }
              goto LABEL_132;
            }
          }
          else
          {
            v125 = *(_QWORD **)(v151 + 32);
            v126 = &v125[*(unsigned int *)(v151 + 40)];
            if ( v126 == sub_3510810(v125, (__int64)v126, (__int64 *)&v181) )
              goto LABEL_132;
          }
        }
LABEL_130:
        v127 = (__int64 ***)*sub_3515040(v8 + 888, (__int64 *)&v181);
        if ( !v127 )
        {
          v128 = (__int64)v181;
LABEL_143:
          v136 = sub_2E441D0(*(_QWORD *)(v8 + 528), v179, v128);
          if ( v163 >= v136 )
            v136 = v163;
          v163 = v136;
          goto LABEL_132;
        }
        v128 = (__int64)v181;
        if ( **v127 == v181 )
          goto LABEL_143;
LABEL_132:
        if ( v154 == ++v124 )
        {
          v121 = v179;
LABEL_160:
          v138 = sub_2E441D0(*(_QWORD *)(v8 + 528), v121, v116);
          if ( v138 > v163 )
          {
            v139 = *(_QWORD *)(v8 + 536);
            if ( *(_BYTE *)(v8 + 788) )
            {
              v181 = sub_2F06D30(v139, v179);
              v140 = 0;
              v182 = v141;
              if ( (_BYTE)v141 )
                v140 = v181;
            }
            else
            {
              v140 = (__int64 *)sub_2F06CB0(v139, v179);
            }
            v181 = v140;
            v142 = sub_1098D20((unsigned __int64 *)&v181, v138 - v163);
            if ( *(_QWORD *)(v8 + 776) * sub_35108D0(v116) < v142 )
            {
              v160 = v76;
              v143 = srcb;
              if ( srcb != &v193[(unsigned int)v194] )
                v143 = srcb + 1;
              srcb = v143;
            }
          }
LABEL_50:
          if ( v162 == ++v73 )
            goto LABEL_111;
          goto LABEL_51;
        }
      }
    }
    v104 = -1;
    v105 = sub_1098D20((unsigned __int64 *)&v178, v156);
    v108 = (unsigned __int64)v178;
    v109 = __CFADD__(v178, v105);
    v110 = (unsigned __int64)v178 + v105;
    if ( !v109 )
      v104 = v110;
    if ( srcb == &v193[(unsigned int)v194] )
    {
      if ( (_DWORD)v194 )
        goto LABEL_104;
LABEL_120:
      v108 = 0;
      goto LABEL_104;
    }
    v153 = (unsigned __int64)v178;
    v111 = sub_2E441D0(*(_QWORD *)(v8 + 528), v177, *srcb);
    v112 = sub_1098D20((unsigned __int64 *)&v178, v111);
    if ( v153 <= v112 )
      goto LABEL_120;
    v108 = v153 - v112;
LABEL_104:
    if ( v108 >= v104 || v152 >= v104 - v108 )
      goto LABEL_50;
    v113 = (unsigned int)v188;
    v114 = (unsigned int)v188 + 1LL;
    if ( v114 > HIDWORD(v188) )
    {
      sub_C8D5F0((__int64)&v187, v189, v114, 8u, v106, v107);
      v113 = (unsigned int)v188;
    }
    *(_QWORD *)&v187[8 * v113] = v76;
    LODWORD(v188) = v188 + 1;
    v115 = srcb + 1;
    if ( srcb == &v193[(unsigned int)v194] )
      v115 = srcb;
    ++v73;
    srcb = v115;
    if ( v162 != v73 )
      continue;
    break;
  }
LABEL_111:
  v9 = a2;
  if ( v160 )
    goto LABEL_112;
  v72 = (unsigned int)v191;
LABEL_150:
  if ( (_DWORD)v188 && (unsigned int)v188 < v72 )
  {
    *(_QWORD *)v187 = *(_QWORD *)&v187[8 * (unsigned int)v188 - 8];
    LODWORD(v188) = v188 - 1;
  }
LABEL_112:
  if ( v193 != (__int64 *)v195 )
    _libc_free((unsigned __int64)v193);
  if ( v190 != v192 )
    _libc_free((unsigned __int64)v190);
  if ( (_DWORD)v188 )
  {
    v13 = &v187;
    if ( *(_DWORD *)(v9 + 72) > (unsigned int)v188 )
      goto LABEL_5;
    goto LABEL_4;
  }
  v81 = 0;
LABEL_69:
  if ( v187 != v189 )
    _libc_free((unsigned __int64)v187);
  if ( v184 != (void **)v186 )
    _libc_free((unsigned __int64)v184);
  return v81;
}
