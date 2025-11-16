// Function: sub_22D13B0
// Address: 0x22d13b0
//
__int64 __fastcall sub_22D13B0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // r9
  void **v7; // rax
  void **v8; // rdx
  char v9; // bl
  char *v10; // r15
  char *v11; // rbx
  __int64 v12; // rsi
  char v14; // r15
  __int64 **v15; // rdx
  __int64 **v16; // rcx
  __int64 **v17; // rax
  __int64 v18; // r14
  char v19; // cl
  __int64 v20; // rdi
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // rax
  void *v24; // r8
  __int64 v25; // rdx
  char v26; // si
  __int64 v27; // r14
  char v28; // cl
  __int64 k; // r8
  int v30; // edi
  unsigned int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r14
  char v35; // cl
  __int64 v36; // rdi
  int v37; // esi
  unsigned int v38; // edx
  __int64 v39; // rax
  void *v40; // r8
  __int64 v41; // rdx
  __int64 v42; // r14
  char v43; // cl
  unsigned int v44; // edx
  __int64 v45; // rdi
  int v46; // edx
  unsigned int v47; // esi
  __int64 v48; // rax
  void *v49; // r8
  __int64 v50; // rdx
  __int64 v51; // r14
  char v52; // r9
  __int64 v53; // rcx
  int v54; // r8d
  unsigned int v55; // edx
  __int64 v56; // rax
  void *v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rsi
  __int64 v60; // rdx
  __int64 v61; // rcx
  int v62; // r9d
  unsigned int jj; // eax
  __int64 v64; // rsi
  unsigned int v65; // eax
  unsigned int v66; // esi
  __int64 v67; // r13
  char v68; // cl
  __int64 v69; // rdi
  int v70; // esi
  unsigned int v71; // esi
  unsigned int v72; // edx
  __int64 v73; // rax
  void *v74; // r8
  __int64 v75; // rdx
  __int64 *v76; // rax
  unsigned int v77; // edi
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rsi
  int v81; // r9d
  unsigned int j; // eax
  __int64 v83; // rcx
  unsigned int v84; // eax
  unsigned int v85; // esi
  void **v86; // rcx
  __int64 v87; // rax
  int v88; // eax
  __int64 v89; // rax
  __int64 v90; // r10
  __int64 v91; // rax
  __int64 v92; // rdx
  unsigned int v93; // r8d
  void **v94; // rcx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdi
  int v98; // esi
  unsigned int i; // eax
  __int64 v100; // rcx
  unsigned int v101; // eax
  __int64 v102; // rax
  __int64 v103; // rdi
  __int64 (*v104)(); // rax
  __int64 v105; // rdi
  __int64 (__fastcall *v106)(__int64, __int64, __int64, __int64 *); // rax
  char v107; // al
  __int64 v108; // rdi
  __int64 (__fastcall *v109)(__int64, __int64, __int64, __int64 *); // rax
  char v110; // al
  __int64 *v111; // rbx
  __int64 v112; // r8
  __int64 *v113; // r13
  __int64 v114; // rax
  char v115; // al
  __int64 v116; // r9
  __int64 v117; // r12
  char v118; // si
  __int64 v119; // rdi
  int v120; // edx
  __int64 v121; // rax
  __int64 v122; // r10
  __int64 v123; // rdx
  __int64 v124; // rax
  unsigned __int64 v125; // rdx
  __int64 v126; // r12
  __int64 *v127; // rbx
  __int64 v128; // r13
  __int64 *v129; // rdx
  __int64 **v130; // rax
  _QWORD *v131; // rax
  __int64 *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  __int64 v135; // rcx
  int v136; // r10d
  unsigned int kk; // eax
  _QWORD *v138; // rsi
  unsigned int v139; // eax
  __int64 v140; // rax
  __int64 v141; // rcx
  __int64 v142; // r8
  __int64 v143; // r9
  int v144; // eax
  __int64 v145; // rbx
  __int64 v146; // rax
  int v147; // eax
  __int64 v148; // rax
  __int64 v149; // rdx
  __int64 v150; // r9
  int v151; // esi
  unsigned int m; // eax
  __int64 v153; // rcx
  unsigned int v154; // eax
  int v155; // eax
  unsigned int v156; // r8d
  __int64 v157; // rax
  int v158; // eax
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // r9
  int v162; // esi
  unsigned int n; // eax
  __int64 v164; // rcx
  unsigned int v165; // eax
  __int64 v166; // rax
  int v167; // eax
  __int64 v168; // rax
  __int64 v169; // rdx
  __int64 v170; // r9
  int v171; // esi
  unsigned int ii; // eax
  __int64 v173; // rcx
  unsigned int v174; // eax
  __int64 v175; // rdi
  __int64 (__fastcall *v176)(__int64, __int64, __int64, __int64); // rax
  char v177; // al
  __int64 v178; // rax
  __int64 v179; // rax
  __int64 v180; // rdi
  __int64 (__fastcall *v181)(__int64); // rax
  char v182; // al
  int v183; // r9d
  __int64 v184; // rax
  int v185; // r10d
  __int64 v186; // rdi
  __int64 (__fastcall *v187)(__int64); // rax
  char v188; // al
  int v189; // eax
  void **v190; // rax
  __int64 **v191; // rcx
  __int64 **v192; // rdx
  int v193; // r9d
  __int64 v194; // rax
  int v195; // edi
  __int64 v196; // rax
  unsigned __int64 v197; // [rsp+0h] [rbp-180h]
  char *v198; // [rsp+8h] [rbp-178h]
  __int64 *v199; // [rsp+18h] [rbp-168h]
  __int64 v200; // [rsp+20h] [rbp-160h]
  __int64 *v201; // [rsp+28h] [rbp-158h]
  __int64 *v202; // [rsp+30h] [rbp-150h]
  __int64 v203; // [rsp+30h] [rbp-150h]
  char *v204; // [rsp+40h] [rbp-140h]
  __int64 v205; // [rsp+48h] [rbp-138h]
  bool v206; // [rsp+56h] [rbp-12Ah]
  unsigned __int8 v207; // [rsp+57h] [rbp-129h]
  __int64 v211; // [rsp+70h] [rbp-110h] BYREF
  char v212[8]; // [rsp+78h] [rbp-108h] BYREF
  void *v213; // [rsp+80h] [rbp-100h] BYREF
  char v214[8]; // [rsp+88h] [rbp-F8h] BYREF
  __int64 v215; // [rsp+90h] [rbp-F0h]
  char *v216; // [rsp+B0h] [rbp-D0h] BYREF
  int v217; // [rsp+B8h] [rbp-C8h]
  char v218; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD v219[20]; // [rsp+E0h] [rbp-A0h] BYREF

  sub_D480B0((__int64)&v216, *(_QWORD *)(a1 + 8));
  v205 = a3 + 48;
  if ( !*(_BYTE *)(a3 + 76) )
  {
    v76 = sub_C8CA60(v205, (__int64)&unk_4FDBCE0);
    v14 = *(_BYTE *)(a1 + 16);
    v9 = v76 != 0;
    if ( !v14 )
      goto LABEL_78;
LABEL_68:
    v67 = *a4;
    v68 = *(_BYTE *)(*a4 + 8) & 1;
    if ( v68 )
    {
      v69 = v67 + 16;
      v70 = 7;
    }
    else
    {
      v71 = *(_DWORD *)(v67 + 24);
      v69 = *(_QWORD *)(v67 + 16);
      if ( !v71 )
        goto LABEL_122;
      v70 = v71 - 1;
    }
    v72 = v70 & (((unsigned int)&unk_4F8F810 >> 9) ^ ((unsigned int)&unk_4F8F810 >> 4));
    v73 = v69 + 16LL * v72;
    v74 = *(void **)v73;
    if ( *(_UNKNOWN **)v73 == &unk_4F8F810 )
    {
LABEL_73:
      v75 = 128;
      if ( !v68 )
        v75 = 16LL * *(unsigned int *)(v67 + 24);
      if ( v73 == v69 + v75 )
      {
        v95 = a4[1];
        v96 = *(unsigned int *)(v95 + 24);
        v97 = *(_QWORD *)(v95 + 8);
        if ( (_DWORD)v96 )
        {
          v98 = 1;
          for ( i = (v96 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)&unk_4F8F810 >> 9) ^ ((unsigned int)&unk_4F8F810 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v96 - 1) & v101 )
          {
            v100 = v97 + 24LL * i;
            if ( *(_UNKNOWN **)v100 == &unk_4F8F810 && a2 == *(_QWORD *)(v100 + 8) )
              break;
            if ( *(_QWORD *)v100 == -4096 && *(_QWORD *)(v100 + 8) == -4096 )
              goto LABEL_129;
            v101 = v98 + i;
            ++v98;
          }
        }
        else
        {
LABEL_129:
          v100 = v97 + 24 * v96;
        }
        v105 = *(_QWORD *)(*(_QWORD *)(v100 + 16) + 24LL);
        v106 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v105 + 16LL);
        if ( v106 == sub_22CFE40 )
          v107 = sub_1047D40(v105 + 8, a2, a3, a4);
        else
          v107 = v106(v105, a2, a3, a4);
        v214[0] = v107;
        v213 = &unk_4F8F810;
        sub_BBCF50((__int64)v219, v67, (__int64 *)&v213, v214);
        v73 = v219[2];
      }
      v14 = *(_BYTE *)(v73 + 8);
LABEL_78:
      if ( v9 )
        goto LABEL_7;
LABEL_15:
      if ( *(_BYTE *)(a3 + 28) )
      {
        v15 = *(__int64 ***)(a3 + 8);
        v16 = &v15[*(unsigned int *)(a3 + 20)];
        if ( v15 == v16 )
          goto LABEL_7;
        v17 = *(__int64 ***)(a3 + 8);
        while ( *v17 != &qword_4F82400 )
        {
          if ( v16 == ++v17 )
            goto LABEL_111;
        }
      }
      else if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
      {
        if ( *(_BYTE *)(a3 + 28) )
        {
          v15 = *(__int64 ***)(a3 + 8);
          v17 = &v15[*(unsigned int *)(a3 + 20)];
          if ( v15 == v17 )
            goto LABEL_7;
LABEL_111:
          v94 = (void **)v15;
          while ( *v94 != &unk_4FDBCE0 )
          {
            if ( ++v94 == (void **)v17 )
              goto LABEL_94;
          }
        }
        else if ( !sub_C8CA60(a3, (__int64)&unk_4FDBCE0) )
        {
          if ( *(_BYTE *)(a3 + 28) )
          {
            v15 = *(__int64 ***)(a3 + 8);
            v17 = &v15[*(unsigned int *)(a3 + 20)];
            if ( v15 == v17 )
              goto LABEL_7;
LABEL_94:
            v86 = (void **)v15;
            while ( *v15 != &qword_4F82400 )
            {
              if ( v17 == ++v15 )
                goto LABEL_238;
            }
          }
          else if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
          {
            if ( *(_BYTE *)(a3 + 28) )
            {
              v86 = *(void ***)(a3 + 8);
              v17 = (__int64 **)&v86[*(unsigned int *)(a3 + 20)];
              if ( v17 == (__int64 **)v86 )
                goto LABEL_7;
LABEL_238:
              while ( *v86 != &unk_4F82420 )
              {
                if ( v17 == (__int64 **)++v86 )
                  goto LABEL_7;
              }
            }
            else if ( !sub_C8CA60(a3, (__int64)&unk_4F82420) )
            {
              goto LABEL_7;
            }
          }
        }
      }
      v18 = *a4;
      v19 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v19 )
      {
        v20 = v18 + 16;
        v21 = 7;
      }
      else
      {
        v66 = *(_DWORD *)(v18 + 24);
        v20 = *(_QWORD *)(v18 + 16);
        if ( !v66 )
          goto LABEL_99;
        v21 = v66 - 1;
      }
      v22 = v21 & (((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4));
      v23 = v20 + 16LL * v22;
      v24 = *(void **)v23;
      if ( *(_UNKNOWN **)v23 == &unk_4F86540 )
        goto LABEL_23;
      v88 = 1;
      while ( v24 != (void *)-4096LL )
      {
        v6 = (unsigned int)(v88 + 1);
        v178 = v21 & (v22 + v88);
        v22 = v178;
        v23 = v20 + 16 * v178;
        v24 = *(void **)v23;
        if ( *(_UNKNOWN **)v23 == &unk_4F86540 )
          goto LABEL_23;
        v88 = v6;
      }
      if ( v19 )
      {
        v87 = 128;
        goto LABEL_100;
      }
      v66 = *(_DWORD *)(v18 + 24);
LABEL_99:
      v87 = 16LL * v66;
LABEL_100:
      v23 = v20 + v87;
LABEL_23:
      v25 = 128;
      if ( !v19 )
        v25 = 16LL * *(unsigned int *)(v18 + 24);
      if ( v23 == v20 + v25 )
      {
        v78 = a4[1];
        v79 = *(unsigned int *)(v78 + 24);
        v80 = *(_QWORD *)(v78 + 8);
        if ( (_DWORD)v79 )
        {
          v81 = 1;
          for ( j = (v79 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v79 - 1) & v84 )
          {
            v83 = v80 + 24LL * j;
            if ( *(_UNKNOWN **)v83 == &unk_4F86540 && a2 == *(_QWORD *)(v83 + 8) )
              break;
            if ( *(_QWORD *)v83 == -4096 && *(_QWORD *)(v83 + 8) == -4096 )
              goto LABEL_134;
            v84 = v81 + j;
            ++v81;
          }
        }
        else
        {
LABEL_134:
          v83 = v80 + 24 * v79;
        }
        v108 = *(_QWORD *)(*(_QWORD *)(v83 + 16) + 24LL);
        v109 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v108 + 16LL);
        if ( v109 == sub_D32140 )
          v110 = sub_CF8780(v108 + 8, a2, a3, a4);
        else
          v110 = v109(v108, a2, a3, a4);
        v214[0] = v110;
        v213 = &unk_4F86540;
        sub_BBCF50((__int64)v219, v18, (__int64 *)&v213, v214);
        v23 = v219[2];
      }
      v26 = *(_BYTE *)(v23 + 8);
      if ( v26 )
        goto LABEL_7;
      v27 = *a4;
      v28 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v28 )
      {
        k = v27 + 16;
        v30 = 7;
      }
      else
      {
        v77 = *(_DWORD *)(v27 + 24);
        k = *(_QWORD *)(v27 + 16);
        if ( !v77 )
          goto LABEL_143;
        v30 = v77 - 1;
      }
      v31 = v30 & (((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4));
      v32 = k + 16LL * v31;
      v6 = *(_QWORD *)v32;
      if ( *(_UNKNOWN **)v32 == &unk_4F86630 )
        goto LABEL_30;
      v147 = 1;
      while ( v6 != -4096 )
      {
        v185 = v147 + 1;
        v31 = v30 & (v147 + v31);
        v32 = k + 16LL * v31;
        v6 = *(_QWORD *)v32;
        if ( *(_UNKNOWN **)v32 == &unk_4F86630 )
          goto LABEL_30;
        v147 = v185;
      }
      if ( v28 )
      {
        v114 = 128;
        goto LABEL_144;
      }
      v77 = *(_DWORD *)(v27 + 24);
LABEL_143:
      v114 = 16LL * v77;
LABEL_144:
      v32 = k + v114;
LABEL_30:
      v33 = 128;
      if ( !v28 )
        v33 = 16LL * *(unsigned int *)(v27 + 24);
      if ( v32 == k + v33 )
      {
        v89 = a4[1];
        v90 = *(_QWORD *)(v89 + 8);
        v91 = *(unsigned int *)(v89 + 24);
        if ( (_DWORD)v91 )
        {
          v6 = 1;
          for ( k = ((_DWORD)v91 - 1)
                  & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                   * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                                    | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9)
                                                        ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; k = ((_DWORD)v91 - 1) & v93 )
          {
            v92 = v90 + 24LL * (unsigned int)k;
            if ( *(_UNKNOWN **)v92 == &unk_4F86630 && a2 == *(_QWORD *)(v92 + 8) )
              break;
            if ( *(_QWORD *)v92 == -4096 && *(_QWORD *)(v92 + 8) == -4096 )
              goto LABEL_257;
            v93 = v6 + k;
            v6 = (unsigned int)(v6 + 1);
          }
        }
        else
        {
LABEL_257:
          v92 = v90 + 24 * v91;
        }
        v103 = *(_QWORD *)(*(_QWORD *)(v92 + 16) + 24LL);
        v104 = *(__int64 (**)())(*(_QWORD *)v103 + 16LL);
        if ( v104 != sub_D000F0 )
          v26 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, __int64, __int64))v104)(
                  v103,
                  a2,
                  a3,
                  a4,
                  k,
                  v6);
        v214[0] = v26;
        v213 = &unk_4F86630;
        sub_BBCF50((__int64)v219, v27, (__int64 *)&v213, v214);
        v32 = v219[2];
      }
      if ( *(_BYTE *)(v32 + 8) )
        goto LABEL_7;
      v34 = *a4;
      v35 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v35 )
      {
        v36 = v34 + 16;
        v37 = 7;
      }
      else
      {
        v85 = *(_DWORD *)(v34 + 24);
        v36 = *(_QWORD *)(v34 + 16);
        if ( !v85 )
          goto LABEL_254;
        v37 = v85 - 1;
      }
      v38 = v37 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
      v39 = v36 + 16LL * v38;
      v40 = *(void **)v39;
      if ( *(_UNKNOWN **)v39 == &unk_4F81450 )
        goto LABEL_37;
      v158 = 1;
      while ( v40 != (void *)-4096LL )
      {
        v183 = v158 + 1;
        v184 = v37 & (v38 + v158);
        v38 = v184;
        v39 = v36 + 16 * v184;
        v40 = *(void **)v39;
        if ( *(_UNKNOWN **)v39 == &unk_4F81450 )
          goto LABEL_37;
        v158 = v183;
      }
      if ( v35 )
      {
        v157 = 128;
        goto LABEL_255;
      }
      v85 = *(_DWORD *)(v34 + 24);
LABEL_254:
      v157 = 16LL * v85;
LABEL_255:
      v39 = v36 + v157;
LABEL_37:
      v41 = 128;
      if ( !v35 )
        v41 = 16LL * *(unsigned int *)(v34 + 24);
      if ( v39 == v36 + v41 )
      {
        v148 = a4[1];
        v149 = *(unsigned int *)(v148 + 24);
        v150 = *(_QWORD *)(v148 + 8);
        if ( (_DWORD)v149 )
        {
          v151 = 1;
          for ( m = (v149 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; m = (v149 - 1) & v154 )
          {
            v153 = v150 + 24LL * m;
            if ( *(_UNKNOWN **)v153 == &unk_4F81450 && a2 == *(_QWORD *)(v153 + 8) )
              break;
            if ( *(_QWORD *)v153 == -4096 && *(_QWORD *)(v153 + 8) == -4096 )
              goto LABEL_291;
            v154 = v151 + m;
            ++v151;
          }
        }
        else
        {
LABEL_291:
          v153 = v150 + 24 * v149;
        }
        v175 = *(_QWORD *)(*(_QWORD *)(v153 + 16) + 24LL);
        v176 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v175 + 16LL);
        if ( v176 == sub_D00100 )
          v177 = sub_B19B20(v175 + 8, a2, a3, (__int64)a4);
        else
          v177 = v176(v175, a2, a3, (__int64)a4);
        v214[0] = v177;
        v213 = &unk_4F81450;
        sub_BBCF50((__int64)v219, v34, (__int64 *)&v213, v214);
        v39 = v219[2];
      }
      if ( *(_BYTE *)(v39 + 8) )
        goto LABEL_7;
      v42 = *a4;
      v43 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v43 )
      {
        v45 = v42 + 16;
        v46 = 7;
      }
      else
      {
        v44 = *(_DWORD *)(v42 + 24);
        v45 = *(_QWORD *)(v42 + 16);
        if ( !v44 )
          goto LABEL_270;
        v46 = v44 - 1;
      }
      v47 = v46 & (((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4));
      v48 = v45 + 16LL * v47;
      v49 = *(void **)v48;
      if ( *(_UNKNOWN **)v48 == &unk_4F875F0 )
        goto LABEL_45;
      v167 = 1;
      while ( v49 != (void *)-4096LL )
      {
        v193 = v167 + 1;
        v194 = v46 & (v47 + v167);
        v47 = v194;
        v48 = v45 + 16 * v194;
        v49 = *(void **)v48;
        if ( *(_UNKNOWN **)v48 == &unk_4F875F0 )
          goto LABEL_45;
        v167 = v193;
      }
      if ( v43 )
      {
        v166 = 128;
        goto LABEL_271;
      }
      v44 = *(_DWORD *)(v42 + 24);
LABEL_270:
      v166 = 16LL * v44;
LABEL_271:
      v48 = v45 + v166;
LABEL_45:
      v50 = 128;
      if ( !v43 )
        v50 = 16LL * *(unsigned int *)(v42 + 24);
      if ( v48 == v45 + v50 )
      {
        v159 = a4[1];
        v160 = *(unsigned int *)(v159 + 24);
        v161 = *(_QWORD *)(v159 + 8);
        if ( (_DWORD)v160 )
        {
          v162 = 1;
          for ( n = (v160 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; n = (v160 - 1) & v165 )
          {
            v164 = v161 + 24LL * n;
            if ( *(_UNKNOWN **)v164 == &unk_4F875F0 && a2 == *(_QWORD *)(v164 + 8) )
              break;
            if ( *(_QWORD *)v164 == -4096 && *(_QWORD *)(v164 + 8) == -4096 )
              goto LABEL_296;
            v165 = v162 + n;
            ++v162;
          }
        }
        else
        {
LABEL_296:
          v164 = v161 + 24 * v160;
        }
        v180 = *(_QWORD *)(*(_QWORD *)(v164 + 16) + 24LL);
        v181 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v180 + 16LL);
        if ( v181 == sub_D32160 )
          v182 = sub_D49500(v180 + 8, a2, a3, (__int64)a4);
        else
          v182 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v181)(v180, a2, a3, a4);
        v214[0] = v182;
        v213 = &unk_4F875F0;
        sub_BBCF50((__int64)v219, v42, (__int64 *)&v213, v214);
        v48 = v219[2];
      }
      if ( *(_BYTE *)(v48 + 8) )
        goto LABEL_7;
      v51 = *a4;
      v52 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v52 )
      {
        v53 = v51 + 16;
        v54 = 7;
      }
      else
      {
        v156 = *(_DWORD *)(v51 + 24);
        v53 = *(_QWORD *)(v51 + 16);
        if ( !v156 )
          goto LABEL_293;
        v54 = v156 - 1;
      }
      v55 = v54 & (((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4));
      v56 = v53 + 16LL * v55;
      v57 = *(void **)v56;
      if ( *(_UNKNOWN **)v56 == &unk_4F881D0 )
        goto LABEL_52;
      v189 = 1;
      while ( v57 != (void *)-4096LL )
      {
        v195 = v189 + 1;
        v196 = v54 & (v55 + v189);
        v55 = v196;
        v56 = v53 + 16 * v196;
        v57 = *(void **)v56;
        if ( *(_UNKNOWN **)v56 == &unk_4F881D0 )
          goto LABEL_52;
        v189 = v195;
      }
      if ( v52 )
      {
        v179 = 128;
        goto LABEL_294;
      }
      v156 = *(_DWORD *)(v51 + 24);
LABEL_293:
      v179 = 16LL * v156;
LABEL_294:
      v56 = v53 + v179;
LABEL_52:
      v58 = 128;
      if ( !v52 )
        v58 = 16LL * *(unsigned int *)(v51 + 24);
      if ( v56 == v53 + v58 )
      {
        v168 = a4[1];
        v169 = *(unsigned int *)(v168 + 24);
        v170 = *(_QWORD *)(v168 + 8);
        if ( (_DWORD)v169 )
        {
          v171 = 1;
          for ( ii = (v169 - 1)
                   & (((0xBF58476D1CE4E5B9LL
                      * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                       | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
                    ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; ii = (v169 - 1) & v174 )
          {
            v173 = v170 + 24LL * ii;
            if ( *(_UNKNOWN **)v173 == &unk_4F881D0 && a2 == *(_QWORD *)(v173 + 8) )
              break;
            if ( *(_QWORD *)v173 == -4096 && *(_QWORD *)(v173 + 8) == -4096 )
              goto LABEL_317;
            v174 = v171 + ii;
            ++v171;
          }
        }
        else
        {
LABEL_317:
          v173 = v170 + 24 * v169;
        }
        v186 = *(_QWORD *)(*(_QWORD *)(v173 + 16) + 24LL);
        v187 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v186 + 16LL);
        if ( v187 == sub_D32150 )
          v188 = sub_DF3010(v186 + 8, a2, a3, a4);
        else
          v188 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v187)(v186, a2, a3, a4);
        v214[0] = v188;
        v213 = &unk_4F881D0;
        sub_BBCF50((__int64)v219, v51, (__int64 *)&v213, v214);
        v56 = v219[2];
      }
      v207 = v14 | *(_BYTE *)(v56 + 8);
      if ( v207 )
        goto LABEL_7;
      if ( *(_DWORD *)(a3 + 68) != *(_DWORD *)(a3 + 72) )
        goto LABEL_57;
      if ( *(_BYTE *)(a3 + 28) )
      {
        v190 = *(void ***)(a3 + 8);
        v191 = (__int64 **)&v190[*(unsigned int *)(a3 + 20)];
        if ( v190 != (void **)v191 )
        {
          v192 = *(__int64 ***)(a3 + 8);
          while ( *v192 != &qword_4F82400 )
          {
            if ( v191 == ++v192 )
              goto LABEL_337;
          }
          goto LABEL_328;
        }
      }
      else
      {
        if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
        {
LABEL_328:
          v206 = 1;
          goto LABEL_58;
        }
        if ( !*(_BYTE *)(a3 + 28) )
        {
          v206 = sub_C8CA60(a3, (__int64)&unk_4FDBCE8) != 0;
LABEL_58:
          v10 = &v216[8 * v217];
          v198 = v216;
          if ( v216 == v10 )
            goto LABEL_11;
          v204 = &v216[8 * v217];
          v199 = (__int64 *)a1;
          while ( 1 )
          {
            v59 = *((_QWORD *)v204 - 1);
            memset(v219, 0, 0x68u);
            v200 = v59;
            v60 = *(unsigned int *)(*v199 + 88);
            v61 = *(_QWORD *)(*v199 + 72);
            if ( !(_DWORD)v60 )
              goto LABEL_139;
            v62 = 1;
            v197 = (unsigned __int64)(((unsigned int)&unk_4FDBCD8 >> 9) ^ ((unsigned int)&unk_4FDBCD8 >> 4)) << 32;
            for ( jj = (v60 - 1)
                     & (((0xBF58476D1CE4E5B9LL * (v197 | ((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4))) >> 31)
                      ^ (484763065 * (v197 | ((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4)))); ; jj = (v60 - 1) & v65 )
            {
              v64 = v61 + 24LL * jj;
              if ( *(_UNKNOWN **)v64 == &unk_4FDBCD8 && v200 == *(_QWORD *)(v64 + 8) )
                break;
              if ( *(_QWORD *)v64 == -4096 && *(_QWORD *)(v64 + 8) == -4096 )
                goto LABEL_139;
              v65 = v62 + jj;
              ++v62;
            }
            if ( v64 != v61 + 24 * v60 )
            {
              v112 = *(_QWORD *)(*(_QWORD *)(v64 + 16) + 24LL);
              if ( v112 )
                break;
            }
LABEL_139:
            if ( !v206 )
            {
              sub_22D08B0(*v199, v200, a3);
              if ( LOBYTE(v219[12]) )
              {
LABEL_146:
                if ( !BYTE4(v219[9]) )
                  _libc_free(v219[7]);
                if ( !BYTE4(v219[3]) )
                  _libc_free(v219[1]);
              }
            }
LABEL_140:
            v204 -= 8;
            if ( v198 == v204 )
            {
              v10 = v216;
              goto LABEL_11;
            }
          }
          v115 = *(_BYTE *)(v112 + 24) & 1;
          if ( *(_DWORD *)(v112 + 24) >> 1 )
          {
            if ( v115 )
            {
              v111 = (__int64 *)(v112 + 32);
              v113 = (__int64 *)(v112 + 64);
            }
            else
            {
              v111 = *(__int64 **)(v112 + 32);
              v112 = 16LL * *(unsigned int *)(v112 + 40);
              v113 = (__int64 *)((char *)v111 + v112);
              if ( v111 == (__int64 *)((char *)v111 + v112) )
                goto LABEL_139;
            }
            do
            {
              if ( *v111 != -4096 && *v111 != -8192 )
                break;
              v111 += 2;
            }
            while ( v113 != v111 );
          }
          else
          {
            if ( v115 )
            {
              v145 = v112 + 32;
              v146 = 32;
            }
            else
            {
              v145 = *(_QWORD *)(v112 + 32);
              v146 = 16LL * *(unsigned int *)(v112 + 40);
            }
            v111 = (__int64 *)(v146 + v145);
            v113 = v111;
          }
          if ( v111 == v113 )
            goto LABEL_139;
          while ( 1 )
          {
            v116 = *v111;
            v117 = *a4;
            v118 = *(_BYTE *)(*a4 + 8) & 1;
            if ( v118 )
            {
              v119 = v117 + 16;
              v120 = 7;
            }
            else
            {
              v124 = *(unsigned int *)(v117 + 24);
              v119 = *(_QWORD *)(v117 + 16);
              if ( !(_DWORD)v124 )
                goto LABEL_211;
              v120 = v124 - 1;
            }
            v61 = v120 & (((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4));
            v121 = v119 + 16 * v61;
            v122 = *(_QWORD *)v121;
            if ( v116 == *(_QWORD *)v121 )
              goto LABEL_163;
            v144 = 1;
            while ( v122 != -4096 )
            {
              v112 = (unsigned int)(v144 + 1);
              v61 = v120 & (unsigned int)(v144 + v61);
              v121 = v119 + 16LL * (unsigned int)v61;
              v122 = *(_QWORD *)v121;
              if ( v116 == *(_QWORD *)v121 )
                goto LABEL_163;
              v144 = v112;
            }
            if ( v118 )
            {
              v140 = 128;
              goto LABEL_212;
            }
            v124 = *(unsigned int *)(v117 + 24);
LABEL_211:
            v140 = 16 * v124;
LABEL_212:
            v121 = v119 + v140;
LABEL_163:
            v123 = 128;
            if ( !v118 )
              v123 = 16LL * *(unsigned int *)(v117 + 24);
            if ( v121 == v119 + v123 )
            {
              v133 = a4[1];
              v134 = *(unsigned int *)(v133 + 24);
              v135 = *(_QWORD *)(v133 + 8);
              if ( (_DWORD)v134 )
              {
                v136 = 1;
                for ( kk = (v134 - 1)
                         & (((0xBF58476D1CE4E5B9LL
                            * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                             | ((unsigned __int64)(((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4)) << 32))) >> 31)
                          ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; kk = (v134 - 1) & v139 )
                {
                  v138 = (_QWORD *)(v135 + 24LL * kk);
                  if ( v116 == *v138 && a2 == v138[1] )
                    break;
                  if ( *v138 == -4096 && v138[1] == -4096 )
                    goto LABEL_218;
                  v139 = v136 + kk;
                  ++v136;
                }
              }
              else
              {
LABEL_218:
                v138 = (_QWORD *)(v135 + 24 * v134);
              }
              v203 = *v111;
              v212[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v138[2] + 24LL)
                                                                                       + 16LL))(
                          *(_QWORD *)(v138[2] + 24LL),
                          a2,
                          a3,
                          a4);
              v211 = v203;
              sub_BBCF50((__int64)&v213, v117, &v211, v212);
              v121 = v215;
            }
            if ( !*(_BYTE *)(v121 + 8) )
            {
              v111 += 2;
              goto LABEL_168;
            }
            if ( !LOBYTE(v219[12]) )
            {
              sub_C8CD80((__int64)v219, (__int64)&v219[4], a3, v61, v112, v116);
              sub_C8CD80((__int64)&v219[6], (__int64)&v219[10], v205, v141, v142, v143);
              LOBYTE(v219[12]) = 1;
            }
            v125 = v111[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v111[1] & 4) != 0 )
            {
              v61 = *(_QWORD *)v125;
              v111 += 2;
              v126 = *(_QWORD *)v125 + 8LL * *(unsigned int *)(v125 + 8);
            }
            else
            {
              v61 = (__int64)(v111 + 1);
              v111 += 2;
              if ( !v125 )
                goto LABEL_168;
              v126 = (__int64)v111;
            }
            if ( v126 != v61 )
            {
              v202 = v113;
              v201 = v111;
              v127 = (__int64 *)v61;
              while ( 1 )
              {
                v128 = *v127;
                if ( BYTE4(v219[3]) )
                {
                  v112 = v219[1];
                  v129 = (__int64 *)(v219[1] + 8LL * HIDWORD(v219[2]));
                  v130 = (__int64 **)v219[1];
                  if ( (__int64 *)v219[1] != v129 )
                  {
                    while ( (__int64 *)v128 != *v130 )
                    {
                      if ( v129 == (__int64 *)++v130 )
                        goto LABEL_192;
                    }
                    --HIDWORD(v219[2]);
                    v129 = *(__int64 **)(v219[1] + 8LL * HIDWORD(v219[2]));
                    *v130 = v129;
                    ++v219[0];
                  }
                }
                else
                {
                  v132 = sub_C8CA60((__int64)v219, v128);
                  if ( v132 )
                  {
                    *v132 = -2;
                    ++LODWORD(v219[3]);
                    ++v219[0];
                  }
                }
LABEL_192:
                if ( !BYTE4(v219[9]) )
                  goto LABEL_199;
                v131 = (_QWORD *)v219[7];
                v129 = (__int64 *)(v219[7] + 8LL * HIDWORD(v219[8]));
                if ( (__int64 *)v219[7] != v129 )
                {
                  while ( v128 != *v131 )
                  {
                    if ( v129 == ++v131 )
                      goto LABEL_200;
                  }
                  goto LABEL_197;
                }
LABEL_200:
                if ( HIDWORD(v219[8]) < LODWORD(v219[8]) )
                {
                  ++HIDWORD(v219[8]);
                  *v129 = v128;
                  ++v219[6];
                }
                else
                {
LABEL_199:
                  sub_C8CC70((__int64)&v219[6], v128, (__int64)v129, v61, v112, v116);
                }
LABEL_197:
                if ( (__int64 *)v126 == ++v127 )
                {
                  v113 = v202;
                  v111 = v201;
                  break;
                }
              }
            }
LABEL_168:
            if ( v111 != v113 )
            {
              while ( *v111 == -8192 || *v111 == -4096 )
              {
                v111 += 2;
                if ( v113 == v111 )
                  goto LABEL_172;
              }
              if ( v113 != v111 )
                continue;
            }
LABEL_172:
            if ( LOBYTE(v219[12]) )
            {
              sub_22D08B0(*v199, v200, (__int64)v219);
              if ( LOBYTE(v219[12]) )
                goto LABEL_146;
              goto LABEL_140;
            }
            goto LABEL_139;
          }
        }
        v190 = *(void ***)(a3 + 8);
        v192 = (__int64 **)&v190[*(unsigned int *)(a3 + 20)];
        if ( v192 != (__int64 **)v190 )
        {
LABEL_337:
          while ( *v190 != &unk_4FDBCE8 )
          {
            if ( v192 == (__int64 **)++v190 )
              goto LABEL_57;
          }
          goto LABEL_328;
        }
      }
LABEL_57:
      v206 = 0;
      goto LABEL_58;
    }
    v155 = 1;
    while ( v74 != (void *)-4096LL )
    {
      v6 = (unsigned int)(v155 + 1);
      v72 = v70 & (v155 + v72);
      v73 = v69 + 16LL * v72;
      v74 = *(void **)v73;
      if ( *(_UNKNOWN **)v73 == &unk_4F8F810 )
        goto LABEL_73;
      v155 = v6;
    }
    if ( v68 )
    {
      v102 = 128;
      goto LABEL_123;
    }
    v71 = *(_DWORD *)(v67 + 24);
LABEL_122:
    v102 = 16LL * v71;
LABEL_123:
    v73 = v69 + v102;
    goto LABEL_73;
  }
  v7 = *(void ***)(a3 + 56);
  v8 = &v7[*(unsigned int *)(a3 + 68)];
  if ( v7 == v8 )
  {
LABEL_14:
    v14 = *(_BYTE *)(a1 + 16);
    if ( !v14 )
      goto LABEL_15;
    v9 = 0;
    goto LABEL_68;
  }
  while ( *v7 != &unk_4FDBCE0 )
  {
    if ( v8 == ++v7 )
      goto LABEL_14;
  }
  v9 = *(_BYTE *)(a1 + 16);
  if ( v9 )
    goto LABEL_68;
LABEL_7:
  v10 = v216;
  v11 = &v216[8 * v217];
  if ( v11 != v216 )
  {
    do
    {
      v12 = *(_QWORD *)v10;
      v10 += 8;
      sub_22D0060(*(_QWORD *)a1, v12, (__int64)"<possibly invalidated loop>", 27);
    }
    while ( v11 != v10 );
    v10 = v216;
  }
  *(_QWORD *)a1 = 0;
  v207 = 1;
LABEL_11:
  if ( v10 != &v218 )
    _libc_free((unsigned __int64)v10);
  return v207;
}
