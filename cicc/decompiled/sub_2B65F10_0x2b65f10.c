// Function: sub_2B65F10
// Address: 0x2b65f10
//
_BOOL8 __fastcall sub_2B65F10(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rdx
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 *v11; // rbx
  unsigned __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // rax
  char **v16; // rcx
  int v17; // r11d
  __int64 v18; // r8
  __int64 v19; // r9
  _BYTE *v20; // rdi
  __int64 v21; // rcx
  _DWORD *v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // r14
  unsigned int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // rcx
  __int64 v30; // r10
  __int64 v31; // rsi
  __int64 v32; // r11
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rsi
  int v36; // r14d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // ebx
  __int64 i; // r15
  int v42; // eax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rcx
  unsigned __int64 *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  _DWORD *v48; // rax
  unsigned __int64 *v49; // rbx
  int v50; // ebx
  __int64 v51; // r13
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // r10
  unsigned __int64 v55; // rax
  int v56; // edx
  _QWORD *v57; // rax
  __int64 v58; // r12
  __int64 v59; // rdx
  unsigned __int64 v60; // r14
  __int64 v61; // r13
  __int64 v62; // rcx
  __int64 v63; // rsi
  int v64; // eax
  int v65; // r12d
  __int64 v66; // r15
  __int64 v67; // rax
  int v68; // eax
  unsigned __int64 v69; // r14
  __int64 v70; // r13
  int v71; // r12d
  __int64 v72; // r15
  __int64 v73; // rax
  int v74; // eax
  unsigned __int64 v75; // r14
  __int64 v76; // r13
  int v77; // r12d
  __int64 v78; // r15
  __int64 v79; // rax
  int v80; // eax
  unsigned __int64 v81; // r14
  __int64 v82; // r13
  int v83; // r12d
  __int64 v84; // r15
  __int64 v85; // rax
  int v86; // eax
  __int64 v87; // rax
  unsigned __int64 *v88; // rax
  _QWORD *v89; // rax
  unsigned __int64 v90; // r15
  __int64 v91; // r13
  __int64 v92; // rcx
  __int64 v93; // rsi
  int v94; // eax
  char v95; // r12
  int v96; // r14d
  __int64 j; // rbx
  int v98; // eax
  unsigned __int64 v99; // r15
  __int64 v100; // r13
  __int64 v101; // rcx
  __int64 v102; // rsi
  char v103; // r12
  int v104; // r14d
  __int64 m; // rbx
  int v106; // eax
  unsigned __int64 v107; // r14
  __int64 v108; // r12
  __int64 v109; // rcx
  __int64 v110; // rsi
  char v111; // r13
  int v112; // r15d
  __int64 k; // rbx
  int v114; // eax
  __int64 v115; // [rsp+30h] [rbp-1A0h]
  char v116; // [rsp+47h] [rbp-189h]
  unsigned int v117; // [rsp+A0h] [rbp-130h]
  _QWORD *v118; // [rsp+A8h] [rbp-128h]
  unsigned __int64 *v119; // [rsp+B0h] [rbp-120h]
  __int64 v120; // [rsp+B0h] [rbp-120h]
  unsigned __int64 *v122; // [rsp+B8h] [rbp-118h]
  unsigned __int64 *v123; // [rsp+C0h] [rbp-110h]
  unsigned __int64 *v124; // [rsp+C0h] [rbp-110h]
  __int64 v125; // [rsp+C0h] [rbp-110h]
  bool v126; // [rsp+C8h] [rbp-108h]
  unsigned __int64 *v127; // [rsp+C8h] [rbp-108h]
  unsigned __int64 v128; // [rsp+D0h] [rbp-100h]
  _QWORD *v129; // [rsp+D0h] [rbp-100h]
  __int64 v130; // [rsp+D0h] [rbp-100h]
  char v131; // [rsp+E0h] [rbp-F0h]
  char v132; // [rsp+E0h] [rbp-F0h]
  char v133; // [rsp+E0h] [rbp-F0h]
  char v134; // [rsp+E0h] [rbp-F0h]
  char v135; // [rsp+E0h] [rbp-F0h]
  __int64 v136; // [rsp+E0h] [rbp-F0h]
  __int64 v137; // [rsp+E8h] [rbp-E8h]
  int v138; // [rsp+E8h] [rbp-E8h]
  unsigned __int64 v139; // [rsp+E8h] [rbp-E8h]
  unsigned __int64 *v140; // [rsp+E8h] [rbp-E8h]
  __int64 v141; // [rsp+E8h] [rbp-E8h]
  int v142[2]; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v143; // [rsp+F8h] [rbp-D8h]
  __int64 v144; // [rsp+100h] [rbp-D0h]
  _QWORD *v145; // [rsp+108h] [rbp-C8h]
  int v146; // [rsp+110h] [rbp-C0h]
  int v147; // [rsp+114h] [rbp-BCh]
  _BYTE *v148; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v149; // [rsp+128h] [rbp-A8h]
  _BYTE v150[32]; // [rsp+130h] [rbp-A0h] BYREF
  unsigned __int64 *v151; // [rsp+150h] [rbp-80h] BYREF
  __int64 v152; // [rsp+158h] [rbp-78h]
  _BYTE v153[112]; // [rsp+160h] [rbp-70h] BYREF

  v6 = *(__int64 **)a1;
  v126 = 0;
  if ( !**(_QWORD **)a1 )
    return v126;
  v126 = a3 > 2 || v6[1] == **(_QWORD **)a1 || v6[1] == 0;
  if ( v126 )
    return 0;
  if ( (unsigned int)qword_500FB68 > *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) )
    return v126;
  if ( (unsigned int)(qword_500FC48 - 1) <= *(_DWORD *)(a1 + 16) )
    return 1;
  v148 = v150;
  v149 = 0x800000000LL;
  v9 = 8 * a3;
  v10 = &a2[a3];
  v128 = v9;
  if ( a2 != v10 )
  {
    v11 = a2;
    v12 = 8;
    v13 = 0;
    while ( 1 )
    {
      v14 = *v11;
      v15 = 4LL * (*(_DWORD *)(*v11 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(*v11 + 7) & 0x40) != 0 )
      {
        v16 = *(char ***)(v14 - 8);
        v14 = (__int64)&v16[v15];
        if ( &v16[v15] == v16 )
          goto LABEL_21;
      }
      else
      {
        v16 = (char **)(v14 - v15 * 8);
        if ( v14 == v14 - v15 * 8 )
        {
LABEL_21:
          v17 = 0;
          goto LABEL_16;
        }
      }
      v17 = 0;
      do
      {
        if ( (unsigned __int8)**v16 > 0x1Cu || (unsigned __int8)sub_2B15E10(*v16, v13, (__int64)v6, (__int64)v16, a5) )
          ++v17;
        v16 += 4;
      }
      while ( (char **)v14 != v16 );
LABEL_16:
      v6 = (__int64 *)(v13 + 1);
      if ( v13 + 1 > v12 )
      {
        v138 = v17;
        sub_C8D5F0((__int64)&v148, v150, (unsigned __int64)v6, 4u, a5, v14);
        v13 = (unsigned int)v149;
        v17 = v138;
      }
      ++v11;
      *(_DWORD *)&v148[4 * v13] = v17;
      v13 = (unsigned int)(v149 + 1);
      LODWORD(v149) = v149 + 1;
      if ( v10 == v11 )
      {
        v6 = *(__int64 **)a1;
        break;
      }
      v12 = HIDWORD(v149);
    }
  }
  if ( sub_2B17690(*v6) || (v116 = sub_2B17690(*(_QWORD *)(*(_QWORD *)a1 + 8LL))) != 0 )
  {
    v20 = v148;
    v21 = (__int64)&v148[4 * (unsigned int)v149];
    if ( (_BYTE *)v21 == v148 )
      goto LABEL_78;
    v22 = v148;
    LODWORD(v23) = 0;
    do
      v23 = (unsigned int)(*v22++ + v23);
    while ( (_DWORD *)v21 != v22 );
    if ( (int)v23 <= 1 )
      goto LABEL_78;
    v116 = 1;
LABEL_29:
    v24 = v128;
    v25 = *a2;
    v151 = (unsigned __int64 *)v153;
    v26 = *(__int64 *)((char *)a2 + v128 - 8);
    v152 = 0x100000000LL;
    v115 = v26;
    v27 = *(_DWORD *)(**(_QWORD **)a1 + 4LL) & 0x7FFFFFF;
    v137 = v27;
    if ( v27 )
    {
      v28 = 0;
      do
      {
        v34 = sub_2B3B600((__int64)&v151, v24, v23, v21, v18, v19);
        if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
          v29 = *(_QWORD *)(v26 - 8);
        else
          v29 = v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
        v30 = *(_QWORD *)(v29 + 32LL * (unsigned int)v28);
        if ( (*(_BYTE *)(v25 + 7) & 0x40) != 0 )
          v31 = *(_QWORD *)(v25 - 8);
        else
          v31 = v25 - 32LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
        v32 = *(_QWORD *)(v31 + 32LL * (unsigned int)v28);
        v33 = *(unsigned int *)(v34 + 8);
        v24 = *(unsigned int *)(v34 + 12);
        v21 = v33;
        if ( v33 >= v24 )
        {
          v18 = v33 + 1;
          if ( v24 < v33 + 1 )
          {
            v24 = v34 + 16;
            v120 = v30;
            v125 = v32;
            v136 = v34;
            sub_C8D5F0(v34, (const void *)(v34 + 16), v33 + 1, 0x10u, v18, v19);
            v34 = v136;
            v30 = v120;
            v32 = v125;
            v33 = *(unsigned int *)(v136 + 8);
          }
          v23 = *(_QWORD *)v34 + 16 * v33;
          *(_QWORD *)v23 = v32;
          *(_QWORD *)(v23 + 8) = v30;
          ++*(_DWORD *)(v34 + 8);
        }
        else
        {
          v23 = *(_QWORD *)v34 + 16 * v33;
          if ( v23 )
          {
            *(_QWORD *)v23 = v32;
            *(_QWORD *)(v23 + 8) = v30;
            LODWORD(v21) = *(_DWORD *)(v34 + 8);
          }
          v21 = (unsigned int)(v21 + 1);
          *(_DWORD *)(v34 + 8) = v21;
        }
        ++v28;
      }
      while ( v137 != v28 );
      v35 = (__int64)v151;
      v123 = v151;
      v118 = *(_QWORD **)(a1 + 8);
      v119 = &v151[8 * (unsigned __int64)(unsigned int)v152];
      if ( v151 != v119 )
      {
        v117 = 0;
        do
        {
          v36 = *((_DWORD *)v123 + 2);
          v139 = *v123;
          v37 = v118[411];
          v38 = v118[418];
          v39 = v118[413];
          v145 = v118;
          v144 = v37;
          *(_QWORD *)v142 = v39;
          v147 = qword_500F9A8;
          v143 = v38;
          v146 = 2;
          if ( v36 )
          {
            v131 = 0;
            v40 = 1;
            for ( i = 0; i != v36; ++i )
            {
              v42 = sub_2B65A50(
                      (__int64)v142,
                      *(_QWORD *)(v139 + 16LL * (int)i),
                      *(_QWORD *)(v139 + 16LL * (int)i + 8),
                      0,
                      0,
                      1,
                      0,
                      0);
              if ( v42 > v40 )
              {
                v131 = 1;
                v40 = v42;
              }
            }
            v117 -= (v131 == 0) - 1;
          }
          v123 += 8;
        }
        while ( v119 != v123 );
        v35 = (__int64)v151;
        v43 = *(_QWORD *)a1;
        v123 = v151;
        v119 = &v151[8 * (unsigned __int64)(unsigned int)v152];
        v44 = (*(_DWORD *)(**(_QWORD **)a1 + 4LL) & 0x7FFFFFFu) >> 1;
        if ( (unsigned int)v44 <= v117 )
          goto LABEL_55;
        if ( (*(_DWORD *)(**(_QWORD **)a1 + 4LL) & 0x7FFFFFFu) > 2 || !v116 )
          goto LABEL_54;
        v49 = &v151[8 * (unsigned __int64)(unsigned int)v152];
        if ( v151 != v119 )
        {
          do
          {
            v49 -= 8;
            if ( (unsigned __int64 *)*v49 != v49 + 2 )
              _libc_free(*v49);
          }
          while ( v49 != (unsigned __int64 *)v35 );
          v43 = *(_QWORD *)a1;
        }
        goto LABEL_84;
      }
      v43 = *(_QWORD *)a1;
      v44 = *(_DWORD *)(**(_QWORD **)a1 + 4LL) & 0x7FFFFFF;
      if ( (*(_DWORD *)(**(_QWORD **)a1 + 4LL) & 0x7FFFFFE) != 0 )
      {
        if ( (unsigned int)v44 > 2 || !v116 )
        {
LABEL_54:
          v126 = 1;
          goto LABEL_55;
        }
LABEL_84:
        LODWORD(v152) = 0;
        v50 = *(_DWORD *)(*(_QWORD *)v43 + 4LL) & 0x7FFFFFF;
        if ( !v50 )
        {
          v88 = v151;
          v127 = v151;
          v122 = v151;
          goto LABEL_127;
        }
        v51 = 0;
        do
        {
          v58 = sub_2B3B600((__int64)&v151, v35, v43, v44, v18, v19);
          if ( (*(_BYTE *)(v115 + 7) & 0x40) != 0 )
            v52 = *(_QWORD *)(v115 - 8);
          else
            v52 = v115 - 32LL * (*(_DWORD *)(v115 + 4) & 0x7FFFFFF);
          v19 = *(_QWORD *)(v52 + 32LL * (((int)v51 + 1) % v50));
          if ( (*(_BYTE *)(v25 + 7) & 0x40) != 0 )
            v53 = *(_QWORD *)(v25 - 8);
          else
            v53 = v25 - 32LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
          v44 = *(unsigned int *)(v58 + 12);
          v54 = *(_QWORD *)(v53 + 32 * v51);
          v55 = *(unsigned int *)(v58 + 8);
          v56 = *(_DWORD *)(v58 + 8);
          if ( v55 >= v44 )
          {
            v43 = v55 + 1;
            if ( v44 < v55 + 1 )
            {
              v35 = v58 + 16;
              v130 = v19;
              v141 = v54;
              sub_C8D5F0(v58, (const void *)(v58 + 16), v43, 0x10u, v18, v19);
              v55 = *(unsigned int *)(v58 + 8);
              v19 = v130;
              v54 = v141;
            }
            v89 = (_QWORD *)(*(_QWORD *)v58 + 16 * v55);
            *v89 = v54;
            v89[1] = v19;
            ++*(_DWORD *)(v58 + 8);
          }
          else
          {
            v57 = (_QWORD *)(*(_QWORD *)v58 + 16 * v55);
            if ( v57 )
            {
              *v57 = v54;
              v57[1] = v19;
              v56 = *(_DWORD *)(v58 + 8);
            }
            v43 = (unsigned int)(v56 + 1);
            *(_DWORD *)(v58 + 8) = v43;
          }
          ++v51;
        }
        while ( v50 > (int)v51 );
        v140 = v151;
        v129 = *(_QWORD **)(a1 + 8);
        v122 = &v151[8 * (unsigned __int64)(unsigned int)v152];
        if ( (unsigned __int8)((__int64)(unsigned int)v152 >> 2) )
        {
          v124 = &v151[32 * (__int64)(char)((__int64)(unsigned int)v152 >> 2)];
          do
          {
            v59 = v129[411];
            v60 = *v140;
            v61 = *((int *)v140 + 2);
            v62 = v129[418];
            v63 = v129[413];
            v64 = qword_500F9A8;
            v144 = v59;
            v143 = v62;
            *(_QWORD *)v142 = v63;
            v145 = v129;
            v146 = 2;
            v147 = qword_500F9A8;
            if ( v61 )
            {
              v132 = 0;
              v65 = 1;
              v66 = 0;
              do
              {
                while ( 1 )
                {
                  v67 = 16LL * (int)v66++;
                  v68 = sub_2B65A50((__int64)v142, *(_QWORD *)(v60 + v67), *(_QWORD *)(v60 + v67 + 8), 0, 0, 1, 0, 0);
                  if ( v68 <= v65 )
                    break;
                  if ( v61 == v66 )
                    goto LABEL_136;
                  v132 = 1;
                  v65 = v68;
                }
              }
              while ( v61 != v66 );
              if ( v132 )
                goto LABEL_136;
              v64 = qword_500F9A8;
              v59 = v129[411];
              v62 = v129[418];
              v63 = v129[413];
            }
            v69 = v140[8];
            v70 = *((int *)v140 + 18);
            *(_QWORD *)v142 = v63;
            v127 = v140 + 8;
            v143 = v62;
            v144 = v59;
            v145 = v129;
            v146 = 2;
            v147 = v64;
            if ( v70 )
            {
              v133 = 0;
              v71 = 1;
              v72 = 0;
              do
              {
                while ( 1 )
                {
                  v73 = 16LL * (int)v72++;
                  v74 = sub_2B65A50((__int64)v142, *(_QWORD *)(v69 + v73), *(_QWORD *)(v69 + v73 + 8), 0, 0, 1, 0, 0);
                  if ( v74 <= v71 )
                    break;
                  if ( v70 == v72 )
                    goto LABEL_137;
                  v133 = 1;
                  v71 = v74;
                }
              }
              while ( v70 != v72 );
              if ( v133 )
                goto LABEL_137;
              v64 = qword_500F9A8;
              v59 = v129[411];
              v62 = v129[418];
              v63 = v129[413];
            }
            v75 = v140[16];
            v76 = *((int *)v140 + 34);
            *(_QWORD *)v142 = v63;
            v127 = v140 + 16;
            v143 = v62;
            v144 = v59;
            v145 = v129;
            v146 = 2;
            v147 = v64;
            if ( v76 )
            {
              v134 = 0;
              v77 = 1;
              v78 = 0;
              do
              {
                while ( 1 )
                {
                  v79 = 16LL * (int)v78++;
                  v80 = sub_2B65A50((__int64)v142, *(_QWORD *)(v75 + v79), *(_QWORD *)(v75 + v79 + 8), 0, 0, 1, 0, 0);
                  if ( v77 >= v80 )
                    break;
                  if ( v76 == v78 )
                    goto LABEL_137;
                  v134 = 1;
                  v77 = v80;
                }
              }
              while ( v76 != v78 );
              if ( v134 )
                goto LABEL_137;
              v64 = qword_500F9A8;
              v59 = v129[411];
              v62 = v129[418];
              v63 = v129[413];
            }
            v81 = v140[24];
            v82 = *((int *)v140 + 50);
            *(_QWORD *)v142 = v63;
            v127 = v140 + 24;
            v143 = v62;
            v144 = v59;
            v145 = v129;
            v146 = 2;
            v147 = v64;
            if ( v82 )
            {
              v135 = 0;
              v83 = 1;
              v84 = 0;
              do
              {
                while ( 1 )
                {
                  v85 = 16LL * (int)v84++;
                  v86 = sub_2B65A50((__int64)v142, *(_QWORD *)(v81 + v85), *(_QWORD *)(v81 + v85 + 8), 0, 0, 1, 0, 0);
                  if ( v83 >= v86 )
                    break;
                  if ( v82 == v84 )
                    goto LABEL_137;
                  v135 = 1;
                  v83 = v86;
                }
              }
              while ( v82 != v84 );
              if ( v135 )
                goto LABEL_137;
            }
            v140 += 32;
          }
          while ( v124 != v140 );
        }
        v87 = (char *)v122 - (char *)v140;
        if ( (char *)v122 - (char *)v140 == 128 )
        {
          v94 = qword_500F9A8;
        }
        else
        {
          if ( v87 != 192 )
          {
            if ( v87 != 64 )
            {
              v127 = v122;
              v88 = v151;
LABEL_127:
              v123 = v88;
              v126 = v127 == v122;
              v119 = &v88[8 * (unsigned __int64)(unsigned int)v152];
              goto LABEL_55;
            }
            v94 = qword_500F9A8;
            goto LABEL_171;
          }
          v90 = *v140;
          v91 = *((int *)v140 + 2);
          v92 = v129[418];
          v93 = v129[413];
          v94 = qword_500F9A8;
          v144 = v129[411];
          v143 = v92;
          *(_QWORD *)v142 = v93;
          v145 = v129;
          v146 = 2;
          v147 = qword_500F9A8;
          if ( v91 )
          {
            v95 = 0;
            v96 = 1;
            for ( j = 0; j != v91; ++j )
            {
              v98 = sub_2B65A50(
                      (__int64)v142,
                      *(_QWORD *)(v90 + 16LL * (int)j),
                      *(_QWORD *)(v90 + 16LL * (int)j + 8),
                      0,
                      0,
                      1,
                      0,
                      0);
              if ( v96 < v98 )
              {
                v96 = v98;
                v95 = 1;
              }
            }
            if ( v95 )
              goto LABEL_136;
            v94 = qword_500F9A8;
          }
          v140 += 8;
        }
        v99 = *v140;
        v100 = *((int *)v140 + 2);
        v101 = v129[418];
        v102 = v129[413];
        v144 = v129[411];
        v143 = v101;
        *(_QWORD *)v142 = v102;
        v145 = v129;
        v146 = 2;
        v147 = v94;
        if ( !v100 )
        {
LABEL_170:
          v140 += 8;
LABEL_171:
          v107 = *v140;
          v108 = *((int *)v140 + 2);
          v109 = v129[418];
          v110 = v129[413];
          v144 = v129[411];
          v143 = v109;
          *(_QWORD *)v142 = v110;
          v145 = v129;
          v146 = 2;
          v147 = v94;
          if ( !v108 )
            goto LABEL_177;
          v111 = 0;
          v112 = 1;
          for ( k = 0; k != v108; ++k )
          {
            v114 = sub_2B65A50(
                     (__int64)v142,
                     *(_QWORD *)(v107 + 16LL * (int)k),
                     *(_QWORD *)(v107 + 16LL * (int)k + 8),
                     0,
                     0,
                     1,
                     0,
                     0);
            if ( v114 > v112 )
            {
              v112 = v114;
              v111 = 1;
            }
          }
          if ( !v111 )
          {
LABEL_177:
            v126 = 1;
            v123 = v151;
            v119 = &v151[8 * (unsigned __int64)(unsigned int)v152];
LABEL_55:
            v45 = v119;
            if ( v123 != v119 )
            {
              do
              {
                v45 -= 8;
                if ( (unsigned __int64 *)*v45 != v45 + 2 )
                  _libc_free(*v45);
              }
              while ( v45 != v123 );
              v119 = v151;
            }
            goto LABEL_60;
          }
          goto LABEL_136;
        }
        v103 = 0;
        v104 = 1;
        for ( m = 0; m != v100; ++m )
        {
          v106 = sub_2B65A50(
                   (__int64)v142,
                   *(_QWORD *)(v99 + 16LL * (int)m),
                   *(_QWORD *)(v99 + 16LL * (int)m + 8),
                   0,
                   0,
                   1,
                   0,
                   0);
          if ( v106 > v104 )
          {
            v104 = v106;
            v103 = 1;
          }
        }
        if ( !v103 )
        {
          v94 = qword_500F9A8;
          goto LABEL_170;
        }
LABEL_136:
        v127 = v140;
LABEL_137:
        v88 = v151;
        goto LABEL_127;
      }
    }
    else
    {
      v119 = (unsigned __int64 *)v153;
    }
LABEL_60:
    if ( v119 != (unsigned __int64 *)v153 )
      _libc_free((unsigned __int64)v119);
    v20 = v148;
    goto LABEL_63;
  }
  v20 = v148;
  v46 = 4LL * (unsigned int)v149;
  v21 = (__int64)&v148[v46];
  v47 = v46 >> 4;
  if ( v47 )
  {
    v23 = (__int64)&v148[16 * v47];
    v48 = v148;
    while ( *v48 <= 1u )
    {
      if ( v48[1] > 1u )
      {
        ++v48;
        goto LABEL_77;
      }
      if ( v48[2] > 1u )
      {
        v48 += 2;
        goto LABEL_77;
      }
      if ( v48[3] > 1u )
      {
        v48 += 3;
        goto LABEL_77;
      }
      v48 += 4;
      if ( (_DWORD *)v23 == v48 )
        goto LABEL_139;
    }
    goto LABEL_77;
  }
  v48 = v148;
LABEL_139:
  v23 = v21 - (_QWORD)v48;
  if ( v21 - (_QWORD)v48 == 8 )
    goto LABEL_153;
  if ( v23 == 12 )
  {
    if ( *v48 > 1u )
      goto LABEL_77;
    ++v48;
LABEL_153:
    if ( *v48 > 1u )
      goto LABEL_77;
    ++v48;
    goto LABEL_142;
  }
  if ( v23 != 4 )
    goto LABEL_78;
LABEL_142:
  if ( *v48 <= 1u )
    goto LABEL_78;
LABEL_77:
  if ( (_DWORD *)v21 != v48 )
    goto LABEL_29;
LABEL_78:
  v126 = 1;
LABEL_63:
  if ( v20 != v150 )
    _libc_free((unsigned __int64)v20);
  return v126;
}
