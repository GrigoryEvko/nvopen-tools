// Function: sub_316C3A0
// Address: 0x316c3a0
//
void __fastcall sub_316C3A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rdx
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r12
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rbx
  _BYTE *v19; // r12
  __int64 v20; // r15
  int v21; // r10d
  unsigned int v22; // ecx
  __int64 *v23; // rax
  _BYTE *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // r12
  __int64 v35; // r13
  __int64 **v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 *v42; // r12
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 *v45; // r13
  __int64 *v46; // rsi
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 *v55; // r14
  __int64 v56; // r12
  __int64 v57; // rax
  __int64 v58; // r12
  __int64 v59; // r12
  __int64 v60; // r13
  __int64 v61; // rax
  unsigned int v62; // ecx
  __int64 v63; // rsi
  __int64 v64; // rax
  bool v65; // al
  __int64 *v66; // rdi
  unsigned __int64 v67; // rsi
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  unsigned __int64 *v75; // rbx
  __int64 v76; // rdx
  int v77; // edx
  __int64 v78; // r14
  int v79; // ecx
  __int64 v80; // rsi
  __int64 v81; // rcx
  int v82; // edi
  __int64 *v83; // rsi
  __int64 *v84; // rbx
  __int64 v85; // rdi
  __int64 v86; // r12
  __int64 v87; // r13
  __int64 v88; // rax
  unsigned int v89; // ecx
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // r12
  __int64 v93; // r13
  __int64 v94; // rax
  unsigned int v95; // ecx
  __int64 v96; // rsi
  __int64 v97; // rax
  __int64 v98; // r12
  __int64 v99; // r13
  __int64 v100; // rax
  unsigned int v101; // ecx
  __int64 v102; // rsi
  __int64 v103; // rax
  __int64 v104; // r12
  __int64 v105; // rax
  unsigned int v106; // esi
  __int64 v107; // rdi
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // rax
  unsigned int v111; // esi
  __int64 v112; // rdi
  __int64 v113; // rax
  __int64 v114; // r12
  __int64 v115; // rax
  unsigned int v116; // esi
  __int64 v117; // rdi
  __int64 v118; // rax
  __int64 **v119; // rcx
  __int64 *v120; // rdx
  __int64 **v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rdi
  __int64 *v124; // rdi
  char v125; // [rsp+27h] [rbp-719h]
  __int64 *v126; // [rsp+30h] [rbp-710h]
  unsigned __int64 *v128; // [rsp+38h] [rbp-708h]
  __int64 *v130; // [rsp+40h] [rbp-700h]
  unsigned int *v131; // [rsp+48h] [rbp-6F8h]
  __int64 *v133; // [rsp+50h] [rbp-6F0h]
  __int64 v134; // [rsp+50h] [rbp-6F0h]
  __int64 v135; // [rsp+50h] [rbp-6F0h]
  __int64 v136; // [rsp+50h] [rbp-6F0h]
  __int64 *v137; // [rsp+58h] [rbp-6E8h]
  __int64 v138; // [rsp+58h] [rbp-6E8h]
  __int64 *v139; // [rsp+58h] [rbp-6E8h]
  __int64 *v140; // [rsp+68h] [rbp-6D8h] BYREF
  __int64 v141[3]; // [rsp+70h] [rbp-6D0h] BYREF
  char v142; // [rsp+88h] [rbp-6B8h]
  __int64 v143; // [rsp+90h] [rbp-6B0h] BYREF
  __int64 **v144; // [rsp+98h] [rbp-6A8h]
  __int64 v145; // [rsp+A0h] [rbp-6A0h]
  unsigned int v146; // [rsp+A8h] [rbp-698h]
  _QWORD *v147; // [rsp+B0h] [rbp-690h] BYREF
  __int64 v148; // [rsp+B8h] [rbp-688h]
  _QWORD v149[4]; // [rsp+C0h] [rbp-680h] BYREF
  unsigned __int64 *v150; // [rsp+E0h] [rbp-660h] BYREF
  __int64 v151; // [rsp+E8h] [rbp-658h]
  _BYTE v152[192]; // [rsp+F0h] [rbp-650h] BYREF
  unsigned __int64 v153[2]; // [rsp+1B0h] [rbp-590h] BYREF
  _QWORD v154[176]; // [rsp+1C0h] [rbp-580h] BYREF

  v7 = (unsigned __int64 *)&v150;
  v150 = (unsigned __int64 *)v152;
  v8 = 0x400000000LL;
  v125 = a5;
  v151 = 0x400000000LL;
  v141[0] = (__int64)&v150;
  v141[1] = (__int64)a1;
  v141[2] = a3;
  v142 = 1;
  if ( !(_BYTE)a5 )
  {
    v9 = *(_QWORD *)(a3 + 8);
    v10 = *(__int64 **)v9;
    v11 = *(unsigned int *)(v9 + 8);
    if ( v10 != &v10[6 * v11] )
    {
      v137 = &v10[6 * v11];
      v12 = v10;
      do
      {
        v13 = *v12;
        v7 = v153;
        v153[0] = (unsigned __int64)v154;
        v153[1] = 0x400000001LL;
        v154[0] = v13;
        sub_315E290((__int64)&v150, (__int64)v153, v13, v8, a5, a6);
        if ( (_QWORD *)v153[0] != v154 )
          _libc_free(v153[0]);
        v12 += 6;
      }
      while ( v137 != v12 );
    }
    sub_3163490(v141, (__int64)v7, a3, v8, a5, a6);
    v14 = v150;
    v15 = &v150[6 * (unsigned int)v151];
    if ( v150 != v15 )
    {
      do
      {
        v15 -= 6;
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
          _libc_free(*v15);
      }
      while ( v14 != v15 );
LABEL_12:
      v15 = v150;
      goto LABEL_13;
    }
    goto LABEL_13;
  }
  v16 = *(_QWORD *)(a4 + 120);
  v17 = *(unsigned int *)(a4 + 128);
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v138 = v16 + 8 * v17;
  if ( v138 != v16 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)v16 + 16LL);
      if ( v18 )
        break;
LABEL_30:
      v16 += 8;
      if ( v138 == v16 )
        goto LABEL_31;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v19 = *(_BYTE **)(v18 + 24);
        if ( *v19 == 32 )
          break;
LABEL_19:
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_30;
      }
      v20 = *(_QWORD *)(*((_QWORD *)v19 - 1) + 32LL);
      if ( !v146 )
        break;
      a5 = v146 - 1;
      v21 = 1;
      a6 = 0;
      v22 = a5 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = (__int64 *)&v144[2 * v22];
      v24 = (_BYTE *)*v23;
      if ( v19 == (_BYTE *)*v23 )
        goto LABEL_23;
      while ( v24 != (_BYTE *)-4096LL )
      {
        if ( v24 == (_BYTE *)-8192LL && !a6 )
          a6 = (__int64)v23;
        v22 = a5 & (v21 + v22);
        v23 = (__int64 *)&v144[2 * v22];
        v24 = (_BYTE *)*v23;
        if ( v19 == (_BYTE *)*v23 )
          goto LABEL_23;
        ++v21;
      }
      if ( a6 )
        v23 = (__int64 *)a6;
      ++v143;
      v77 = v145 + 1;
      if ( 4 * ((int)v145 + 1) >= 3 * v146 )
        goto LABEL_94;
      if ( v146 - HIDWORD(v145) - v77 <= v146 >> 3 )
      {
        sub_31638E0((__int64)&v143, v146);
        if ( !v146 )
          goto LABEL_188;
        a5 = (__int64)v144;
        a6 = 0;
        LODWORD(v78) = (v146 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v77 = v145 + 1;
        v79 = 1;
        v23 = (__int64 *)&v144[2 * (unsigned int)v78];
        v80 = *v23;
        if ( v19 != (_BYTE *)*v23 )
        {
          while ( v80 != -4096 )
          {
            if ( !a6 && v80 == -8192 )
              a6 = (__int64)v23;
            v78 = (v146 - 1) & ((_DWORD)v78 + v79);
            v23 = (__int64 *)&v144[2 * v78];
            v80 = *v23;
            if ( v19 == (_BYTE *)*v23 )
              goto LABEL_84;
            ++v79;
          }
          if ( a6 )
            v23 = (__int64 *)a6;
        }
      }
LABEL_84:
      LODWORD(v145) = v77;
      if ( *v23 != -4096 )
        --HIDWORD(v145);
      *v23 = (__int64)v19;
      v23[1] = 0;
LABEL_23:
      v23[1] = v20;
      v25 = *((_QWORD *)v19 - 1);
      v26 = *(_QWORD *)(v25 + 96);
      if ( !v26 )
      {
        if ( *(_QWORD *)(v25 + 32) )
        {
          v76 = *(_QWORD *)(v25 + 40);
          **(_QWORD **)(v25 + 48) = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 16) = *(_QWORD *)(v25 + 48);
          *(_QWORD *)(v25 + 32) = 0;
        }
        goto LABEL_19;
      }
      if ( *(_QWORD *)(v25 + 32) )
      {
        v27 = *(_QWORD *)(v25 + 40);
        **(_QWORD **)(v25 + 48) = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = *(_QWORD *)(v25 + 48);
      }
      *(_QWORD *)(v25 + 32) = v26;
      v28 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v25 + 40) = v28;
      if ( v28 )
      {
        a5 = v25 + 40;
        *(_QWORD *)(v28 + 16) = v25 + 40;
      }
      *(_QWORD *)(v25 + 48) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v25 + 32;
      v18 = *(_QWORD *)(v18 + 8);
      if ( !v18 )
        goto LABEL_30;
    }
    ++v143;
LABEL_94:
    sub_31638E0((__int64)&v143, 2 * v146);
    if ( !v146 )
    {
LABEL_188:
      LODWORD(v145) = v145 + 1;
      BUG();
    }
    a5 = v146 - 1;
    v77 = v145 + 1;
    LODWORD(v81) = a5 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v23 = (__int64 *)&v144[2 * (unsigned int)v81];
    a6 = *v23;
    if ( v19 != (_BYTE *)*v23 )
    {
      v82 = 1;
      v83 = 0;
      while ( a6 != -4096 )
      {
        if ( a6 == -8192 && !v83 )
          v83 = v23;
        v81 = (unsigned int)a5 & ((_DWORD)v81 + v82);
        v23 = (__int64 *)&v144[2 * v81];
        a6 = *v23;
        if ( v19 == (_BYTE *)*v23 )
          goto LABEL_84;
        ++v82;
      }
      if ( v83 )
        v23 = v83;
    }
    goto LABEL_84;
  }
LABEL_31:
  v147 = v149;
  v148 = 0x400000000LL;
  v29 = *(_QWORD *)(a3 + 8);
  v30 = *(unsigned int *)(v29 + 8);
  if ( (unsigned int)v30 > 4 )
  {
    sub_C8D5F0((__int64)&v147, v149, v30, 8u, a5, a6);
    v31 = (unsigned int)v148;
    v29 = *(_QWORD *)(a3 + 8);
    v32 = (unsigned int)v148;
    v30 = *(unsigned int *)(v29 + 8);
  }
  else
  {
    v31 = 0;
    v32 = 0;
  }
  v33 = *(__int64 **)v29;
  v34 = *(_QWORD *)v29 + 48 * v30;
  if ( *(_QWORD *)v29 != v34 )
  {
    do
    {
      v35 = *v33;
      if ( v32 + 1 > (unsigned __int64)HIDWORD(v148) )
      {
        sub_C8D5F0((__int64)&v147, v149, v32 + 1, 8u, a5, v31);
        v32 = (unsigned int)v148;
      }
      v33 += 6;
      v147[v32] = v35;
      v32 = (unsigned int)(v148 + 1);
      LODWORD(v148) = v148 + 1;
    }
    while ( (__int64 *)v34 != v33 );
    v31 = (unsigned int)v32;
  }
  v36 = (__int64 **)a2;
  sub_1054520((__int64)v153, a2, (__int64)v147, v31, 0, v31);
  if ( v147 != v149 )
    _libc_free((unsigned __int64)v147);
  sub_1051870((__int64)v153, a2, v37, v38, v39, v40);
  v140 = a1;
  v41 = *(_QWORD *)(a3 + 8);
  v42 = *(__int64 **)v41;
  v43 = *(unsigned int *)(v41 + 8);
  v44 = 48 * v43;
  v45 = &v42[6 * v43];
  if ( v42 != v45 )
  {
    v46 = &v42[6 * v43];
    _BitScanReverse64(&v47, 0xAAAAAAAAAAAAAAABLL * (v44 >> 4));
    sub_316BB60(v42, v46, 2LL * (int)(63 - (v47 ^ 0x3F)), &v140);
    if ( (unsigned __int64)v44 > 0x300 )
    {
      v84 = v42 + 96;
      v36 = (__int64 **)(v42 + 96);
      sub_315EDB0((__int64)v42, v42 + 96, &v140);
      if ( v45 != v42 + 96 )
      {
        do
        {
          v85 = (__int64)v84;
          v36 = &v140;
          v84 += 6;
          sub_315EB70(v85, &v140);
        }
        while ( v45 != v84 );
      }
    }
    else
    {
      v36 = (__int64 **)v45;
      sub_315EDB0((__int64)v42, v45, &v140);
    }
    v51 = *(_QWORD *)(a3 + 8);
    v52 = *(__int64 **)v51;
    v53 = 48LL * *(unsigned int *)(v51 + 8);
    v126 = (__int64 *)((char *)v52 + v53);
    if ( v52 != (__int64 *)((char *)v52 + v53) )
    {
      v130 = v52;
      while ( 1 )
      {
        v54 = *v130;
        v128 = &v150[6 * (unsigned int)v151];
        if ( v128 != v150 )
          break;
LABEL_133:
        v149[0] = v54;
        v36 = &v147;
        v148 = 0x400000001LL;
        v147 = v149;
        sub_315E290((__int64)&v150, (__int64)&v147, v53, v48, v49, v50);
        if ( v147 != v149 )
          _libc_free((unsigned __int64)v147);
LABEL_61:
        v130 += 6;
        if ( v126 == v130 )
          goto LABEL_62;
      }
      v131 = (unsigned int *)v150;
      while ( 1 )
      {
        v55 = *(__int64 **)v131;
        v56 = 8LL * v131[2];
        v139 = (__int64 *)(*(_QWORD *)v131 + v56);
        v57 = v56 >> 3;
        v58 = v56 >> 5;
        if ( v58 )
        {
          v133 = &v55[4 * v58];
          while ( 1 )
          {
            v59 = *v55;
            v60 = sub_104D250((__int64)v153, v54);
            v61 = sub_104D250((__int64)v153, v59);
            v62 = *(_DWORD *)(v61 + 8);
            if ( *(_DWORD *)(v60 + 8) <= v62 )
              v62 = *(_DWORD *)(v60 + 8);
            if ( v62 )
              break;
LABEL_105:
            v86 = v55[1];
            v87 = sub_104D250((__int64)v153, v54);
            v88 = sub_104D250((__int64)v153, v86);
            v89 = *(_DWORD *)(v88 + 8);
            if ( *(_DWORD *)(v87 + 8) <= v89 )
              v89 = *(_DWORD *)(v87 + 8);
            if ( v89 )
            {
              v90 = *(_QWORD *)v88;
              v91 = 0;
              while ( (*(_QWORD *)(v90 + 8 * v91) & *(_QWORD *)(*(_QWORD *)v87 + 8 * v91)) == 0 )
              {
                if ( v89 == ++v91 )
                  goto LABEL_112;
              }
              v65 = v139 == v55 + 1;
              goto LABEL_56;
            }
LABEL_112:
            v92 = v55[2];
            v93 = sub_104D250((__int64)v153, v54);
            v94 = sub_104D250((__int64)v153, v92);
            v95 = *(_DWORD *)(v94 + 8);
            if ( *(_DWORD *)(v93 + 8) <= v95 )
              v95 = *(_DWORD *)(v93 + 8);
            if ( v95 )
            {
              v96 = *(_QWORD *)v94;
              v97 = 0;
              while ( (*(_QWORD *)(v96 + 8 * v97) & *(_QWORD *)(*(_QWORD *)v93 + 8 * v97)) == 0 )
              {
                if ( v95 == ++v97 )
                  goto LABEL_119;
              }
              v65 = v139 == v55 + 2;
              goto LABEL_56;
            }
LABEL_119:
            v98 = v55[3];
            v99 = sub_104D250((__int64)v153, v54);
            v100 = sub_104D250((__int64)v153, v98);
            v101 = *(_DWORD *)(v100 + 8);
            if ( *(_DWORD *)(v99 + 8) <= v101 )
              v101 = *(_DWORD *)(v99 + 8);
            if ( v101 )
            {
              v102 = *(_QWORD *)v100;
              v103 = 0;
              while ( (*(_QWORD *)(v102 + 8 * v103) & *(_QWORD *)(*(_QWORD *)v99 + 8 * v103)) == 0 )
              {
                if ( v101 == ++v103 )
                  goto LABEL_126;
              }
              v65 = v139 == v55 + 3;
              goto LABEL_56;
            }
LABEL_126:
            v55 += 4;
            if ( v133 == v55 )
            {
              v57 = v139 - v55;
              goto LABEL_128;
            }
          }
          v63 = *(_QWORD *)v61;
          v64 = 0;
          while ( (*(_QWORD *)(v63 + 8 * v64) & *(_QWORD *)(*(_QWORD *)v60 + 8 * v64)) == 0 )
          {
            if ( v62 == ++v64 )
              goto LABEL_105;
          }
          goto LABEL_55;
        }
LABEL_128:
        if ( v57 != 2 )
        {
          if ( v57 != 3 )
          {
            if ( v57 != 1 )
              goto LABEL_131;
            goto LABEL_136;
          }
          v136 = *v55;
          v114 = sub_104D250((__int64)v153, v54);
          v115 = sub_104D250((__int64)v153, v136);
          v116 = *(_DWORD *)(v115 + 8);
          if ( *(_DWORD *)(v114 + 8) <= v116 )
            v116 = *(_DWORD *)(v114 + 8);
          if ( v116 )
          {
            v117 = *(_QWORD *)v115;
            v49 = *(_QWORD *)v114;
            v118 = 0;
            while ( (*(_QWORD *)(v117 + 8 * v118) & *(_QWORD *)(v49 + 8 * v118)) == 0 )
            {
              if ( v116 == ++v118 )
                goto LABEL_143;
            }
            goto LABEL_55;
          }
LABEL_143:
          ++v55;
        }
        v135 = *v55;
        v109 = sub_104D250((__int64)v153, v54);
        v110 = sub_104D250((__int64)v153, v135);
        v111 = *(_DWORD *)(v110 + 8);
        if ( *(_DWORD *)(v109 + 8) <= v111 )
          v111 = *(_DWORD *)(v109 + 8);
        if ( v111 )
        {
          v112 = *(_QWORD *)v110;
          v49 = *(_QWORD *)v109;
          v113 = 0;
          while ( (*(_QWORD *)(v112 + 8 * v113) & *(_QWORD *)(v49 + 8 * v113)) == 0 )
          {
            if ( v111 == ++v113 )
              goto LABEL_135;
          }
          goto LABEL_55;
        }
LABEL_135:
        ++v55;
LABEL_136:
        v134 = *v55;
        v104 = sub_104D250((__int64)v153, v54);
        v105 = sub_104D250((__int64)v153, v134);
        v106 = *(_DWORD *)(v105 + 8);
        if ( *(_DWORD *)(v104 + 8) <= v106 )
          v106 = *(_DWORD *)(v104 + 8);
        if ( !v106 )
        {
LABEL_131:
          v65 = v125;
          goto LABEL_56;
        }
        v107 = *(_QWORD *)v105;
        v49 = *(_QWORD *)v104;
        v108 = 0;
        while ( (*(_QWORD *)(v107 + 8 * v108) & *(_QWORD *)(v49 + 8 * v108)) == 0 )
        {
          if ( v106 == ++v108 )
            goto LABEL_131;
        }
LABEL_55:
        v65 = v139 == v55;
LABEL_56:
        v66 = *(__int64 **)v131;
        _BitScanReverse64(&v67, 1LL << *(_WORD *)(**(_QWORD **)v131 + 2LL));
        LODWORD(v67) = v67 ^ 0x3F;
        _BitScanReverse64(&v68, 1LL << *(_WORD *)(v54 + 2));
        v48 = (unsigned int)v67;
        v53 = ~(-1LL << (63 - ((unsigned __int8)v68 ^ 0x3Fu)));
        if ( ((0x8000000000000000LL >> v67) & v53) == 0 && v65 )
        {
          v36 = (__int64 **)v131;
          v69 = v131[2];
          v48 = v131[3];
          v53 = v69 + 1;
          if ( v69 + 1 > v48 )
          {
            v36 = (__int64 **)(v131 + 4);
            sub_C8D5F0((__int64)v131, v131 + 4, v53, 8u, v49, v50);
            v66 = *(__int64 **)v131;
            v69 = v131[2];
          }
          v66[v69] = v54;
          ++v131[2];
          goto LABEL_61;
        }
        v131 += 12;
        if ( v128 == (unsigned __int64 *)v131 )
          goto LABEL_133;
      }
    }
  }
LABEL_62:
  if ( (_DWORD)v145 )
  {
    v36 = v144;
    v119 = &v144[2 * v146];
    if ( v144 != v119 )
    {
      while ( 1 )
      {
        v120 = *v36;
        v121 = v36;
        if ( *v36 != (__int64 *)-4096LL && v120 != (__int64 *)-8192LL )
          break;
        v36 += 2;
        if ( v119 == v36 )
          goto LABEL_63;
      }
      while ( v121 != v119 )
      {
        v122 = *(v120 - 1);
        v36 = (__int64 **)v121[1];
        if ( *(_QWORD *)(v122 + 32) )
        {
          v123 = *(_QWORD *)(v122 + 40);
          **(_QWORD **)(v122 + 48) = v123;
          if ( v123 )
            *(_QWORD *)(v123 + 16) = *(_QWORD *)(v122 + 48);
        }
        *(_QWORD *)(v122 + 32) = v36;
        if ( v36 )
        {
          v124 = v36[2];
          *(_QWORD *)(v122 + 40) = v124;
          if ( v124 )
            v124[2] = v122 + 40;
          *(_QWORD *)(v122 + 48) = v36 + 2;
          v36[2] = (__int64 *)(v122 + 32);
        }
        v121 += 2;
        if ( v121 == v119 )
          break;
        while ( 1 )
        {
          v120 = *v121;
          if ( *v121 != (__int64 *)-8192LL && v120 != (__int64 *)-4096LL )
            break;
          v121 += 2;
          if ( v119 == v121 )
            goto LABEL_63;
        }
      }
    }
  }
LABEL_63:
  sub_D896C0((__int64)v153, (__int64)v36);
  v70 = 16LL * v146;
  sub_C7D6A0((__int64)v144, v70, 8);
  if ( v142 )
    sub_3163490(v141, v70, v71, v72, v73, v74);
  v75 = v150;
  v15 = &v150[6 * (unsigned int)v151];
  if ( v150 != v15 )
  {
    do
    {
      v15 -= 6;
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        _libc_free(*v15);
    }
    while ( v75 != v15 );
    goto LABEL_12;
  }
LABEL_13:
  if ( v15 != (unsigned __int64 *)v152 )
    _libc_free((unsigned __int64)v15);
}
