// Function: sub_320E0C0
// Address: 0x320e0c0
//
void __fastcall sub_320E0C0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v5; // r15
  __int64 v7; // rdx
  unsigned __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rbx
  unsigned int v12; // esi
  __int64 *v13; // rax
  __int64 v14; // r9
  char *v15; // r11
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned int v18; // edx
  char **v19; // rcx
  char *v20; // r9
  unsigned __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdi
  char *v24; // r9
  size_t v25; // r10
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rbx
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // r14
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r8
  __int64 v37; // r9
  unsigned __int64 v38; // rcx
  int v39; // esi
  unsigned __int64 v40; // rax
  __int64 v41; // rcx
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // r11
  unsigned __int64 v45; // rcx
  __m128i v46; // xmm1
  __int64 v47; // rax
  size_t v48; // r10
  unsigned __int64 v49; // r8
  unsigned __int64 *v50; // rdi
  __int64 v51; // r15
  unsigned __int64 *v52; // rax
  _QWORD *v53; // rsi
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdi
  __int64 v56; // r15
  unsigned __int64 v57; // r14
  unsigned __int64 v58; // r12
  unsigned __int64 v59; // rbx
  unsigned __int64 v60; // rdi
  _BYTE *v61; // r15
  unsigned __int64 v62; // r14
  unsigned __int64 v63; // r12
  unsigned __int64 v64; // rbx
  unsigned __int64 v65; // rdi
  _BYTE *v66; // r15
  unsigned __int64 v67; // r14
  unsigned __int64 v68; // r12
  unsigned __int64 v69; // rbx
  unsigned __int64 v70; // rdi
  _QWORD *v71; // r15
  __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rax
  unsigned __int8 *v76; // rdi
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rax
  unsigned __int64 v82; // r15
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  bool v85; // cc
  unsigned __int64 v86; // rdi
  unsigned __int64 v87; // rdi
  unsigned __int64 v88; // rdi
  int v89; // eax
  int v90; // r10d
  int v91; // ecx
  int v92; // r10d
  __int64 v93; // rdx
  unsigned __int64 v94; // rsi
  char v95; // al
  unsigned __int64 v96; // rdx
  size_t v97; // r10
  unsigned __int64 v98; // rcx
  _QWORD *v99; // r8
  _QWORD *v100; // rax
  unsigned __int64 v101; // rdx
  unsigned __int64 v102; // rax
  void *v103; // rax
  _QWORD *v104; // rax
  _QWORD *v105; // rdi
  unsigned __int64 v106; // r11
  _QWORD *v107; // rsi
  unsigned __int64 v108; // rdx
  _QWORD **v109; // rax
  unsigned __int64 v110; // rdi
  unsigned __int64 v111; // rcx
  __int64 v112; // [rsp+8h] [rbp-238h]
  unsigned __int64 v113; // [rsp+10h] [rbp-230h]
  _QWORD *v114; // [rsp+18h] [rbp-228h]
  unsigned __int64 v115; // [rsp+18h] [rbp-228h]
  size_t v116; // [rsp+18h] [rbp-228h]
  unsigned __int64 v117; // [rsp+18h] [rbp-228h]
  __int64 v118; // [rsp+40h] [rbp-200h]
  __int64 v119; // [rsp+40h] [rbp-200h]
  _BYTE *v120; // [rsp+48h] [rbp-1F8h]
  unsigned __int64 v121; // [rsp+50h] [rbp-1F0h]
  size_t n; // [rsp+58h] [rbp-1E8h]
  _QWORD *v123; // [rsp+60h] [rbp-1E0h]
  __int64 v124; // [rsp+68h] [rbp-1D8h]
  char *v125; // [rsp+70h] [rbp-1D0h]
  __int64 v126; // [rsp+78h] [rbp-1C8h]
  _BYTE *v127; // [rsp+80h] [rbp-1C0h]
  __int64 v128; // [rsp+88h] [rbp-1B8h]
  _BYTE v129[88]; // [rsp+90h] [rbp-1B0h] BYREF
  _BYTE *v130; // [rsp+E8h] [rbp-158h]
  __int64 v131; // [rsp+F0h] [rbp-150h]
  _BYTE v132[16]; // [rsp+F8h] [rbp-148h] BYREF
  char *v133; // [rsp+108h] [rbp-138h]
  __int64 v134; // [rsp+110h] [rbp-130h]
  char v135; // [rsp+118h] [rbp-128h] BYREF
  __int128 v136; // [rsp+120h] [rbp-120h]
  __int128 v137; // [rsp+130h] [rbp-110h]
  char *v138; // [rsp+140h] [rbp-100h]
  _BYTE *v139; // [rsp+148h] [rbp-F8h] BYREF
  __int64 v140; // [rsp+150h] [rbp-F0h]
  _BYTE v141[88]; // [rsp+158h] [rbp-E8h] BYREF
  char *v142; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v143; // [rsp+1B8h] [rbp-88h]
  _BYTE v144[16]; // [rsp+1C0h] [rbp-80h] BYREF
  char *v145; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 v146; // [rsp+1D8h] [rbp-68h]
  char v147; // [rsp+1E0h] [rbp-60h] BYREF
  __int64 v148; // [rsp+1E8h] [rbp-58h]
  __int64 v149; // [rsp+1F0h] [rbp-50h]
  __m128i v150[4]; // [rsp+1F8h] [rbp-48h] BYREF

  v5 = *(_BYTE *)(a2 + 24);
  if ( v5 )
    return;
  v7 = *(unsigned int *)(a1 + 864);
  v9 = a1;
  v10 = *(_QWORD *)(a1 + 848);
  v11 = a2;
  if ( (_DWORD)v7 )
  {
    v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v10 + 112LL * v12);
    v14 = *v13;
    if ( v11 == *v13 )
    {
LABEL_5:
      if ( v13 != (__int64 *)(v10 + 112 * v7) )
      {
        v126 = (__int64)(v13 + 1);
        goto LABEL_7;
      }
    }
    else
    {
      v89 = 1;
      while ( v14 != -4096 )
      {
        v90 = v89 + 1;
        v12 = (v7 - 1) & (v89 + v12);
        v13 = (__int64 *)(v10 + 112LL * v12);
        v14 = *v13;
        if ( v11 == *v13 )
          goto LABEL_5;
        v89 = v90;
      }
    }
  }
  v126 = 0;
LABEL_7:
  v15 = *(char **)(v11 + 8);
  v16 = *(unsigned int *)(a1 + 896);
  v17 = *(_QWORD *)(a1 + 880);
  v125 = v15;
  if ( (_DWORD)v16 )
  {
    v18 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v19 = (char **)(v17 + 16LL * v18);
    v20 = *v19;
    if ( v15 == *v19 )
    {
LABEL_9:
      if ( v19 != (char **)(v17 + 16 * v16) )
      {
        v124 = (__int64)v19[1];
        v21 = v124 | v126;
        goto LABEL_11;
      }
    }
    else
    {
      v91 = 1;
      while ( v20 != (char *)-4096LL )
      {
        v92 = v91 + 1;
        v18 = (v16 - 1) & (v91 + v18);
        v19 = (char **)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v125 == *v19 )
          goto LABEL_9;
        v91 = v92;
      }
    }
  }
  v124 = 0;
  v21 = v126;
LABEL_11:
  if ( *v125 != 19 )
  {
    v125 = 0;
LABEL_27:
    v5 = 1;
    if ( *(_DWORD *)(v11 + 88) != 1 )
      goto LABEL_14;
    goto LABEL_28;
  }
  if ( !v21 )
    goto LABEL_27;
  if ( *(_DWORD *)(v11 + 88) != 1 )
  {
LABEL_14:
    if ( v126 )
    {
      v125 = (char *)a5;
      sub_31FE2D0(a4, *(_QWORD *)v126, *(_QWORD *)v126 + 88LL * *(unsigned int *)(v126 + 8));
      a5 = (__int64)v125;
    }
    if ( v124 )
    {
      v22 = *(unsigned int *)(v124 + 8);
      v23 = *(unsigned int *)(a5 + 8);
      v24 = *(char **)v124;
      v25 = 16 * v22;
      if ( v22 + v23 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v124 = 16 * v22;
        v125 = v24;
        v126 = a5;
        sub_C8D5F0(a5, (const void *)(a5 + 16), v22 + v23, 0x10u, a5, (__int64)v24);
        a5 = v126;
        v25 = 16 * v22;
        v24 = v125;
        v23 = *(unsigned int *)(v126 + 8);
      }
      if ( v25 )
      {
        v23 = *(_QWORD *)a5 + 16 * v23;
        v126 = a5;
        memcpy((void *)v23, v24, v25);
        a5 = v126;
        LODWORD(v23) = *(_DWORD *)(v126 + 8);
      }
      *(_DWORD *)(a5 + 8) = v22 + v23;
    }
    v26 = *(__int64 **)(v11 + 32);
    v27 = *(unsigned int *)(v11 + 40);
    if ( v26 != &v26[v27] )
    {
      v126 = (__int64)&v26[v27];
      v28 = v26;
      v29 = a5;
      do
      {
        v30 = *v28++;
        sub_320E0C0(v9, v30, a3, a4, v29);
      }
      while ( (__int64 *)v126 != v28 );
    }
    return;
  }
LABEL_28:
  v31 = *(_QWORD *)(v11 + 80);
  v123 = (_QWORD *)a5;
  v32 = sub_3211FB0(a1, *(_QWORD *)(v31 + 8));
  a5 = (__int64)v123;
  LOBYTE(n) = v5 | (v32 == 0);
  if ( (_BYTE)n )
    goto LABEL_14;
  v33 = *(_QWORD **)(a1 + 792);
  v130 = v132;
  v133 = &v135;
  v120 = v129;
  v138 = v125;
  v139 = v141;
  v127 = v129;
  v142 = v144;
  v128 = 0x100000000LL;
  v131 = 0x100000000LL;
  v134 = 0x100000000LL;
  v140 = 0x100000000LL;
  v143 = 0x100000000LL;
  v145 = &v147;
  v146 = 0x100000000LL;
  v148 = 0;
  v149 = 0;
  v136 = 0;
  v137 = 0;
  v150[0] = 0;
  v34 = (_QWORD *)sub_22077B0(0xD0u);
  v123 = v34;
  if ( v34 )
    *v34 = 0;
  v38 = (unsigned __int64)v123;
  v39 = v140;
  v123[1] = v138;
  v113 = v38 + 32;
  *(_QWORD *)(v38 + 16) = v38 + 32;
  *(_QWORD *)(v38 + 24) = 0x100000000LL;
  if ( v39 )
    sub_320B760(v38 + 16, (__int64)&v139, v35, 0x100000000LL, v36, v37);
  v40 = (unsigned __int64)v123;
  v114 = v123 + 17;
  v123[15] = v123 + 17;
  *(_QWORD *)(v40 + 128) = 0x100000000LL;
  v41 = (unsigned int)v143;
  if ( (_DWORD)v143 )
    sub_31F4690(v40 + 120, &v142, v35, (unsigned int)v143, v36, v37);
  v42 = (unsigned __int64)v123;
  v123[20] = 0x100000000LL;
  v43 = (unsigned int)v146;
  v44 = v42 + 168;
  *(_QWORD *)(v42 + 152) = v42 + 168;
  if ( (_DWORD)v43 )
  {
    v121 = v42 + 168;
    sub_31F43D0(v42 + 152, &v145, v43, v41, v36, v37);
    v44 = v121;
  }
  v45 = (unsigned __int64)v123;
  v46 = _mm_loadu_si128(v150);
  v123[22] = v148;
  v47 = v149;
  v48 = *(_QWORD *)(v45 + 8);
  *(__m128i *)(v45 + 192) = v46;
  *(_QWORD *)(v45 + 184) = v47;
  v49 = v33[37];
  v50 = *(unsigned __int64 **)(v33[36] + 8 * (v48 % v49));
  v51 = v48 % v49;
  if ( !v50 )
    goto LABEL_125;
  v52 = (unsigned __int64 *)*v50;
  if ( v48 != *(_QWORD *)(*v50 + 8) )
  {
    do
    {
      v53 = (_QWORD *)*v52;
      if ( !*v52 )
        goto LABEL_125;
      v50 = v52;
      if ( v48 % v49 != v53[1] % v49 )
        goto LABEL_125;
      v52 = (unsigned __int64 *)*v52;
    }
    while ( v48 != v53[1] );
  }
  v121 = *v50;
  if ( !v121 )
  {
LABEL_125:
    v93 = v33[39];
    v94 = v49;
    n = v48;
    v95 = sub_222DA10((__int64)(v33 + 40), v49, v93, 1);
    v97 = n;
    v98 = v96;
    if ( v95 )
    {
      if ( v96 == 1 )
      {
        v99 = v33 + 42;
        v33[42] = 0;
        n = (size_t)(v33 + 42);
      }
      else
      {
        if ( v96 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v33 + 40, v94, v96);
        v116 = n;
        v121 = v96;
        n = 8 * v96;
        v103 = (void *)sub_22077B0(8 * v96);
        v104 = memset(v103, 0, n);
        v97 = v116;
        v98 = v121;
        v99 = v104;
        n = (size_t)(v33 + 42);
      }
      v105 = (_QWORD *)v33[38];
      v33[38] = 0;
      if ( v105 )
      {
        v106 = 0;
        do
        {
          v107 = v105;
          v105 = (_QWORD *)*v105;
          v108 = v107[1] % v98;
          v109 = (_QWORD **)&v99[v108];
          if ( *v109 )
          {
            *v107 = **v109;
            **v109 = v107;
          }
          else
          {
            *v107 = v33[38];
            v33[38] = v107;
            *v109 = v33 + 38;
            if ( *v107 )
              v99[v106] = v107;
            v106 = v108;
          }
        }
        while ( v105 );
      }
      v110 = v33[36];
      if ( v110 != n )
      {
        v117 = v98;
        v121 = (unsigned __int64)v99;
        n = v97;
        j_j___libc_free_0(v110);
        v98 = v117;
        v99 = (_QWORD *)v121;
        v97 = n;
      }
      v33[37] = v98;
      v33[36] = v99;
      v51 = v97 % v98;
    }
    else
    {
      v99 = (_QWORD *)v33[36];
    }
    v100 = (_QWORD *)v99[v51];
    if ( v100 )
    {
      v101 = (unsigned __int64)v123;
      *v123 = *v100;
      **(_QWORD **)(v33[36] + v51 * 8) = v101;
    }
    else
    {
      v111 = (unsigned __int64)v123;
      *v123 = v33[38];
      v33[38] = v111;
      if ( *(_QWORD *)v111 )
        *(_QWORD *)(v33[36] + 8LL * (*(_QWORD *)(*(_QWORD *)v111 + 8LL) % v33[37])) = v111;
      *(_QWORD *)(v33[36] + v51 * 8) = v33 + 38;
    }
    v102 = (unsigned __int64)v123;
    ++v33[39];
    LOBYTE(n) = 1;
    v121 = v102;
    goto LABEL_62;
  }
  v54 = v123[19];
  if ( v44 != v54 )
    _libc_free(v54);
  v55 = v123[15];
  if ( v114 != (_QWORD *)v55 )
    _libc_free(v55);
  v56 = v123[2];
  v57 = v56 + 88LL * *((unsigned int *)v123 + 6);
  if ( v56 != v57 )
  {
    v115 = v9;
    v112 = v11;
    do
    {
      v57 -= 88LL;
      if ( *(_BYTE *)(v57 + 80) )
      {
        v85 = *(_DWORD *)(v57 + 72) <= 0x40u;
        *(_BYTE *)(v57 + 80) = 0;
        if ( !v85 )
        {
          v88 = *(_QWORD *)(v57 + 64);
          if ( v88 )
            j_j___libc_free_0_0(v88);
        }
      }
      v58 = *(_QWORD *)(v57 + 40);
      v59 = v58 + 40LL * *(unsigned int *)(v57 + 48);
      if ( v58 != v59 )
      {
        do
        {
          v59 -= 40LL;
          v60 = *(_QWORD *)(v59 + 8);
          if ( v60 != v59 + 24 )
            _libc_free(v60);
        }
        while ( v58 != v59 );
        v58 = *(_QWORD *)(v57 + 40);
      }
      if ( v58 != v57 + 56 )
        _libc_free(v58);
      sub_C7D6A0(*(_QWORD *)(v57 + 16), 12LL * *(unsigned int *)(v57 + 32), 4);
    }
    while ( v56 != v57 );
    v9 = v115;
    v11 = v112;
    v57 = v123[2];
  }
  if ( v113 != v57 )
    _libc_free(v57);
  j_j___libc_free_0((unsigned __int64)v123);
LABEL_62:
  if ( v145 != &v147 )
    _libc_free((unsigned __int64)v145);
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
  v61 = v139;
  v62 = (unsigned __int64)&v139[88 * (unsigned int)v140];
  if ( v139 != (_BYTE *)v62 )
  {
    v123 = (_QWORD *)v9;
    v118 = v11;
    do
    {
      v62 -= 88LL;
      if ( *(_BYTE *)(v62 + 80) )
      {
        v85 = *(_DWORD *)(v62 + 72) <= 0x40u;
        *(_BYTE *)(v62 + 80) = 0;
        if ( !v85 )
        {
          v86 = *(_QWORD *)(v62 + 64);
          if ( v86 )
            j_j___libc_free_0_0(v86);
        }
      }
      v63 = *(_QWORD *)(v62 + 40);
      v64 = v63 + 40LL * *(unsigned int *)(v62 + 48);
      if ( v63 != v64 )
      {
        do
        {
          v64 -= 40LL;
          v65 = *(_QWORD *)(v64 + 8);
          if ( v65 != v64 + 24 )
            _libc_free(v65);
        }
        while ( v63 != v64 );
        v63 = *(_QWORD *)(v62 + 40);
      }
      if ( v63 != v62 + 56 )
        _libc_free(v63);
      sub_C7D6A0(*(_QWORD *)(v62 + 16), 12LL * *(unsigned int *)(v62 + 32), 4);
    }
    while ( v61 != (_BYTE *)v62 );
    v9 = (unsigned __int64)v123;
    v11 = v118;
    v62 = (unsigned __int64)v139;
  }
  if ( (_BYTE *)v62 != v141 )
    _libc_free(v62);
  if ( v133 != &v135 )
    _libc_free((unsigned __int64)v133);
  if ( v130 != v132 )
    _libc_free((unsigned __int64)v130);
  v66 = v127;
  v67 = (unsigned __int64)&v127[88 * (unsigned int)v128];
  if ( v127 != (_BYTE *)v67 )
  {
    v123 = (_QWORD *)v9;
    v119 = v11;
    do
    {
      v67 -= 88LL;
      if ( *(_BYTE *)(v67 + 80) )
      {
        v85 = *(_DWORD *)(v67 + 72) <= 0x40u;
        *(_BYTE *)(v67 + 80) = 0;
        if ( !v85 )
        {
          v87 = *(_QWORD *)(v67 + 64);
          if ( v87 )
            j_j___libc_free_0_0(v87);
        }
      }
      v68 = *(_QWORD *)(v67 + 40);
      v69 = v68 + 40LL * *(unsigned int *)(v67 + 48);
      if ( v68 != v69 )
      {
        do
        {
          v69 -= 40LL;
          v70 = *(_QWORD *)(v69 + 8);
          if ( v70 != v69 + 24 )
            _libc_free(v70);
        }
        while ( v68 != v69 );
        v68 = *(_QWORD *)(v67 + 40);
      }
      if ( v68 != v67 + 56 )
        _libc_free(v68);
      sub_C7D6A0(*(_QWORD *)(v67 + 16), 12LL * *(unsigned int *)(v67 + 32), 4);
    }
    while ( v66 != (_BYTE *)v67 );
    v9 = (unsigned __int64)v123;
    v11 = v119;
    v67 = (unsigned __int64)v127;
  }
  if ( (_BYTE *)v67 != v120 )
    _libc_free(v67);
  if ( (_BYTE)n )
  {
    v71 = *(_QWORD **)(v11 + 80);
    v72 = v121 + 16;
    v73 = sub_3211F40(v9, *v71);
    *(_QWORD *)(v121 + 176) = v73;
    v74 = v71[1];
    v75 = sub_3211FB0(v9, v74);
    v76 = (unsigned __int8 *)v125;
    *(_QWORD *)(v121 + 184) = v75;
    *(_QWORD *)(v72 + 176) = sub_AF5A10(v76, v74);
    v81 = v126;
    *(_QWORD *)(v72 + 184) = v77;
    if ( v81 )
      sub_320B760(v72, v81, v77, v78, v79, v80);
    v82 = v121 + 120;
    if ( v124 )
      sub_31F4690(v121 + 120, (char **)v124, v77, v78, v79, v80);
    v83 = *(unsigned int *)(a3 + 8);
    if ( v83 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v83 + 1, 8u, v79, v80);
      v83 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v83) = v72;
    v84 = v121;
    ++*(_DWORD *)(a3 + 8);
    sub_320EDF0(v9, v11 + 32, v84 + 152, v72, v82);
  }
}
