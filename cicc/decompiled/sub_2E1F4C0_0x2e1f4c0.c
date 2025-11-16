// Function: sub_2E1F4C0
// Address: 0x2e1f4c0
//
__int64 __fastcall sub_2E1F4C0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  int v8; // r10d
  __int64 *v9; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 *v15; // r9
  _QWORD *v16; // r15
  int v17; // r12d
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rax
  int v24; // edx
  __int64 v25; // rax
  int v26; // r13d
  unsigned __int64 v27; // rdx
  void *v28; // rdx
  char *v29; // r13
  _QWORD *v30; // rax
  char *v31; // r12
  _QWORD *v32; // rbx
  __int64 **v33; // rax
  __int64 v34; // r8
  __int64 *v35; // rax
  __int64 v36; // rcx
  _QWORD *v37; // r9
  _DWORD *v38; // rdi
  __int64 v40; // r8
  unsigned int v41; // esi
  __int64 v42; // rdi
  __int64 v43; // r9
  __int64 v44; // r10
  unsigned int v45; // eax
  __int64 v46; // r13
  __int64 v47; // rdx
  char *v48; // rdi
  char v49; // r14
  __int64 v50; // rdx
  unsigned int *v51; // r11
  unsigned int *v52; // r13
  __int64 v53; // r15
  const __m128i *v54; // r10
  __int64 v55; // rcx
  __int64 v56; // rsi
  unsigned int v57; // eax
  char *v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rsi
  const __m128i *v61; // rcx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // r9
  __m128i *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r8
  char v67; // al
  const void *v68; // rsi
  int v69; // r8d
  __int64 v70; // r11
  int v71; // eax
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rbx
  __int64 v77; // r8
  int v78; // ecx
  unsigned __int64 v79; // rax
  int v80; // ecx
  unsigned int v81; // r15d
  int v82; // ecx
  int v83; // ecx
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rbx
  int v86; // r11d
  int v87; // r11d
  __int64 v88; // r10
  unsigned int v89; // edi
  int v90; // r11d
  int v91; // r11d
  __int64 v92; // r10
  unsigned int v93; // edi
  unsigned int v94; // esi
  int v95; // ecx
  unsigned int v96; // esi
  __int128 v97; // [rsp-20h] [rbp-3E0h]
  __int64 v98; // [rsp+0h] [rbp-3C0h]
  int v99; // [rsp+10h] [rbp-3B0h]
  __int64 *v100; // [rsp+10h] [rbp-3B0h]
  __int64 v101; // [rsp+18h] [rbp-3A8h]
  unsigned int nmemb; // [rsp+28h] [rbp-398h]
  _QWORD *nmemba; // [rsp+28h] [rbp-398h]
  char *v106; // [rsp+40h] [rbp-380h]
  unsigned __int8 v107; // [rsp+48h] [rbp-378h]
  const __m128i *v108; // [rsp+48h] [rbp-378h]
  __int64 v109; // [rsp+48h] [rbp-378h]
  const __m128i *v110; // [rsp+48h] [rbp-378h]
  __int64 v111; // [rsp+50h] [rbp-370h]
  unsigned int *v112; // [rsp+50h] [rbp-370h]
  const __m128i *v113; // [rsp+50h] [rbp-370h]
  char *v114; // [rsp+50h] [rbp-370h]
  unsigned int v115; // [rsp+50h] [rbp-370h]
  __int64 *v116; // [rsp+58h] [rbp-368h]
  __int64 v117; // [rsp+58h] [rbp-368h]
  __int8 *v118; // [rsp+58h] [rbp-368h]
  __int64 v119; // [rsp+58h] [rbp-368h]
  unsigned __int64 v120; // [rsp+58h] [rbp-368h]
  void *base; // [rsp+100h] [rbp-2C0h] BYREF
  __int64 v122; // [rsp+108h] [rbp-2B8h]
  _DWORD v123[16]; // [rsp+110h] [rbp-2B0h] BYREF
  _QWORD *v124; // [rsp+150h] [rbp-270h]
  __int64 v125; // [rsp+158h] [rbp-268h]
  __int64 *v126; // [rsp+160h] [rbp-260h] BYREF
  int v127; // [rsp+190h] [rbp-230h]
  char *v128; // [rsp+198h] [rbp-228h]
  __int64 v129; // [rsp+1A0h] [rbp-220h]
  char v130; // [rsp+1A8h] [rbp-218h] BYREF
  int v131; // [rsp+1D8h] [rbp-1E8h]
  __int64 v132; // [rsp+1E0h] [rbp-1E0h] BYREF
  char *v133; // [rsp+1E8h] [rbp-1D8h] BYREF
  __int64 v134; // [rsp+1F0h] [rbp-1D0h]
  __int64 v135; // [rsp+1F8h] [rbp-1C8h] BYREF
  _BYTE *v136; // [rsp+200h] [rbp-1C0h]
  __int64 v137; // [rsp+208h] [rbp-1B8h]
  _BYTE v138[24]; // [rsp+210h] [rbp-1B0h] BYREF
  int v139; // [rsp+228h] [rbp-198h]
  char *v140; // [rsp+230h] [rbp-190h] BYREF
  __int64 v141; // [rsp+238h] [rbp-188h]
  _BYTE v142[48]; // [rsp+240h] [rbp-180h] BYREF
  int v143; // [rsp+270h] [rbp-150h]

  v8 = 0;
  v9 = 0;
  base = v123;
  v11 = *(unsigned int *)(a3 + 24);
  v101 = a4;
  v99 = *(_DWORD *)(a3 + 24);
  v123[0] = v99;
  v122 = 0x1000000001LL;
  nmemb = 0;
  v107 = 1;
  while ( 2 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8 * v11);
    v13 = *(unsigned int *)(v12 + 72);
    v14 = *(__int64 **)(v12 + 64);
    LOBYTE(a4) = (_DWORD)v13 == 0;
    v8 |= a4;
    if ( v14 == &v14[v13] )
      goto LABEL_17;
    v116 = &v14[v13];
    v15 = v9;
    v16 = a1;
    v17 = v8;
    do
    {
      while ( 1 )
      {
        v19 = *v14;
        v20 = *(unsigned int *)(*v14 + 24);
        v21 = 1LL << v20;
        v22 = (unsigned int)v20 >> 6;
        a4 = 16 * v20;
        if ( (*(_QWORD *)(v16[5] + 8 * v22) & v21) != 0 )
        {
          v18 = *(__int64 **)(v16[18] + a4);
          if ( v18 )
          {
            if ( v18 == v15 || !v15 )
            {
              v15 = *(__int64 **)(v16[18] + a4);
            }
            else
            {
              v107 = 0;
              v15 = *(__int64 **)(v16[18] + a4);
            }
          }
          goto LABEL_8;
        }
        v111 = (__int64)v15;
        v23 = sub_2E0E770(
                a2,
                a7,
                a8,
                *(_QWORD *)(*(_QWORD *)(v16[2] + 152LL) + a4),
                *(_QWORD *)(*(_QWORD *)(v16[2] + 152LL) + a4 + 8));
        a5 = &qword_501EAE0;
        v15 = (__int64 *)v111;
        if ( !(_BYTE)v24 )
          a5 = v23;
        *(_QWORD *)(v16[5] + 8LL * (*(_DWORD *)(v19 + 24) >> 6)) |= 1LL << *(_DWORD *)(v19 + 24);
        a4 = v16[18] + 16LL * *(unsigned int *)(v19 + 24);
        *(_QWORD *)a4 = a5;
        *(_QWORD *)(a4 + 8) = 0;
        if ( !v23 )
        {
          if ( (_BYTE)v24 )
          {
            v17 |= v24;
          }
          else if ( a3 == v19 )
          {
            v101 = 0;
          }
          else
          {
            v25 = (unsigned int)v122;
            a4 = HIDWORD(v122);
            v26 = *(_DWORD *)(v19 + 24);
            v27 = (unsigned int)v122 + 1LL;
            if ( v27 > HIDWORD(v122) )
            {
              sub_C8D5F0((__int64)&base, v123, v27, 4u, (__int64)a5, v111);
              v25 = (unsigned int)v122;
              v15 = (__int64 *)v111;
            }
            *((_DWORD *)base + v25) = v26;
            LODWORD(v122) = v122 + 1;
          }
          goto LABEL_8;
        }
        if ( v111 && (__int64 *)v111 != v23 )
          break;
        v15 = v23;
        v17 |= v24;
LABEL_8:
        if ( v116 == ++v14 )
          goto LABEL_16;
      }
      v107 = 0;
      v15 = v23;
      v17 |= v24;
      ++v14;
    }
    while ( v116 != v14 );
LABEL_16:
    v8 = v17;
    a1 = v16;
    v9 = v15;
LABEL_17:
    if ( (_DWORD)v122 != ++nmemb )
    {
      v11 = *((unsigned int *)base + nmemb);
      continue;
    }
    break;
  }
  *((_DWORD *)a1 + 48) = 0;
  if ( !((unsigned __int8)v8 | (v9 == 0 || v9 == &qword_501EAE0)) || !a8 )
  {
    if ( nmemb <= 4uLL )
      goto LABEL_31;
    goto LABEL_100;
  }
  if ( nmemb > 4uLL )
  {
    v107 = 0;
LABEL_100:
    qsort(base, nmemb, 4u, (__compar_fn_t)sub_2E1D450);
LABEL_31:
    if ( v107 )
    {
      v28 = base;
      v133 = 0;
      v132 = a2;
      v29 = (char *)base;
      v136 = v138;
      v137 = 0x1000000000LL;
      if ( (char *)base + 4 * (unsigned int)v122 != base )
      {
        v30 = a1;
        v31 = (char *)base + 4 * (unsigned int)v122;
        v32 = v30;
        do
        {
          v34 = *(unsigned int *)v29;
          v35 = (__int64 *)(*(_QWORD *)(v32[2] + 152LL) + 16 * v34);
          v36 = *v35;
          v37 = (_QWORD *)v35[1];
          if ( v99 == (_DWORD)v34 && (v101 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v37 = (_QWORD *)v101;
          }
          else
          {
            v33 = (__int64 **)(v32[18] + 16LL
                                       * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v32 + 96LL) + 8 * v34) + 24LL));
            *v33 = v9;
            v33[1] = 0;
          }
          v125 = (__int64)v37;
          v29 += 4;
          v126 = v9;
          *((_QWORD *)&v97 + 1) = v37;
          *(_QWORD *)&v97 = v36;
          v124 = (_QWORD *)v36;
          sub_2E0F380((__int64)&v132, nmemb, (__int64)v28, v36, v34, v37, v97, (__int64)v9);
        }
        while ( v31 != v29 );
      }
      sub_2E0B930((__int64)&v132, nmemb, (__int64)v28, a4, (__int64)a5);
      if ( v136 != v138 )
        _libc_free((unsigned __int64)v136);
      v38 = base;
      goto LABEL_42;
    }
  }
  v40 = a2;
  v41 = *((_DWORD *)a1 + 34);
  v42 = (__int64)(a1 + 14);
  v124 = &v126;
  v125 = 0x600000000LL;
  v127 = 0;
  v128 = &v130;
  v129 = 0x600000000LL;
  v131 = 0;
  v132 = a2;
  v133 = (char *)&v135;
  v134 = 0x600000000LL;
  v139 = 0;
  v140 = v142;
  v141 = 0x600000000LL;
  v143 = 0;
  if ( !v41 )
  {
    ++a1[14];
    goto LABEL_119;
  }
  v43 = v41 - 1;
  v44 = a1[15];
  v45 = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v46 = v44 + 152LL * v45;
  v47 = *(_QWORD *)v46;
  if ( a2 == *(_QWORD *)v46 )
  {
LABEL_48:
    v48 = v133;
    v49 = 0;
    v100 = (__int64 *)(v46 + 8);
    nmemba = (_QWORD *)(v46 + 80);
    if ( v133 != (char *)&v135 )
      goto LABEL_49;
    goto LABEL_50;
  }
  v69 = 1;
  v70 = 0;
  while ( v47 != -4096 )
  {
    if ( !v70 && v47 == -8192 )
      v70 = v46;
    v95 = v69 + 1;
    v45 = v43 & (v45 + v69);
    v40 = 9LL * v45;
    v46 = v44 + 152LL * v45;
    v47 = *(_QWORD *)v46;
    if ( a2 == *(_QWORD *)v46 )
      goto LABEL_48;
    v69 = v95;
  }
  v71 = *((_DWORD *)a1 + 32);
  if ( v70 )
    v46 = v70;
  ++a1[14];
  v72 = (unsigned int)(v71 + 1);
  if ( 4 * (int)v72 >= 3 * v41 )
  {
LABEL_119:
    sub_2E1E780(v42, 2 * v41);
    v90 = *((_DWORD *)a1 + 34);
    if ( v90 )
    {
      v73 = v132;
      v91 = v90 - 1;
      v92 = a1[15];
      v72 = (unsigned int)(*((_DWORD *)a1 + 32) + 1);
      v93 = v91 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
      v74 = 19LL * v93;
      v46 = v92 + 152LL * v93;
      v75 = *(_QWORD *)v46;
      if ( *(_QWORD *)v46 == v132 )
        goto LABEL_76;
      v43 = 1;
      v74 = 0;
      while ( v75 != -4096 )
      {
        if ( !v74 && v75 == -8192 )
          v74 = v46;
        v94 = v43 + 1;
        v43 = v93 + (unsigned int)v43;
        v93 = v91 & v43;
        v46 = v92 + 152LL * (v91 & (unsigned int)v43);
        v75 = *(_QWORD *)v46;
        if ( v132 == *(_QWORD *)v46 )
          goto LABEL_76;
        v43 = v94;
      }
      goto LABEL_115;
    }
LABEL_138:
    ++*((_DWORD *)a1 + 32);
    BUG();
  }
  v73 = a2;
  v74 = v41 - *((_DWORD *)a1 + 33) - (unsigned int)v72;
  v75 = v41 >> 3;
  if ( (unsigned int)v74 <= (unsigned int)v75 )
  {
    sub_2E1E780(v42, v41);
    v86 = *((_DWORD *)a1 + 34);
    if ( v86 )
    {
      v73 = v132;
      v87 = v86 - 1;
      v88 = a1[15];
      v43 = 1;
      v72 = (unsigned int)(*((_DWORD *)a1 + 32) + 1);
      v89 = v87 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
      v46 = v88 + 152LL * v89;
      v74 = 0;
      v75 = *(_QWORD *)v46;
      if ( *(_QWORD *)v46 == v132 )
        goto LABEL_76;
      while ( v75 != -4096 )
      {
        if ( !v74 && v75 == -8192 )
          v74 = v46;
        v96 = v43 + 1;
        v43 = v89 + (unsigned int)v43;
        v89 = v87 & v43;
        v46 = v88 + 152LL * (v87 & (unsigned int)v43);
        v75 = *(_QWORD *)v46;
        if ( v132 == *(_QWORD *)v46 )
          goto LABEL_76;
        v43 = v96;
      }
LABEL_115:
      if ( v74 )
        v46 = v74;
      goto LABEL_76;
    }
    goto LABEL_138;
  }
LABEL_76:
  *((_DWORD *)a1 + 32) = v72;
  if ( *(_QWORD *)v46 != -4096 )
    --*((_DWORD *)a1 + 33);
  *(_QWORD *)v46 = v73;
  v100 = (__int64 *)(v46 + 8);
  *(_QWORD *)(v46 + 8) = v46 + 24;
  *(_QWORD *)(v46 + 16) = 0x600000000LL;
  if ( (_DWORD)v134 )
    sub_2E1D470((__int64)v100, &v133, v74, v75, v72, v43);
  *(_DWORD *)(v46 + 72) = v139;
  nmemba = (_QWORD *)(v46 + 80);
  *(_QWORD *)(v46 + 80) = v46 + 96;
  *(_QWORD *)(v46 + 88) = 0x600000000LL;
  if ( (_DWORD)v141 )
    sub_2E1D470((__int64)nmemba, &v140, v74, v75, v72, v43);
  *(_DWORD *)(v46 + 144) = v143;
  if ( v140 != v142 )
    _libc_free((unsigned __int64)v140);
  v48 = v133;
  v49 = 1;
  if ( v133 != (char *)&v135 )
  {
LABEL_49:
    _libc_free((unsigned __int64)v48);
    if ( !v49 )
      goto LABEL_50;
  }
  v76 = (__int64)(*(_QWORD *)(*a1 + 104LL) - *(_QWORD *)(*a1 + 96LL)) >> 3;
  v77 = (unsigned int)v76;
  v78 = *(_DWORD *)(v46 + 72) & 0x3F;
  if ( v78 )
    *(_QWORD *)(*(_QWORD *)(v46 + 8) + 8LL * *(unsigned int *)(v46 + 16) - 8) &= ~(-1LL << v78);
  v79 = *(unsigned int *)(v46 + 16);
  *(_DWORD *)(v46 + 72) = v76;
  LOBYTE(v80) = v76;
  v81 = (unsigned int)(v76 + 63) >> 6;
  if ( v81 != v79 )
  {
    if ( v81 >= v79 )
    {
      v120 = v81 - v79;
      if ( v81 > (unsigned __int64)*(unsigned int *)(v46 + 20) )
      {
        sub_C8D5F0((__int64)v100, (const void *)(v46 + 24), (unsigned int)(v76 + 63) >> 6, 8u, (unsigned int)v76, v43);
        v79 = *(unsigned int *)(v46 + 16);
        v77 = (unsigned int)v76;
      }
      if ( 8 * v120 )
      {
        v115 = v77;
        memset((void *)(*(_QWORD *)(v46 + 8) + 8 * v79), 0, 8 * v120);
        LODWORD(v79) = *(_DWORD *)(v46 + 16);
        v77 = v115;
      }
      v80 = *(_DWORD *)(v46 + 72);
      *(_DWORD *)(v46 + 16) = v120 + v79;
    }
    else
    {
      *(_DWORD *)(v46 + 16) = v81;
    }
  }
  v82 = v80 & 0x3F;
  if ( v82 )
    *(_QWORD *)(*(_QWORD *)(v46 + 8) + 8LL * *(unsigned int *)(v46 + 16) - 8) &= ~(-1LL << v82);
  v83 = *(_DWORD *)(v46 + 144) & 0x3F;
  if ( v83 )
    *(_QWORD *)(*(_QWORD *)(v46 + 80) + 8LL * *(unsigned int *)(v46 + 88) - 8) &= ~(-1LL << v83);
  v84 = *(unsigned int *)(v46 + 88);
  *(_DWORD *)(v46 + 144) = v76;
  if ( v81 != v84 )
  {
    if ( v81 >= v84 )
    {
      v85 = v81 - v84;
      if ( v81 > (unsigned __int64)*(unsigned int *)(v46 + 92) )
      {
        sub_C8D5F0((__int64)nmemba, (const void *)(v46 + 96), v81, 8u, v77, v43);
        v84 = *(unsigned int *)(v46 + 88);
      }
      if ( 8 * v85 )
      {
        memset((void *)(*(_QWORD *)(v46 + 80) + 8 * v84), 0, 8 * v85);
        LODWORD(v84) = *(_DWORD *)(v46 + 88);
      }
      LODWORD(v77) = *(_DWORD *)(v46 + 144);
      *(_DWORD *)(v46 + 88) = v85 + v84;
    }
    else
    {
      *(_DWORD *)(v46 + 88) = v81;
    }
  }
  v40 = v77 & 0x3F;
  if ( (_DWORD)v40 )
    *(_QWORD *)(*(_QWORD *)(v46 + 80) + 8LL * *(unsigned int *)(v46 + 88) - 8) &= ~(-1LL << v40);
LABEL_50:
  v50 = (unsigned int)v122;
  if ( *((_DWORD *)a1 + 49) < (unsigned int)v122 )
  {
    sub_C8D5F0((__int64)(a1 + 23), a1 + 25, (unsigned int)v122, 0x20u, v40, v43);
    v50 = (unsigned int)v122;
  }
  v38 = base;
  if ( (char *)base + 4 * v50 == base )
  {
    v107 = 0;
  }
  else
  {
    v51 = (unsigned int *)((char *)base + 4 * v50);
    v52 = (unsigned int *)base;
    v53 = a2;
    v54 = (const __m128i *)&v132;
    v98 = (__int64)(a1 + 23);
    do
    {
      while ( 1 )
      {
        v66 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8LL * *v52);
        if ( !a8 )
          break;
        v108 = v54;
        v112 = v51;
        v117 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8LL * *v52);
        v67 = sub_2E1EFB0(a1, v53, a7, a8, v66, v100, nmemba);
        v66 = v117;
        v51 = v112;
        v54 = v108;
        if ( v67 )
          break;
        if ( v112 == ++v52 )
          goto LABEL_65;
      }
      v55 = a1[3];
      v56 = 0;
      v57 = 0;
      if ( v66 )
      {
        v56 = (unsigned int)(*(_DWORD *)(v66 + 24) + 1);
        v57 = *(_DWORD *)(v66 + 24) + 1;
      }
      v58 = 0;
      if ( v57 < *(_DWORD *)(v55 + 32) )
        v58 = *(char **)(*(_QWORD *)(v55 + 24) + 8 * v56);
      v59 = *((unsigned int *)a1 + 48);
      v60 = *((unsigned int *)a1 + 49);
      v133 = v58;
      v61 = v54;
      v132 = v53;
      v62 = a1[23];
      v63 = v59 + 1;
      v134 = 0;
      v135 = 0;
      if ( v59 + 1 > v60 )
      {
        v68 = a1 + 25;
        if ( v62 > (unsigned __int64)v54 || (unsigned __int64)v54 >= v62 + 32 * v59 )
        {
          v110 = v54;
          v114 = (char *)v51;
          v119 = v66;
          sub_C8D5F0(v98, v68, v63, 0x20u, v66, v63);
          v54 = v110;
          v62 = a1[23];
          v59 = *((unsigned int *)a1 + 48);
          v51 = (unsigned int *)v114;
          v66 = v119;
          v61 = v110;
        }
        else
        {
          v106 = (char *)v51;
          v109 = v66;
          v113 = v54;
          v118 = &v54->m128i_i8[-v62];
          sub_C8D5F0(v98, v68, v63, 0x20u, v66, v63);
          v62 = a1[23];
          v54 = v113;
          v66 = v109;
          v61 = (const __m128i *)&v118[v62];
          v51 = (unsigned int *)v106;
          v59 = *((unsigned int *)a1 + 48);
        }
      }
      v64 = (__m128i *)(v62 + 32 * v59);
      *v64 = _mm_loadu_si128(v61);
      v64[1] = _mm_loadu_si128(v61 + 1);
      v65 = (unsigned int)(*((_DWORD *)a1 + 48) + 1);
      *((_DWORD *)a1 + 48) = v65;
      if ( a3 == v66 )
        *(_QWORD *)(a1[23] + 32 * v65 - 16) = v101;
      ++v52;
    }
    while ( v51 != v52 );
LABEL_65:
    v107 = 0;
    v38 = base;
  }
LABEL_42:
  if ( v38 != v123 )
    _libc_free((unsigned __int64)v38);
  return v107;
}
