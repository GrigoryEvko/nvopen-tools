// Function: sub_275AE90
// Address: 0x275ae90
//
_QWORD *__fastcall sub_275AE90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v9; // rbx
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // rax
  int i; // r14d
  __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // r13
  unsigned int v36; // esi
  int v37; // r10d
  __int64 v38; // r8
  _QWORD *v39; // rdx
  unsigned int v40; // edi
  _QWORD *v41; // rax
  __int64 v42; // rcx
  _DWORD *v43; // rax
  __int64 v44; // r15
  __int64 j; // r13
  int v46; // ecx
  unsigned int v47; // edx
  unsigned __int8 **v48; // rax
  unsigned __int8 *v49; // rdi
  unsigned __int8 *v50; // r8
  int v51; // ecx
  unsigned __int8 *v52; // r14
  __int64 v53; // rsi
  __int64 *v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rsi
  __int64 *v59; // rax
  bool v60; // zf
  int v61; // eax
  int v62; // eax
  char v63; // al
  __int64 v64; // r8
  __int64 v65; // r9
  char v66; // al
  __int64 v67; // rax
  int v68; // eax
  int v69; // r8d
  __int64 v70; // rdi
  int v71; // eax
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 v76; // r14
  __int64 v77; // r12
  __int64 v78; // rbx
  _QWORD *result; // rax
  __int64 v80; // rsi
  _QWORD *v81; // rdi
  __int64 v82; // rdx
  __int64 v83; // rsi
  _QWORD *v84; // rsi
  unsigned __int64 v85; // rdx
  __int64 v86; // rcx
  unsigned __int64 v87; // rdx
  __int64 v88; // rcx
  unsigned __int64 v89; // rdx
  __int64 v90; // rcx
  unsigned __int64 v91; // rdx
  bool v92; // dl
  int v93; // esi
  int v94; // esi
  __int64 v95; // r8
  __int64 v96; // rcx
  __int64 v97; // rdi
  int v98; // r11d
  _QWORD *v99; // r10
  int v100; // ecx
  int v101; // ecx
  __int64 v102; // rdi
  _QWORD *v103; // r9
  __int64 v104; // r15
  int v105; // r10d
  __int64 v106; // rsi
  __int64 v107; // rdx
  __int64 v108; // rcx
  unsigned __int64 v109; // rdx
  unsigned __int64 v110; // rdx
  unsigned __int64 v111; // rcx
  __int64 v112; // [rsp+8h] [rbp-AB8h]
  const void *v113; // [rsp+10h] [rbp-AB0h]
  __int64 v115; // [rsp+38h] [rbp-A88h]
  __int64 v117; // [rsp+60h] [rbp-A60h]
  int v118; // [rsp+60h] [rbp-A60h]
  unsigned __int64 v119; // [rsp+68h] [rbp-A58h]
  __int64 v120; // [rsp+68h] [rbp-A58h]
  __int64 v121; // [rsp+68h] [rbp-A58h]
  __int64 v122; // [rsp+68h] [rbp-A58h]
  __int64 v123; // [rsp+68h] [rbp-A58h]
  unsigned __int64 v124[54]; // [rsp+70h] [rbp-A50h] BYREF
  __m128i v125; // [rsp+220h] [rbp-8A0h] BYREF
  int v126; // [rsp+230h] [rbp-890h]
  int v127; // [rsp+234h] [rbp-88Ch]
  int v128; // [rsp+238h] [rbp-888h]
  char v129; // [rsp+23Ch] [rbp-884h]
  __int64 v130; // [rsp+240h] [rbp-880h] BYREF
  char v131; // [rsp+250h] [rbp-870h]
  unsigned __int64 *v132; // [rsp+280h] [rbp-840h]
  __int64 v133; // [rsp+288h] [rbp-838h]
  unsigned __int64 v134; // [rsp+290h] [rbp-830h] BYREF
  int v135; // [rsp+298h] [rbp-828h]
  unsigned __int64 v136; // [rsp+2A0h] [rbp-820h]
  int v137; // [rsp+2A8h] [rbp-818h]
  __int64 v138; // [rsp+2B0h] [rbp-810h]
  char v139[8]; // [rsp+3D0h] [rbp-6F0h] BYREF
  unsigned __int64 v140; // [rsp+3D8h] [rbp-6E8h]
  char v141; // [rsp+3ECh] [rbp-6D4h]
  char v142[64]; // [rsp+3F0h] [rbp-6D0h] BYREF
  _BYTE *v143; // [rsp+430h] [rbp-690h] BYREF
  __int64 v144; // [rsp+438h] [rbp-688h]
  _BYTE v145[320]; // [rsp+440h] [rbp-680h] BYREF
  __int64 v146; // [rsp+580h] [rbp-540h] BYREF
  unsigned __int64 v147; // [rsp+588h] [rbp-538h] BYREF
  char v148; // [rsp+59Ch] [rbp-524h]
  char v149[64]; // [rsp+5A0h] [rbp-520h] BYREF
  _BYTE *v150; // [rsp+5E0h] [rbp-4E0h] BYREF
  __int64 v151; // [rsp+5E8h] [rbp-4D8h]
  _BYTE v152[320]; // [rsp+5F0h] [rbp-4D0h] BYREF
  char v153[8]; // [rsp+730h] [rbp-390h] BYREF
  unsigned __int64 v154; // [rsp+738h] [rbp-388h]
  char v155; // [rsp+74Ch] [rbp-374h]
  char *v156; // [rsp+790h] [rbp-330h] BYREF
  int v157; // [rsp+798h] [rbp-328h]
  char v158; // [rsp+7A0h] [rbp-320h] BYREF
  char v159[8]; // [rsp+8E0h] [rbp-1E0h] BYREF
  unsigned __int64 v160; // [rsp+8E8h] [rbp-1D8h]
  char v161; // [rsp+8FCh] [rbp-1C4h]
  char *v162; // [rsp+940h] [rbp-180h] BYREF
  unsigned int v163; // [rsp+948h] [rbp-178h]
  char v164; // [rsp+950h] [rbp-170h] BYREF

  v9 = a1;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = &unk_49DDC10;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 1;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 32) = a8;
  v10 = (_QWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 104) = a3;
  *(_QWORD *)(a1 + 112) = a3;
  do
  {
    if ( v10 )
    {
      *v10 = -4;
      v10[1] = -3;
      v10[2] = -4;
      v10[3] = -3;
    }
    v10 += 5;
  }
  while ( v10 != (_QWORD *)(a1 + 456) );
  *(_QWORD *)(a1 + 456) = a1 + 16;
  *(_QWORD *)(a1 + 480) = a1 + 496;
  *(_QWORD *)(a1 + 488) = 0x400000000LL;
  *(_QWORD *)(a1 + 464) = 0;
  *(_BYTE *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 1;
  *(_WORD *)(a1 + 624) = 256;
  *(_QWORD *)(a1 + 632) = &unk_49DDBE8;
  v11 = (_QWORD *)(a1 + 656);
  do
  {
    if ( v11 )
      *v11 = -4096;
    v11 += 2;
  }
  while ( v11 != (_QWORD *)(a1 + 784) );
  *(_QWORD *)(a1 + 784) = a4;
  *(_QWORD *)(a1 + 792) = a5;
  *(_QWORD *)(a1 + 800) = a6;
  *(_QWORD *)(a1 + 808) = a7;
  v12 = sub_B2BEC0(a2);
  *(_BYTE *)(a1 + 1396) = 1;
  *(_QWORD *)(a1 + 816) = v12;
  *(_QWORD *)(a1 + 1368) = 0;
  *(_QWORD *)(a1 + 824) = a8;
  v113 = (const void *)(a1 + 856);
  *(_QWORD *)(a1 + 840) = a1 + 856;
  *(_QWORD *)(a1 + 848) = 0x4000000000LL;
  *(_QWORD *)(a1 + 1376) = a1 + 1400;
  v115 = a1 + 1464;
  *(_QWORD *)(a1 + 1504) = a1 + 1528;
  v112 = a1 + 1656;
  *(_QWORD *)(a1 + 1384) = 4;
  *(_DWORD *)(a1 + 1392) = 0;
  *(_QWORD *)(a1 + 1432) = 0;
  *(_QWORD *)(a1 + 1440) = 0;
  *(_QWORD *)(a1 + 1448) = 0;
  *(_DWORD *)(a1 + 1456) = 0;
  *(_QWORD *)(a1 + 1464) = 0;
  *(_QWORD *)(a1 + 1472) = 0;
  *(_QWORD *)(a1 + 1480) = 0;
  *(_DWORD *)(a1 + 1488) = 0;
  *(_QWORD *)(a1 + 1496) = 0;
  *(_QWORD *)(a1 + 1512) = 16;
  *(_DWORD *)(a1 + 1520) = 0;
  *(_BYTE *)(a1 + 1524) = 1;
  *(_QWORD *)(a1 + 1656) = 0;
  *(_QWORD *)(a1 + 1664) = 0;
  *(_QWORD *)(a1 + 1672) = 0;
  *(_DWORD *)(a1 + 1680) = 0;
  *(_QWORD *)(a1 + 1688) = 0;
  *(_QWORD *)(a1 + 1696) = 0;
  *(_QWORD *)(a1 + 1704) = 0;
  *(_QWORD *)(a1 + 1720) = a1 + 1736;
  *(_QWORD *)(a1 + 1744) = a1 + 1760;
  *(_QWORD *)(a1 + 1752) = 0x600000000LL;
  memset(v124, 0, sizeof(v124));
  *(_QWORD *)(a1 + 1728) = 0;
  v13 = *(_QWORD *)(a2 + 80);
  v124[1] = (unsigned __int64)&v124[4];
  *(_DWORD *)(a1 + 1712) = 0;
  if ( v13 )
    v13 -= 24;
  v124[12] = (unsigned __int64)&v124[14];
  v125.m128i_i64[1] = (__int64)&v130;
  HIDWORD(v124[13]) = 8;
  v126 = 8;
  v128 = 0;
  v129 = 1;
  v132 = &v134;
  v133 = 0x800000000LL;
  v127 = 1;
  v130 = v13;
  v125.m128i_i64[0] = 1;
  v14 = *(_QWORD *)(v13 + 48);
  LODWORD(v124[2]) = 8;
  v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v124[3]) = 1;
  if ( v15 == v13 + 48 )
    goto LABEL_167;
  if ( !v15 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
  {
LABEL_167:
    v16 = 0;
    v18 = 0;
    v17 = 0;
  }
  else
  {
    v117 = v13;
    v119 = v15 - 24;
    v16 = sub_B46E30(v15 - 24);
    v17 = v119;
    v13 = v117;
    v18 = v119;
  }
  v135 = v16;
  v136 = v17;
  v138 = v13;
  v134 = v18;
  v137 = 0;
  LODWORD(v133) = 1;
  sub_CE27D0((__int64)&v125);
  sub_CE35F0((__int64)&v146, (__int64)v124);
  sub_CE35F0((__int64)v139, (__int64)&v125);
  sub_CE35F0((__int64)v153, (__int64)v139);
  sub_CE35F0((__int64)v159, (__int64)&v146);
  if ( v143 != v145 )
    _libc_free((unsigned __int64)v143);
  if ( !v141 )
    _libc_free(v140);
  if ( v150 != v152 )
    _libc_free((unsigned __int64)v150);
  if ( !v148 )
    _libc_free(v147);
  if ( v132 != &v134 )
    _libc_free((unsigned __int64)v132);
  if ( !v129 )
    _libc_free(v125.m128i_u64[1]);
  if ( (unsigned __int64 *)v124[12] != &v124[14] )
    _libc_free(v124[12]);
  if ( !BYTE4(v124[3]) )
    _libc_free(v124[1]);
  sub_C8CD80((__int64)v139, (__int64)v142, (__int64)v153, v19, v20, v21);
  v143 = v145;
  v144 = 0x800000000LL;
  if ( v157 )
    sub_27586E0((__int64)&v143, (__int64 *)&v156, v22, v23, v24, v25);
  sub_C8CD80((__int64)&v146, (__int64)v149, (__int64)v159, v23, v24, v25);
  v29 = v163;
  v150 = v152;
  v151 = 0x800000000LL;
  if ( v163 )
  {
    sub_27586E0((__int64)&v150, (__int64 *)&v162, v26, v163, v27, v28);
    v29 = (unsigned int)v151;
  }
  v30 = (unsigned int)v144;
  for ( i = 0; ; i = v118 )
  {
    v32 = (__int64)v143;
    v33 = 40 * v30;
    if ( v30 != v29 )
      goto LABEL_40;
    if ( v143 == &v143[v33] )
      break;
    v29 = (__int64)v150;
    v34 = (unsigned __int64)v143;
    while ( *(_QWORD *)(v34 + 32) == *(_QWORD *)(v29 + 32)
         && *(_DWORD *)(v34 + 24) == *(_DWORD *)(v29 + 24)
         && *(_DWORD *)(v34 + 8) == *(_DWORD *)(v29 + 8) )
    {
      v34 += 40LL;
      v29 += 40;
      if ( &v143[v33] == (_BYTE *)v34 )
        goto LABEL_102;
    }
LABEL_40:
    v35 = *(_QWORD *)&v143[v33 - 8];
    v36 = *(_DWORD *)(v9 + 1680);
    v118 = i + 1;
    if ( v36 )
    {
      v37 = 1;
      v38 = *(_QWORD *)(v9 + 1664);
      v39 = 0;
      v40 = (v36 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v41 = (_QWORD *)(v38 + 16LL * v40);
      v42 = *v41;
      if ( v35 == *v41 )
      {
LABEL_42:
        v43 = v41 + 1;
        goto LABEL_43;
      }
      while ( v42 != -4096 )
      {
        if ( !v39 && v42 == -8192 )
          v39 = v41;
        v40 = (v36 - 1) & (v37 + v40);
        v41 = (_QWORD *)(v38 + 16LL * v40);
        v42 = *v41;
        if ( v35 == *v41 )
          goto LABEL_42;
        ++v37;
      }
      if ( !v39 )
        v39 = v41;
      v61 = *(_DWORD *)(v9 + 1672);
      ++*(_QWORD *)(v9 + 1656);
      v62 = v61 + 1;
      if ( 4 * v62 < 3 * v36 )
      {
        if ( v36 - *(_DWORD *)(v9 + 1676) - v62 <= v36 >> 3 )
        {
          sub_B23080(v112, v36);
          v100 = *(_DWORD *)(v9 + 1680);
          if ( !v100 )
          {
LABEL_199:
            ++*(_DWORD *)(v9 + 1672);
            BUG();
          }
          v101 = v100 - 1;
          v102 = *(_QWORD *)(v9 + 1664);
          v103 = 0;
          LODWORD(v104) = v101 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v105 = 1;
          v62 = *(_DWORD *)(v9 + 1672) + 1;
          v39 = (_QWORD *)(v102 + 16LL * (unsigned int)v104);
          v106 = *v39;
          if ( v35 != *v39 )
          {
            while ( v106 != -4096 )
            {
              if ( !v103 && v106 == -8192 )
                v103 = v39;
              v104 = v101 & (unsigned int)(v104 + v105);
              v39 = (_QWORD *)(v102 + 16 * v104);
              v106 = *v39;
              if ( v35 == *v39 )
                goto LABEL_73;
              ++v105;
            }
            if ( v103 )
              v39 = v103;
          }
        }
        goto LABEL_73;
      }
    }
    else
    {
      ++*(_QWORD *)(v9 + 1656);
    }
    sub_B23080(v112, 2 * v36);
    v93 = *(_DWORD *)(v9 + 1680);
    if ( !v93 )
      goto LABEL_199;
    v94 = v93 - 1;
    v95 = *(_QWORD *)(v9 + 1664);
    LODWORD(v96) = v94 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v62 = *(_DWORD *)(v9 + 1672) + 1;
    v39 = (_QWORD *)(v95 + 16LL * (unsigned int)v96);
    v97 = *v39;
    if ( v35 != *v39 )
    {
      v98 = 1;
      v99 = 0;
      while ( v97 != -4096 )
      {
        if ( !v99 && v97 == -8192 )
          v99 = v39;
        v96 = v94 & (unsigned int)(v96 + v98);
        v39 = (_QWORD *)(v95 + 16 * v96);
        v97 = *v39;
        if ( v35 == *v39 )
          goto LABEL_73;
        ++v98;
      }
      if ( v99 )
        v39 = v99;
    }
LABEL_73:
    *(_DWORD *)(v9 + 1672) = v62;
    if ( *v39 != -4096 )
      --*(_DWORD *)(v9 + 1676);
    *v39 = v35;
    v43 = v39 + 1;
    *((_DWORD *)v39 + 2) = 0;
LABEL_43:
    *v43 = i;
    v44 = *(_QWORD *)(v35 + 56);
    for ( j = v35 + 48; j != v44; v44 = *(_QWORD *)(v44 + 8) )
    {
      while ( 1 )
      {
        v51 = *(_DWORD *)(a4 + 56);
        v52 = (unsigned __int8 *)(v44 - 24);
        if ( !v44 )
          v52 = 0;
        v53 = *(_QWORD *)(a4 + 40);
        if ( v51 )
        {
          v46 = v51 - 1;
          v47 = v46 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
          v48 = (unsigned __int8 **)(v53 + 16LL * v47);
          v49 = *v48;
          if ( v52 == *v48 )
          {
LABEL_46:
            v50 = v48[1];
            if ( v50 )
            {
              if ( *v50 == 27 && (unsigned int)qword_4FFA708 > *(_DWORD *)(v9 + 848) )
              {
                v120 = (__int64)v48[1];
                v63 = sub_B46490((__int64)v52);
                v64 = v120;
                if ( v63 )
                {
                  if ( (unsigned __int8)(*v52 - 34) <= 0x33u
                    && (v75 = 0x8000000000041LL, _bittest64(&v75, (unsigned int)*v52 - 34)) )
                  {
                    sub_D67230(&v125, v52, *(__int64 **)(v9 + 808));
                    v66 = v131;
                    v64 = v120;
                  }
                  else
                  {
                    sub_D66840(&v125, v52);
                    v66 = v131;
                    v64 = v120;
                  }
                  if ( v66 )
                    goto LABEL_80;
                }
                v121 = v64;
                if ( (unsigned __int8)(*v52 - 34) <= 0x33u )
                {
                  v70 = 0x8000000000041LL;
                  if ( _bittest64(&v70, (unsigned int)*v52 - 34) )
                  {
                    v71 = sub_B49240((__int64)v52);
                    v64 = v121;
                    if ( v71 == 210
                      || (v72 = sub_D5D560((__int64)v52, *(__int64 **)(v9 + 808)), v64 = v121, v72)
                      || (_BYTE)qword_4FFA2A8
                      && (unsigned __int8)(*v52 - 34) <= 0x33u
                      && (v73 = 0x8000000000041LL, _bittest64(&v73, (unsigned int)*v52 - 34))
                      && (v74 = sub_B494D0((__int64)v52, 98), v64 = v121, v74) )
                    {
LABEL_80:
                      v67 = *(unsigned int *)(v9 + 848);
                      if ( v67 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 852) )
                      {
                        v123 = v64;
                        sub_C8D5F0(v9 + 840, v113, v67 + 1, 8u, v64, v65);
                        v67 = *(unsigned int *)(v9 + 848);
                        v64 = v123;
                      }
                      *(_QWORD *)(*(_QWORD *)(v9 + 840) + 8 * v67) = v64;
                      ++*(_DWORD *)(v9 + 848);
                    }
                  }
                }
              }
              goto LABEL_49;
            }
          }
          else
          {
            v68 = 1;
            while ( v49 != (unsigned __int8 *)-4096LL )
            {
              v69 = v68 + 1;
              v47 = v46 & (v68 + v47);
              v48 = (unsigned __int8 **)(v53 + 16LL * v47);
              v49 = *v48;
              if ( v52 == *v48 )
                goto LABEL_46;
              v68 = v69;
            }
          }
        }
        if ( (unsigned __int8)sub_B46790(v52, 0) )
          break;
LABEL_49:
        v44 = *(_QWORD *)(v44 + 8);
        if ( j == v44 )
          goto LABEL_60;
      }
      v58 = *((_QWORD *)v52 + 5);
      if ( !*(_BYTE *)(v9 + 1524) )
        goto LABEL_96;
      v59 = *(__int64 **)(v9 + 1504);
      v55 = *(unsigned int *)(v9 + 1516);
      v54 = &v59[v55];
      if ( v59 != v54 )
      {
        while ( v58 != *v59 )
        {
          if ( v54 == ++v59 )
            goto LABEL_58;
        }
        goto LABEL_49;
      }
LABEL_58:
      if ( (unsigned int)v55 >= *(_DWORD *)(v9 + 1512) )
      {
LABEL_96:
        sub_C8CC70(v9 + 1496, v58, (__int64)v54, v55, v56, v57);
        goto LABEL_49;
      }
      *(_DWORD *)(v9 + 1516) = v55 + 1;
      *v54 = v58;
      ++*(_QWORD *)(v9 + 1496);
    }
LABEL_60:
    v60 = (_DWORD)v144 == 1;
    v30 = (unsigned int)(v144 - 1);
    LODWORD(v144) = v144 - 1;
    if ( !v60 )
    {
      sub_CE27D0((__int64)v139);
      v30 = (unsigned int)v144;
    }
    v29 = (unsigned int)v151;
  }
LABEL_102:
  if ( v150 != v152 )
    _libc_free((unsigned __int64)v150);
  if ( !v148 )
    _libc_free(v147);
  if ( v143 != v145 )
    _libc_free((unsigned __int64)v143);
  if ( !v141 )
    _libc_free(v140);
  if ( v162 != &v164 )
    _libc_free((unsigned __int64)v162);
  if ( !v161 )
    _libc_free(v160);
  if ( v156 != &v158 )
    _libc_free((unsigned __int64)v156);
  if ( !v155 )
    _libc_free(v154);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, v32, v33, v29);
    v76 = *(_QWORD *)(a2 + 96);
    v77 = v76 + 40LL * *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, v32, v107, v108);
      v76 = *(_QWORD *)(a2 + 96);
    }
  }
  else
  {
    v76 = *(_QWORD *)(a2 + 96);
    v77 = v76 + 40LL * *(_QWORD *)(a2 + 104);
  }
  if ( v76 != v77 )
  {
    v122 = v9;
    v78 = v76;
    do
    {
      while ( !(unsigned __int8)sub_B2BAE0(v78) )
      {
        v78 += 40;
        if ( v77 == v78 )
          goto LABEL_125;
      }
      v146 = v78;
      v78 += 40;
      LOBYTE(v147) = 1;
      sub_275ABE0((__int64)v153, v115, &v146, &v147);
    }
    while ( v77 != v78 );
LABEL_125:
    v9 = v122;
  }
  *(_BYTE *)(v9 + 832) = sub_31052D0(a2, a8);
  result = *(_QWORD **)a6;
  v80 = 8LL * *(unsigned int *)(a6 + 8);
  v81 = (_QWORD *)(*(_QWORD *)a6 + v80);
  v82 = v80 >> 3;
  v83 = v80 >> 5;
  if ( !v83 )
    goto LABEL_146;
  v84 = &result[4 * v83];
  do
  {
    v85 = *(_QWORD *)(*result + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v85 == *result + 48LL || !v85 || (unsigned int)*(unsigned __int8 *)(v85 - 24) - 30 > 0xA )
      goto LABEL_198;
    if ( *(_BYTE *)(v85 - 24) == 36 )
      goto LABEL_182;
    v86 = result[1];
    v87 = *(_QWORD *)(v86 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v87 == v86 + 48 || !v87 || (unsigned int)*(unsigned __int8 *)(v87 - 24) - 30 > 0xA )
      goto LABEL_198;
    if ( *(_BYTE *)(v87 - 24) == 36 )
    {
      v92 = v81 != ++result;
      goto LABEL_150;
    }
    v88 = result[2];
    v89 = *(_QWORD *)(v88 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v89 == v88 + 48 || !v89 || (unsigned int)*(unsigned __int8 *)(v89 - 24) - 30 > 0xA )
      goto LABEL_198;
    if ( *(_BYTE *)(v89 - 24) == 36 )
    {
      result += 2;
      v92 = v81 != result;
      goto LABEL_150;
    }
    v90 = result[3];
    v91 = *(_QWORD *)(v90 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v91 == v90 + 48 || !v91 || (unsigned int)*(unsigned __int8 *)(v91 - 24) - 30 > 0xA )
      goto LABEL_198;
    if ( *(_BYTE *)(v91 - 24) == 36 )
    {
      result += 3;
      v92 = v81 != result;
      goto LABEL_150;
    }
    result += 4;
  }
  while ( v84 != result );
  v82 = v81 - result;
LABEL_146:
  switch ( v82 )
  {
    case 2LL:
LABEL_173:
      v110 = *(_QWORD *)(*result + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v110 == *result + 48LL || !v110 || (unsigned int)*(unsigned __int8 *)(v110 - 24) - 30 > 0xA )
        goto LABEL_198;
      if ( *(_BYTE *)(v110 - 24) == 36 )
        goto LABEL_182;
      ++result;
      goto LABEL_178;
    case 3LL:
      v109 = *(_QWORD *)(*result + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v109 == *result + 48LL || !v109 || (unsigned int)*(unsigned __int8 *)(v109 - 24) - 30 > 0xA )
        goto LABEL_198;
      if ( *(_BYTE *)(v109 - 24) == 36 )
      {
        v92 = result != v81;
        goto LABEL_150;
      }
      ++result;
      goto LABEL_173;
    case 1LL:
LABEL_178:
      v111 = *(_QWORD *)(*result + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v111 != *result + 48LL && v111 && (unsigned int)*(unsigned __int8 *)(v111 - 24) - 30 <= 0xA )
      {
        v92 = 0;
        if ( *(_BYTE *)(v111 - 24) != 36 )
          goto LABEL_150;
LABEL_182:
        v92 = v81 != result;
        goto LABEL_150;
      }
LABEL_198:
      BUG();
  }
  v92 = 0;
LABEL_150:
  *(_BYTE *)(v9 + 1736) = v92;
  return result;
}
