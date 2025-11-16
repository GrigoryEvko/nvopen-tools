// Function: sub_1A5E350
// Address: 0x1a5e350
//
__int64 __fastcall sub_1A5E350(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 **a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        void (__fastcall *a15)(__int64, _QWORD, char *, _QWORD),
        __int64 a16)
{
  __int64 *v16; // r13
  __int64 *v17; // rbx
  __int64 *v20; // rax
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rdi
  int v24; // esi
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  unsigned __int64 v29; // rcx
  char v30; // al
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // r12
  unsigned __int64 v34; // r12
  int v35; // eax
  unsigned int v36; // eax
  __int128 v37; // rdi
  double v38; // xmm4_8
  double v39; // xmm5_8
  unsigned __int64 *v40; // rax
  __int64 v41; // rbx
  _BYTE *v42; // r14
  _QWORD *v43; // rax
  int v44; // r8d
  __int64 v45; // r15
  unsigned __int8 v46; // al
  __int64 v47; // rax
  __int64 *v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // r9
  _QWORD *v53; // rcx
  int v54; // esi
  _QWORD *v55; // r10
  __int64 *v56; // rax
  __int64 v57; // rsi
  int v58; // r14d
  char v59; // al
  unsigned __int64 v60; // rcx
  char *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r15
  char v64; // dl
  __int64 v65; // r12
  unsigned __int64 v66; // rdi
  int v67; // ebx
  int v68; // eax
  __int64 v69; // rdi
  unsigned int v70; // r12d
  int v71; // ebx
  char v72; // dl
  __int64 v73; // rdx
  _QWORD *v74; // rdi
  __int64 v75; // rsi
  unsigned int v76; // ecx
  __int64 *v77; // rax
  __int64 v78; // r8
  __int64 v79; // rbx
  __int64 *v80; // rax
  __int64 *v81; // rsi
  __int64 *v82; // rcx
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // r9
  const void **v86; // r8
  void *v87; // rax
  unsigned int v88; // r10d
  int v89; // edx
  _QWORD *v90; // rax
  int v91; // eax
  int v92; // r9d
  _BYTE *v93; // r13
  unsigned __int64 v95; // rdi
  _BYTE *v96; // rbx
  __int64 v97; // rax
  unsigned __int64 *v98; // rax
  unsigned __int64 *v99; // r12
  unsigned __int64 v100; // rax
  unsigned __int64 v101; // rdi
  _QWORD *v102; // r14
  _QWORD *v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  _BYTE *v107; // rdx
  _QWORD *v108; // rax
  _QWORD *i; // rdx
  unsigned int v110; // eax
  unsigned int v111; // esi
  unsigned int v112; // edx
  __int64 v113; // rax
  unsigned __int64 *v114; // rax
  __int64 v115; // r13
  _QWORD *v116; // rax
  __int64 v117; // rdx
  size_t v118; // rdx
  int v119; // eax
  int v120; // r9d
  unsigned __int64 v121; // rax
  __int64 *v122; // [rsp+8h] [rbp-2A8h]
  __int64 v123; // [rsp+10h] [rbp-2A0h]
  _QWORD *v124; // [rsp+18h] [rbp-298h]
  int v125; // [rsp+24h] [rbp-28Ch]
  _QWORD *v126; // [rsp+28h] [rbp-288h]
  __int64 *v127; // [rsp+30h] [rbp-280h]
  __int64 v128; // [rsp+30h] [rbp-280h]
  __int64 *v132; // [rsp+50h] [rbp-260h]
  unsigned __int64 v133; // [rsp+50h] [rbp-260h]
  int v136; // [rsp+70h] [rbp-240h]
  unsigned __int8 v137; // [rsp+77h] [rbp-239h]
  int v138; // [rsp+78h] [rbp-238h]
  unsigned __int64 v139; // [rsp+78h] [rbp-238h]
  __int64 *v140; // [rsp+80h] [rbp-230h]
  __int64 v141; // [rsp+80h] [rbp-230h]
  int v142; // [rsp+90h] [rbp-220h]
  unsigned __int64 v143; // [rsp+90h] [rbp-220h]
  unsigned int v144; // [rsp+90h] [rbp-220h]
  unsigned __int64 v146; // [rsp+98h] [rbp-218h]
  int v147; // [rsp+A0h] [rbp-210h]
  char v148; // [rsp+A0h] [rbp-210h]
  unsigned int v149; // [rsp+A0h] [rbp-210h]
  unsigned __int64 v150; // [rsp+A0h] [rbp-210h]
  unsigned __int64 v151; // [rsp+A8h] [rbp-208h]
  __int64 v152; // [rsp+A8h] [rbp-208h]
  __int64 v153; // [rsp+A8h] [rbp-208h]
  __int64 v154; // [rsp+A8h] [rbp-208h]
  unsigned __int64 v155; // [rsp+A8h] [rbp-208h]
  __int64 v156; // [rsp+A8h] [rbp-208h]
  void **v157; // [rsp+A8h] [rbp-208h]
  _QWORD v158[2]; // [rsp+B0h] [rbp-200h] BYREF
  _QWORD *v159; // [rsp+C0h] [rbp-1F0h]
  __int64 v160; // [rsp+C8h] [rbp-1E8h]
  unsigned int v161; // [rsp+D0h] [rbp-1E0h]
  _QWORD v162[2]; // [rsp+D8h] [rbp-1D8h] BYREF
  __int64 v163; // [rsp+E8h] [rbp-1C8h]
  __int64 v164; // [rsp+F0h] [rbp-1C0h] BYREF
  _BYTE *v165; // [rsp+F8h] [rbp-1B8h]
  _BYTE *v166; // [rsp+100h] [rbp-1B0h]
  __int64 v167; // [rsp+108h] [rbp-1A8h]
  int v168; // [rsp+110h] [rbp-1A0h]
  _BYTE v169[40]; // [rsp+118h] [rbp-198h] BYREF
  __int64 v170; // [rsp+140h] [rbp-170h] BYREF
  __int64 *v171; // [rsp+148h] [rbp-168h]
  __int64 *v172; // [rsp+150h] [rbp-160h]
  __int64 v173; // [rsp+158h] [rbp-158h]
  int v174; // [rsp+160h] [rbp-150h]
  _BYTE v175[40]; // [rsp+168h] [rbp-148h] BYREF
  _BYTE *v176; // [rsp+190h] [rbp-120h] BYREF
  __int64 v177; // [rsp+198h] [rbp-118h]
  _BYTE v178[64]; // [rsp+1A0h] [rbp-110h] BYREF
  __int64 v179; // [rsp+1E0h] [rbp-D0h] BYREF
  __int64 v180; // [rsp+1E8h] [rbp-C8h]
  __int64 v181; // [rsp+1F0h] [rbp-C0h] BYREF
  unsigned int v182; // [rsp+1F8h] [rbp-B8h]
  _QWORD *v183; // [rsp+230h] [rbp-80h] BYREF
  __int64 v184; // [rsp+238h] [rbp-78h] BYREF
  _QWORD v185[8]; // [rsp+240h] [rbp-70h] BYREF
  char v186; // [rsp+280h] [rbp-30h] BYREF

  v16 = *(__int64 **)(a1 + 32);
  v17 = *(__int64 **)(a1 + 40);
  v176 = v178;
  v177 = 0x400000000LL;
  if ( v17 == v16 )
    return 0;
  do
  {
    while ( 1 )
    {
      v22 = *(_DWORD *)(a3 + 24);
      if ( !v22 )
        goto LABEL_5;
      v23 = *v16;
      v24 = v22 - 1;
      v25 = *(_QWORD *)(a3 + 8);
      v26 = (v22 - 1) & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( *v16 != *v27 )
      {
        v91 = 1;
        while ( v28 != -8 )
        {
          v92 = v91 + 1;
          v26 = v24 & (v91 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v23 == *v27 )
            goto LABEL_8;
          v91 = v92;
        }
        goto LABEL_5;
      }
LABEL_8:
      if ( a1 != v27[1] )
        goto LABEL_5;
      v29 = sub_157EBA0(v23);
      v30 = *(_BYTE *)(v29 + 16);
      if ( v30 == 27 )
      {
        if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
        {
          v20 = *(__int64 **)(v29 - 8);
          v21 = *v20;
          if ( *(_BYTE *)(*v20 + 16) > 0x10u )
            goto LABEL_97;
        }
        else
        {
          v21 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(v21 + 16) > 0x10u )
          {
LABEL_97:
            v154 = v29;
            if ( sub_13FC1A0(a1, v21) )
            {
              v83 = *(_QWORD *)sub_13CF970(v154);
              v183 = (_QWORD *)v154;
              v179 = v83;
              v184 = v83;
              if ( (v83 & 4) != 0 )
              {
                v155 = v83 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v83 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                {
                  v84 = sub_22077B0(48);
                  v85 = v84;
                  if ( v84 )
                  {
                    v86 = (const void **)v155;
                    v87 = (void *)(v84 + 16);
                    *(_QWORD *)(v85 + 8) = 0x400000000LL;
                    *(_QWORD *)v85 = v87;
                    v88 = *(_DWORD *)(v155 + 8);
                    if ( v88 )
                    {
                      if ( v85 != v155 )
                      {
                        v118 = 8LL * v88;
                        if ( v88 <= 4 )
                          goto LABEL_199;
                        v144 = *(_DWORD *)(v155 + 8);
                        v150 = v155;
                        v157 = (void **)v85;
                        sub_16CD150(v85, v87, v88, 8, (int)v86, v85);
                        v86 = (const void **)v150;
                        v85 = (__int64)v157;
                        v88 = v144;
                        v87 = *v157;
                        v118 = 8LL * *(unsigned int *)(v150 + 8);
                        if ( v118 )
                        {
LABEL_199:
                          v149 = v88;
                          v156 = v85;
                          memcpy(v87, *v86, v118);
                          v88 = v149;
                          v85 = v156;
                        }
                        *(_DWORD *)(v85 + 8) = v88;
                      }
                    }
                  }
                  v184 = v85 | 4;
                }
              }
              v89 = v177;
              if ( (unsigned int)v177 >= HIDWORD(v177) )
              {
                sub_1A53FD0((unsigned int *)&v176, 0);
                v89 = v177;
              }
              v90 = &v176[16 * v89];
              if ( v90 )
              {
                *v90 = v183;
                v90[1] = v184;
                v89 = v177;
                v184 = 0;
              }
              LODWORD(v177) = v89 + 1;
              sub_1A517A0((unsigned __int64 **)&v184);
              sub_1A517A0((unsigned __int64 **)&v179);
            }
            goto LABEL_5;
          }
        }
        goto LABEL_5;
      }
      if ( v30 != 26 )
        goto LABEL_5;
      if ( (*(_DWORD *)(v29 + 20) & 0xFFFFFFF) != 3 )
        goto LABEL_5;
      v31 = *(_QWORD *)(v29 - 72);
      if ( *(_BYTE *)(v31 + 16) <= 0x10u )
        goto LABEL_5;
      v151 = v29;
      if ( *(_QWORD *)(v29 - 48) == *(_QWORD *)(v29 - 24) )
        goto LABEL_5;
      if ( sub_13FC1A0(a1, v31) )
      {
        v114 = *(unsigned __int64 **)(v151 - 72);
        v183 = (_QWORD *)v151;
        v179 = (__int64)v114;
        sub_1A53F10(&v184, &v179);
        sub_1A54170((unsigned int *)&v176, &v183);
        sub_1A517A0((unsigned __int64 **)&v184);
        sub_1A517A0((unsigned __int64 **)&v179);
        goto LABEL_5;
      }
      v32 = *(_QWORD *)(v151 - 72);
      if ( (unsigned __int8)(*(_BYTE *)(v32 + 16) - 50) <= 1u )
        break;
LABEL_5:
      if ( v17 == ++v16 )
        goto LABEL_21;
    }
    sub_1A511F0(&v179, a1, v32);
    if ( (v179 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v179 & 4) == 0 || *(_DWORD *)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
    {
      v183 = (_QWORD *)v151;
      v184 = v179;
      v179 = 0;
      sub_1A54170((unsigned int *)&v176, &v183);
      sub_1A517A0((unsigned __int64 **)&v184);
    }
    ++v16;
    sub_1A517A0((unsigned __int64 **)&v179);
  }
  while ( v17 != v16 );
LABEL_21:
  if ( (_DWORD)v177 )
  {
    v33 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
    v158[0] = a1;
    v34 = (unsigned int)(v33 >> 3);
    v35 = sub_1454B60((unsigned int)v34);
    v158[1] = 0;
    if ( v35 )
    {
      v36 = sub_1454B60(4 * v35 / 3u + 1);
      v161 = v36;
      if ( !v36 )
        goto LABEL_24;
      v108 = (_QWORD *)sub_22077B0(16LL * v36);
      v160 = 0;
      v159 = v108;
      for ( i = &v108[2 * v161]; i != v108; v108 += 2 )
      {
        if ( v108 )
          *v108 = -8;
      }
      v34 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3);
    }
    else
    {
      v161 = 0;
LABEL_24:
      v159 = 0;
      v160 = 0;
    }
    v162[0] = 0;
    v162[1] = 0;
    v163 = 0;
    sub_13FC0C0((__int64)v162, v34);
    *(_QWORD *)&v37 = v158;
    *((_QWORD *)&v37 + 1) = a3;
    sub_13FF3D0(v37);
    v137 = sub_1A523B0((__int64)v158, a3);
    if ( v137 )
    {
      v137 = 0;
      goto LABEL_123;
    }
    v165 = v169;
    v166 = v169;
    v164 = 0;
    v167 = 4;
    v168 = 0;
    sub_14D04F0(a1, a4, (__int64)&v164);
    v40 = (unsigned __int64 *)&v181;
    v179 = 0;
    v180 = 1;
    do
    {
      *v40 = -8;
      v40 += 2;
    }
    while ( v40 != (unsigned __int64 *)&v183 );
    v127 = *(__int64 **)(a1 + 40);
    if ( *(__int64 **)(a1 + 32) == v127 )
    {
      v136 = 0;
      goto LABEL_67;
    }
    v132 = *(__int64 **)(a1 + 32);
    v136 = 0;
    while ( 2 )
    {
      v147 = 0;
      v170 = *v132;
      v41 = *(_QWORD *)(v170 + 48);
      v152 = v170 + 40;
      if ( v170 + 40 == v41 )
        goto LABEL_65;
      while ( 2 )
      {
        while ( 2 )
        {
          v44 = v41 - 24;
          v45 = 0;
          v43 = v165;
          if ( v41 )
            v45 = v41 - 24;
          if ( v166 == v165 )
          {
            v42 = &v165[8 * HIDWORD(v167)];
            if ( v165 == v42 )
            {
              v107 = v165;
            }
            else
            {
              do
              {
                if ( v45 == *v43 )
                  break;
                ++v43;
              }
              while ( v42 != (_BYTE *)v43 );
              v107 = &v165[8 * HIDWORD(v167)];
            }
LABEL_47:
            while ( v107 != (_BYTE *)v43 )
            {
              if ( *v43 < 0xFFFFFFFFFFFFFFFELL )
                goto LABEL_35;
              ++v43;
            }
            if ( v43 == (_QWORD *)v42 )
              break;
            goto LABEL_36;
          }
          v42 = &v166[8 * (unsigned int)v167];
          v43 = sub_16CC9F0((__int64)&v164, v45);
          if ( v45 == *v43 )
          {
            if ( v166 == v165 )
              v107 = &v166[8 * HIDWORD(v167)];
            else
              v107 = &v166[8 * (unsigned int)v167];
            goto LABEL_47;
          }
          if ( v166 == v165 )
          {
            v43 = &v166[8 * HIDWORD(v167)];
            v107 = v43;
            goto LABEL_47;
          }
          v43 = &v166[8 * (unsigned int)v167];
LABEL_35:
          if ( v43 != (_QWORD *)v42 )
          {
LABEL_36:
            v41 = *(_QWORD *)(v41 + 8);
            if ( v152 == v41 )
              goto LABEL_64;
            continue;
          }
          break;
        }
        if ( *(_BYTE *)(*(_QWORD *)v45 + 8LL) == 10 && (unsigned __int8)sub_15F2E00(v45, v170) )
          goto LABEL_120;
        v46 = *(_BYTE *)(v45 + 16);
        if ( v46 > 0x17u )
        {
          if ( v46 == 78 )
          {
            v100 = v45 | 4;
          }
          else
          {
            if ( v46 != 29 )
              goto LABEL_53;
            v100 = v45 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v101 = v100 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v100 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v102 = (_QWORD *)(v101 + 56);
            v143 = v100 & 0xFFFFFFFFFFFFFFF8LL;
            v103 = (_QWORD *)(v101 + 56);
            if ( (v100 & 4) != 0 )
            {
              if ( (unsigned __int8)sub_1560260(v103, -1, 8) )
                goto LABEL_120;
              v104 = *(_QWORD *)(v143 - 24);
              if ( !*(_BYTE *)(v104 + 16) )
              {
                v183 = *(_QWORD **)(v104 + 112);
                if ( (unsigned __int8)sub_1560260(&v183, -1, 8) )
                  goto LABEL_120;
              }
              if ( (unsigned __int8)sub_1560260(v102, -1, 24) )
                goto LABEL_120;
              v105 = *(_QWORD *)(v143 - 24);
              if ( !*(_BYTE *)(v105 + 16) )
                goto LABEL_142;
            }
            else
            {
              if ( (unsigned __int8)sub_1560260(v103, -1, 8) )
                goto LABEL_120;
              v106 = *(_QWORD *)(v143 - 72);
              if ( !*(_BYTE *)(v106 + 16) )
              {
                v183 = *(_QWORD **)(v106 + 112);
                if ( (unsigned __int8)sub_1560260(&v183, -1, 8) )
                  goto LABEL_120;
              }
              if ( (unsigned __int8)sub_1560260(v102, -1, 24) )
                goto LABEL_120;
              v105 = *(_QWORD *)(v143 - 72);
              if ( !*(_BYTE *)(v105 + 16) )
              {
LABEL_142:
                v183 = *(_QWORD **)(v105 + 112);
                if ( (unsigned __int8)sub_1560260(&v183, -1, 24) )
                  goto LABEL_120;
              }
            }
          }
        }
LABEL_53:
        v47 = 3LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v45 + 23) & 0x40) != 0 )
        {
          v48 = *(__int64 **)(v45 - 8);
          v49 = (__int64)&v48[v47];
        }
        else
        {
          v48 = (__int64 *)(v45 - v47 * 8);
          v49 = v45;
        }
        v50 = v49 - (_QWORD)v48;
        v183 = v185;
        v184 = 0x400000000LL;
        v51 = 0xAAAAAAAAAAAAAAABLL * (v50 >> 3);
        v52 = v51;
        if ( (unsigned __int64)v50 > 0x60 )
        {
          v139 = 0xAAAAAAAAAAAAAAABLL * (v50 >> 3);
          v141 = v50;
          sub_16CD150((__int64)&v183, v185, v51, 8, v44, v51);
          v55 = v183;
          v54 = v184;
          LODWORD(v51) = v139;
          v50 = v141;
          v52 = v139;
          v53 = &v183[(unsigned int)v184];
        }
        else
        {
          v53 = v185;
          v54 = 0;
          v55 = v185;
        }
        if ( v50 > 0 )
        {
          v56 = v48;
          do
          {
            v57 = *v56;
            ++v53;
            v56 += 3;
            *(v53 - 1) = v57;
            --v52;
          }
          while ( v52 );
          v55 = v183;
          v54 = v184;
        }
        LODWORD(v184) = v54 + v51;
        v58 = sub_14A5330(a5, v45, (__int64)v55, (unsigned int)(v54 + v51));
        if ( v183 != v185 )
          _libc_free((unsigned __int64)v183);
        v147 += v58;
        v41 = *(_QWORD *)(v41 + 8);
        if ( v152 != v41 )
          continue;
        break;
      }
LABEL_64:
      v136 += v147;
LABEL_65:
      v59 = sub_1A545D0((__int64)&v179, &v170, &v183);
      v60 = (unsigned __int64)v183;
      if ( !v59 )
      {
        ++v179;
        v110 = ((unsigned int)v180 >> 1) + 1;
        if ( (v180 & 1) != 0 )
        {
          v112 = 12;
          v111 = 4;
        }
        else
        {
          v111 = v182;
          v112 = 3 * v182;
        }
        if ( v112 <= 4 * v110 )
        {
          v111 *= 2;
        }
        else if ( v111 - (v110 + HIDWORD(v180)) > v111 >> 3 )
        {
LABEL_162:
          LODWORD(v180) = v180 & 1 | (2 * v110);
          if ( *(_QWORD *)v60 != -8 )
            --HIDWORD(v180);
          v113 = v170;
          *(_DWORD *)(v60 + 8) = 0;
          *(_QWORD *)v60 = v113;
          goto LABEL_66;
        }
        sub_1A55E90((__int64)&v179, v111);
        sub_1A545D0((__int64)&v179, &v170, &v183);
        v60 = (unsigned __int64)v183;
        v110 = ((unsigned int)v180 >> 1) + 1;
        goto LABEL_162;
      }
LABEL_66:
      ++v132;
      *(_DWORD *)(v60 + 8) = v147;
      if ( v127 != v132 )
        continue;
      break;
    }
LABEL_67:
    v183 = 0;
    v61 = (char *)v185;
    v184 = 1;
    do
    {
      *(_QWORD *)v61 = -8;
      v61 += 16;
    }
    while ( v61 != &v186 );
    v124 = 0;
    v123 = 0;
    v62 = 16LL * (unsigned int)v177;
    v122 = (__int64 *)&v176[v62];
    if ( v176 == &v176[v62] )
    {
      v128 = 0;
LABEL_187:
      if ( v125 < dword_4FB42C0 )
        v137 = sub_1A5B3D0(a1, v128, v124, v123, a2, a3, a7, a8, a9, a10, v38, v39, a13, a14, a4, a15, a16, a6);
      if ( (v184 & 1) == 0 )
        j___libc_free_0(v185[0]);
LABEL_120:
      if ( (v180 & 1) != 0 )
      {
        v95 = (unsigned __int64)v166;
        if ( v166 == v165 )
        {
LABEL_123:
          if ( v162[0] )
            j_j___libc_free_0(v162[0], v163 - v162[0]);
          j___libc_free_0(v159);
          v96 = v176;
          v93 = &v176[16 * (unsigned int)v177];
          if ( v176 != v93 )
          {
            do
            {
              v97 = *((_QWORD *)v93 - 1);
              v93 -= 16;
              if ( (v97 & 4) != 0 )
              {
                v98 = (unsigned __int64 *)(v97 & 0xFFFFFFFFFFFFFFF8LL);
                v99 = v98;
                if ( v98 )
                {
                  if ( (unsigned __int64 *)*v98 != v98 + 2 )
                    _libc_free(*v98);
                  j_j___libc_free_0(v99, 48);
                }
              }
            }
            while ( v96 != v93 );
            v93 = v176;
          }
          goto LABEL_113;
        }
      }
      else
      {
        j___libc_free_0(v181);
        v95 = (unsigned __int64)v166;
        if ( v166 == v165 )
          goto LABEL_123;
      }
      _libc_free(v95);
      goto LABEL_123;
    }
    v140 = (__int64 *)v176;
    v128 = 0;
    while ( 2 )
    {
      v63 = *v140;
      v64 = *(_BYTE *)(*v140 + 16);
      v133 = v140[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v133 )
      {
        v126 = 0;
        v148 = v64 != 26;
        goto LABEL_73;
      }
      v148 = (v140[1] >> 2) & 1;
      if ( ((v140[1] >> 2) & 1) != 0 )
      {
        v121 = *(unsigned int *)(v133 + 8);
        v126 = *(_QWORD **)v133;
        v133 = v121;
        if ( v64 == 26 )
        {
          v148 = 0;
          if ( v121 == 1 )
          {
LABEL_195:
            v133 = 1;
            v148 = *v126 == *(_QWORD *)(v63 - 72);
          }
        }
      }
      else
      {
        v126 = v140 + 1;
        if ( v64 == 26 )
          goto LABEL_195;
        v133 = 1;
        v148 = 1;
      }
LABEL_73:
      v65 = *(_QWORD *)(v63 + 40);
      v170 = 0;
      v173 = 4;
      v153 = v65;
      v171 = (__int64 *)v175;
      v172 = (__int64 *)v175;
      v174 = 0;
      v66 = sub_157EBA0(v65);
      v67 = -v136;
      if ( !v66 )
      {
LABEL_184:
        if ( !v128 || v67 < v125 )
        {
          v125 = v67;
          v128 = v63;
          v124 = v126;
          v123 = v133;
        }
        v140 += 2;
        if ( v122 == v140 )
          goto LABEL_187;
        continue;
      }
      break;
    }
    v68 = sub_15F4D60(v66);
    v69 = v65;
    v70 = 0;
    v142 = v68;
    v71 = v68;
    v138 = v136;
    v146 = sub_157EBA0(v69);
    if ( !v71 )
    {
LABEL_182:
      v67 = v138 * (HIDWORD(v173) - 1 - v174);
      if ( v171 != v172 )
        _libc_free((unsigned __int64)v172);
      goto LABEL_184;
    }
    while ( 2 )
    {
      v79 = sub_15F4DF0(v146, v70);
      v80 = v171;
      if ( v172 != v171 )
        goto LABEL_76;
      v81 = &v171[HIDWORD(v173)];
      if ( v171 != v81 )
      {
        v82 = 0;
        while ( v79 != *v80 )
        {
          if ( *v80 == -2 )
            v82 = v80;
          if ( v81 == ++v80 )
          {
            if ( !v82 )
              goto LABEL_191;
            *v82 = v79;
            --v174;
            ++v170;
            goto LABEL_77;
          }
        }
LABEL_86:
        if ( ++v70 == v142 )
          goto LABEL_182;
        continue;
      }
      break;
    }
LABEL_191:
    if ( HIDWORD(v173) < (unsigned int)v173 )
    {
      ++HIDWORD(v173);
      *v81 = v79;
      ++v170;
    }
    else
    {
LABEL_76:
      sub_16CCBA0((__int64)&v170, v79);
      if ( !v72 )
        goto LABEL_86;
    }
LABEL_77:
    if ( !v148 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v63 - 72) + 16LL) == 50 )
      {
        if ( v79 == *(_QWORD *)(v63 - 48) )
          goto LABEL_86;
      }
      else if ( v79 == *(_QWORD *)(v63 - 24) )
      {
        goto LABEL_86;
      }
    }
    if ( !sub_157F120(v79) )
    {
      v115 = *(_QWORD *)(v79 + 8);
      if ( v115 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v115) + 16) - 25) > 9u )
        {
          v115 = *(_QWORD *)(v115 + 8);
          if ( !v115 )
            goto LABEL_81;
        }
        v116 = sub_1648700(v115);
LABEL_175:
        v117 = v116[5];
        if ( v153 != v117 && !sub_15CC8F0(a2, v79, v117) )
          goto LABEL_86;
        while ( 1 )
        {
          v115 = *(_QWORD *)(v115 + 8);
          if ( !v115 )
            break;
          v116 = sub_1648700(v115);
          if ( (unsigned __int8)(*((_BYTE *)v116 + 16) - 25) <= 9u )
            goto LABEL_175;
        }
      }
    }
LABEL_81:
    v73 = *(unsigned int *)(a2 + 48);
    v74 = 0;
    if ( (_DWORD)v73 )
    {
      v75 = *(_QWORD *)(a2 + 32);
      v76 = (v73 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
      v77 = (__int64 *)(v75 + 16LL * v76);
      v78 = *v77;
      if ( *v77 == v79 )
      {
LABEL_83:
        if ( v77 != (__int64 *)(v75 + 16 * v73) )
        {
          v74 = (_QWORD *)v77[1];
          goto LABEL_85;
        }
      }
      else
      {
        v119 = 1;
        while ( v78 != -8 )
        {
          v120 = v119 + 1;
          v76 = (v73 - 1) & (v119 + v76);
          v77 = (__int64 *)(v75 + 16LL * v76);
          v78 = *v77;
          if ( v79 == *v77 )
            goto LABEL_83;
          v119 = v120;
        }
      }
      v74 = 0;
    }
LABEL_85:
    v138 -= sub_1A55C50(v74, (__int64)&v179, (__int64)&v183);
    goto LABEL_86;
  }
  v137 = 0;
  v93 = v176;
LABEL_113:
  if ( v93 != v178 )
    _libc_free((unsigned __int64)v93);
  return v137;
}
