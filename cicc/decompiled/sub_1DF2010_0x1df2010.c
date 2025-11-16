// Function: sub_1DF2010
// Address: 0x1df2010
//
__int64 __fastcall sub_1DF2010(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // ecx
  __int64 v5; // rsi
  int v6; // ecx
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  unsigned int v11; // ebx
  void *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  unsigned int v21; // r12d
  unsigned int v22; // r13d
  __int64 v23; // rdx
  __int64 v24; // r14
  int v25; // r8d
  __int64 v26; // r9
  __int64 v27; // r14
  unsigned int v28; // r13d
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rsi
  unsigned int v32; // ecx
  int *v33; // rdx
  int v34; // r8d
  __int64 v35; // rax
  unsigned int v36; // r15d
  unsigned int v37; // eax
  int v38; // eax
  int v39; // r10d
  unsigned int v40; // r10d
  __int64 v41; // rax
  unsigned int v42; // ebx
  unsigned int v43; // r14d
  unsigned int v44; // r14d
  __int64 v45; // rsi
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // rdx
  bool v49; // r12
  __int64 v50; // r13
  __int64 v51; // r12
  _BYTE *v52; // rbx
  _DWORD *v53; // rdi
  __int64 v54; // rax
  _DWORD *v55; // rax
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // r15
  int v59; // eax
  int v60; // r10d
  int v61; // r10d
  __int64 v62; // rcx
  unsigned int v63; // edx
  __int64 *v64; // rax
  __int64 v65; // rdi
  unsigned int v66; // eax
  int v67; // r9d
  __int64 v68; // rax
  unsigned int v69; // ebx
  unsigned int v70; // r12d
  unsigned int v71; // eax
  __int64 v72; // rcx
  int v73; // r8d
  int v74; // r9d
  __int64 v75; // rdx
  __int64 v76; // rax
  char v77; // r12
  __int64 v78; // rbx
  unsigned int v79; // ecx
  unsigned int v80; // edx
  __int64 v81; // rdx
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rax
  int v84; // ebx
  __int64 v85; // r12
  _DWORD *v86; // rax
  _DWORD *i; // rdx
  int v88; // r9d
  __int64 v89; // rbx
  char v90; // bl
  _DWORD *v91; // rax
  __int64 v92; // r8
  int v93; // r9d
  __int64 v94; // rcx
  __int64 v95; // r8
  int v96; // r9d
  unsigned int v97; // r14d
  int v98; // eax
  int v99; // eax
  unsigned int *v100; // rbx
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rdi
  __int64 v104; // rdx
  int v106; // esi
  int v107; // edi
  int v108; // eax
  int v109; // edi
  __int64 v110; // [rsp+18h] [rbp-4C8h]
  __int64 v111; // [rsp+30h] [rbp-4B0h]
  unsigned __int8 v112; // [rsp+3Ah] [rbp-4A6h]
  unsigned __int8 v113; // [rsp+3Bh] [rbp-4A5h]
  unsigned int v114; // [rsp+3Ch] [rbp-4A4h]
  __int64 v115; // [rsp+40h] [rbp-4A0h]
  __int64 v116; // [rsp+48h] [rbp-498h]
  __int64 v117; // [rsp+50h] [rbp-490h]
  __int64 v118; // [rsp+58h] [rbp-488h]
  unsigned int *v119; // [rsp+68h] [rbp-478h]
  char v120; // [rsp+83h] [rbp-45Dh]
  unsigned int v121; // [rsp+84h] [rbp-45Ch]
  __int64 v123; // [rsp+A8h] [rbp-438h]
  unsigned int *v124; // [rsp+B0h] [rbp-430h]
  __int64 v125; // [rsp+B8h] [rbp-428h]
  _BYTE *v127; // [rsp+C8h] [rbp-418h]
  __int64 v128; // [rsp+D0h] [rbp-410h]
  __int64 v129; // [rsp+D8h] [rbp-408h]
  int v130; // [rsp+E0h] [rbp-400h]
  int v131; // [rsp+E0h] [rbp-400h]
  __int64 v132; // [rsp+E8h] [rbp-3F8h]
  unsigned int v133; // [rsp+E8h] [rbp-3F8h]
  __int64 v134; // [rsp+E8h] [rbp-3F8h]
  unsigned int v135; // [rsp+E8h] [rbp-3F8h]
  unsigned int *v136; // [rsp+E8h] [rbp-3F8h]
  _QWORD v137[2]; // [rsp+F0h] [rbp-3F0h] BYREF
  _QWORD *v138; // [rsp+100h] [rbp-3E0h]
  __int64 v139; // [rsp+108h] [rbp-3D8h]
  _QWORD v140[2]; // [rsp+110h] [rbp-3D0h] BYREF
  __int64 v141; // [rsp+120h] [rbp-3C0h] BYREF
  _DWORD *v142; // [rsp+128h] [rbp-3B8h]
  __int64 v143; // [rsp+130h] [rbp-3B0h]
  unsigned int v144; // [rsp+138h] [rbp-3A8h]
  unsigned int *v145; // [rsp+140h] [rbp-3A0h] BYREF
  __int64 v146; // [rsp+148h] [rbp-398h]
  _BYTE v147[64]; // [rsp+150h] [rbp-390h] BYREF
  _BYTE *v148; // [rsp+190h] [rbp-350h] BYREF
  __int64 v149; // [rsp+198h] [rbp-348h]
  _BYTE v150[128]; // [rsp+1A0h] [rbp-340h] BYREF
  __int64 *v151; // [rsp+220h] [rbp-2C0h] BYREF
  __int64 v152; // [rsp+228h] [rbp-2B8h]
  __int64 v153; // [rsp+230h] [rbp-2B0h] BYREF
  int v154; // [rsp+238h] [rbp-2A8h]
  _BYTE *v155; // [rsp+2B0h] [rbp-230h] BYREF
  __int64 v156; // [rsp+2B8h] [rbp-228h]
  _BYTE v157[128]; // [rsp+2C0h] [rbp-220h] BYREF
  _DWORD *v158; // [rsp+340h] [rbp-1A0h] BYREF
  __int64 v159; // [rsp+348h] [rbp-198h]
  _DWORD v160[32]; // [rsp+350h] [rbp-190h] BYREF
  unsigned __int64 v161[2]; // [rsp+3D0h] [rbp-110h] BYREF
  _BYTE v162[192]; // [rsp+3E0h] [rbp-100h] BYREF
  unsigned __int64 v163; // [rsp+4A0h] [rbp-40h]
  unsigned int v164; // [rsp+4A8h] [rbp-38h]

  v123 = *(_QWORD *)(a2 + 32);
  v2 = *(_QWORD *)(a1 + 336);
  v111 = 0;
  v3 = *(_DWORD *)(v2 + 256);
  v118 = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(v2 + 240);
    v6 = v3 - 1;
    v7 = a2;
    v8 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( v7 == *v9 )
    {
LABEL_3:
      v118 = v9[1];
    }
    else
    {
      v108 = 1;
      while ( v10 != -8 )
      {
        v109 = v108 + 1;
        v8 = v6 & (v108 + v8);
        v9 = (__int64 *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v108 = v109;
      }
      v118 = 0;
    }
  }
  if ( !*(_QWORD *)(a1 + 352) )
    *(_QWORD *)(a1 + 352) = sub_1E816F0(*(_QWORD *)(a1 + 344), 0);
  v163 = 0;
  v161[0] = (unsigned __int64)v162;
  v161[1] = 0x800000000LL;
  v164 = 0;
  v11 = *(_DWORD *)(*(_QWORD *)(a1 + 248) + 44LL);
  if ( v11 )
  {
    v12 = _libc_calloc(v11, 1u);
    if ( !v12 )
      sub_16BD1C0("Allocation failed", 1u);
    v163 = (unsigned __int64)v12;
    v164 = v11;
  }
  v120 = 0;
  v113 = 0;
  v110 = a2 + 24;
  if ( v123 == a2 + 24 )
    goto LABEL_163;
  while ( 2 )
  {
    v13 = v123;
    if ( !v123 )
      BUG();
    if ( (*(_BYTE *)v123 & 4) == 0 && (*(_BYTE *)(v123 + 46) & 8) != 0 )
    {
      do
        v13 = *(_QWORD *)(v13 + 8);
      while ( (*(_BYTE *)(v13 + 46) & 8) != 0 );
    }
    v14 = *(_QWORD *)(v13 + 8);
    v145 = (unsigned int *)v147;
    v115 = v14;
    v146 = 0x1000000000LL;
    v112 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned int **))(**(_QWORD **)(a1 + 240) + 440LL))(
             *(_QWORD *)(a1 + 240),
             v123,
             &v145);
    if ( !v112 )
    {
      v15 = (unsigned __int64)v145;
      if ( v145 != (unsigned int *)v147 )
        goto LABEL_17;
      goto LABEL_18;
    }
    if ( byte_4FC5AA0 )
    {
      v119 = v145;
      if ( &v145[(unsigned int)v146] == v145 )
        goto LABEL_69;
      v136 = &v145[(unsigned int)v146];
      v100 = v145;
      do
      {
        v103 = *(_QWORD *)(a1 + 240);
        v104 = *v100;
        v155 = v157;
        v158 = v160;
        v156 = 0x1000000000LL;
        v159 = 0x1000000000LL;
        v151 = 0;
        v152 = 0;
        v153 = 0;
        v154 = 0;
        (*(void (__fastcall **)(__int64, __int64, __int64, _BYTE **, _DWORD **, __int64 **))(*(_QWORD *)v103 + 472LL))(
          v103,
          v123,
          v104,
          &v155,
          &v158,
          &v151);
        if ( (_DWORD)v156
          && ((unsigned __int8)((__int64 (*)(void))sub_1F4B670)() || (unsigned __int8)sub_1F4B690(a1 + 360)) )
        {
          v101 = sub_1E84C70(*(_QWORD *)(a1 + 352), a2);
          sub_1DF1930(a1, v123, (__int64 *)&v155, (__int64)&v158, v101, v102);
        }
        j___libc_free_0(v152);
        if ( v158 != v160 )
          _libc_free((unsigned __int64)v158);
        if ( v155 != v157 )
          _libc_free((unsigned __int64)v155);
        ++v100;
      }
      while ( v136 != v100 );
    }
    v119 = &v145[(unsigned int)v146];
    if ( v145 == v119 )
      goto LABEL_69;
    v124 = v145;
    v114 = ((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4);
    while ( 1 )
    {
      v16 = *v124;
      v151 = &v153;
      v141 = 0;
      v148 = v150;
      v149 = 0x1000000000LL;
      v152 = 0x1000000000LL;
      v142 = 0;
      v17 = *(_QWORD *)(a1 + 240);
      v121 = v16;
      v143 = 0;
      v144 = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64, _BYTE **, __int64 **, __int64 *))(*(_QWORD *)v17 + 472LL))(
        v17,
        v123,
        v16,
        &v148,
        &v151,
        &v141);
      v21 = v149;
      if ( (_DWORD)v149 )
        break;
LABEL_132:
      v53 = v142;
LABEL_63:
      j___libc_free_0(v53);
      if ( v151 != &v153 )
        _libc_free((unsigned __int64)v151);
      if ( v148 != v150 )
        _libc_free((unsigned __int64)v148);
      if ( v119 == ++v124 )
      {
        v119 = v145;
        goto LABEL_69;
      }
    }
    v22 = v152;
    if ( v118 )
    {
      v90 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 240) + 448LL))(*(_QWORD *)(a1 + 240), v121);
      if ( !v120 )
        goto LABEL_124;
      goto LABEL_127;
    }
    if ( v120 )
    {
      v90 = 0;
LABEL_127:
      sub_1E82EE0(*(_QWORD *)(a1 + 352), v111, v115, v161);
      v111 = v115;
LABEL_124:
      if ( v90 )
        goto LABEL_109;
    }
    if ( v22 > v21 && *(_BYTE *)(a1 + 640)
      || (v128 = a1 + 360, !(unsigned __int8)sub_1F4B670(a1 + 360)) && !(unsigned __int8)sub_1F4B690(v128) )
    {
LABEL_109:
      v88 = v152;
      v89 = *(_QWORD *)(a1 + 352);
      v158 = v160;
      v159 = 0x1000000000LL;
      if ( (_DWORD)v152 )
        sub_1DF1B30((__int64)&v158, (__int64)&v151, v18, v19, v20, v152);
      v155 = v157;
      v156 = 0x1000000000LL;
      if ( (_DWORD)v149 )
        sub_1DF1B30((__int64)&v155, (__int64)&v148, v18, v19, v149, v88);
      sub_1DF1C10(a2, (__int64 *)v123, (__int64)&v155, (__int64)&v158, v89, (__int64)v161, v120);
      goto LABEL_114;
    }
    v117 = sub_1E84C70(*(_QWORD *)(a1 + 352), a2);
    v24 = v23;
    v116 = v23;
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 344) + 128LL))(*(_QWORD *)(a1 + 344));
    v158 = v160;
    v26 = (__int64)v148;
    v159 = 0x1000000000LL;
    v155 = (_BYTE *)v117;
    v156 = v24;
    v125 = v117;
    v127 = &v148[8 * (unsigned int)v149];
    if ( v148 == v127 )
    {
      v42 = v160[(unsigned int)(v149 - 1)];
      goto LABEL_49;
    }
    do
    {
      v27 = *(_QWORD *)v26;
      v28 = 0;
      v29 = *(_QWORD *)(*(_QWORD *)v26 + 32LL);
      v30 = v29 + 40LL * *(unsigned int *)(*(_QWORD *)v26 + 40LL);
      if ( v29 == v30 )
        goto LABEL_43;
      v129 = v26;
      do
      {
        if ( *(_BYTE *)v29 )
          goto LABEL_41;
        v31 = *(unsigned int *)(v29 + 8);
        if ( (int)v31 >= 0 || (*(_BYTE *)(v29 + 3) & 0x10) != 0 )
          goto LABEL_41;
        if ( !v144 )
          goto LABEL_73;
        v32 = (v144 - 1) & (37 * v31);
        v33 = &v142[2 * v32];
        v34 = *v33;
        if ( (_DWORD)v31 != *v33 )
        {
          v56 = 1;
          while ( v34 != -1 )
          {
            v67 = v56 + 1;
            v32 = (v144 - 1) & (v56 + v32);
            v33 = &v142[2 * v32];
            v34 = *v33;
            if ( (_DWORD)v31 == *v33 )
              goto LABEL_37;
            v56 = v67;
          }
LABEL_73:
          v57 = sub_1E69D60(*(_QWORD *)(a1 + 328));
          v58 = v57;
          if ( !v57 )
            goto LABEL_41;
          v59 = **(unsigned __int16 **)(v57 + 16);
          if ( !v59 || v59 == 45 )
            goto LABEL_41;
          v60 = *(_DWORD *)(v117 + 400);
          if ( v60 )
          {
            v61 = v60 - 1;
            v62 = *(_QWORD *)(v117 + 384);
            v63 = v61 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
            v64 = (__int64 *)(v62 + 16LL * v63);
            v65 = *v64;
            if ( *v64 == v58 )
            {
LABEL_78:
              v60 = *((_DWORD *)v64 + 2);
            }
            else
            {
              v98 = 1;
              while ( v65 != -8 )
              {
                v106 = v98 + 1;
                v63 = v61 & (v98 + v63);
                v64 = (__int64 *)(v62 + 16LL * v63);
                v65 = *v64;
                if ( v58 == *v64 )
                  goto LABEL_78;
                v98 = v106;
              }
              v60 = 0;
            }
          }
          v131 = v60;
          v133 = sub_1E165A0(v27, *(unsigned int *)(v29 + 8), 0, 0);
          v66 = sub_1E16810(v58, *(unsigned int *)(v29 + 8), 0, 0, 0);
          v38 = sub_1F4BB70(v128, v58, v66, v27, v133);
          v39 = v131;
          goto LABEL_39;
        }
LABEL_37:
        if ( v33 == &v142[2 * v144] )
          goto LABEL_73;
        v35 = (unsigned int)v33[1];
        v132 = *(_QWORD *)&v148[8 * v35];
        v130 = v158[v35];
        v36 = sub_1E16810(v132, v31, 0, 0, 0);
        v37 = sub_1E165A0(v27, *(unsigned int *)(v29 + 8), 0, 0);
        v38 = sub_1F4BB70(v128, v132, v36, v27, v37);
        v39 = v130;
LABEL_39:
        v40 = v38 + v39;
        if ( v28 < v40 )
          v28 = v40;
LABEL_41:
        v29 += 40;
      }
      while ( v30 != v29 );
      v26 = v129;
LABEL_43:
      v41 = (unsigned int)v159;
      if ( (unsigned int)v159 >= HIDWORD(v159) )
      {
        v134 = v26;
        sub_16CD150((__int64)&v158, v160, 0, 4, v25, v26);
        v41 = (unsigned int)v159;
        v26 = v134;
      }
      v26 += 8;
      v158[v41] = v28;
      LODWORD(v159) = v159 + 1;
    }
    while ( v127 != (_BYTE *)v26 );
    v42 = v158[(unsigned int)(v149 - 1)];
    if ( v158 != v160 )
      _libc_free((unsigned __int64)v158);
    v125 = (__int64)v155;
LABEL_49:
    v43 = *(_DWORD *)(v125 + 400);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(v125 + 384);
      v46 = v44 & v114;
      v47 = (__int64 *)(v45 + 16LL * (v44 & v114));
      v48 = *v47;
      if ( *v47 == v123 )
      {
LABEL_51:
        v43 = *((_DWORD *)v47 + 2);
      }
      else
      {
        v99 = 1;
        while ( v48 != -8 )
        {
          v107 = v99 + 1;
          v46 = v44 & (v99 + v46);
          v47 = (__int64 *)(v45 + 16LL * v46);
          v48 = *v47;
          if ( *v47 == v123 )
            goto LABEL_51;
          v99 = v107;
        }
        v43 = 0;
      }
    }
    if ( v121 > 3 )
    {
      v68 = sub_1DF1930(a1, v123, (__int64 *)&v148, (__int64)&v151, (__int64)v155, v156);
      v69 = v68 + v42;
      v70 = v43 + HIDWORD(v68);
      v71 = v43 + HIDWORD(v68) + sub_1E80C40(&v155, v123);
      if ( !v120 )
        v70 = v71;
      v49 = v69 <= v70;
      if ( v49 )
        goto LABEL_85;
LABEL_54:
      v50 = (__int64)v148;
      v51 = *(_QWORD *)(a2 + 56);
      v52 = &v148[8 * (unsigned int)v149];
      if ( v148 != v52 )
      {
        do
        {
          v50 += 8;
          sub_1E0A0F0(v51);
        }
        while ( v52 != (_BYTE *)v50 );
      }
      ++v141;
      v53 = v142;
      if ( (_DWORD)v143 )
      {
        v79 = 4 * v143;
        v54 = v144;
        if ( (unsigned int)(4 * v143) < 0x40 )
          v79 = 64;
        if ( v144 > v79 )
        {
          if ( (_DWORD)v143 == 1 )
          {
            v85 = 1024;
            v84 = 128;
          }
          else
          {
            _BitScanReverse(&v80, v143 - 1);
            v81 = (unsigned int)(1 << (33 - (v80 ^ 0x1F)));
            if ( (int)v81 < 64 )
              v81 = 64;
            if ( (_DWORD)v81 == v144 )
            {
              v143 = 0;
              v91 = &v142[2 * v81];
              do
              {
                if ( v53 )
                  *v53 = -1;
                v53 += 2;
              }
              while ( v91 != v53 );
              goto LABEL_132;
            }
            v82 = (4 * (int)v81 / 3u + 1) | ((unsigned __int64)(4 * (int)v81 / 3u + 1) >> 1);
            v83 = ((v82 | (v82 >> 2)) >> 4) | v82 | (v82 >> 2) | ((((v82 | (v82 >> 2)) >> 4) | v82 | (v82 >> 2)) >> 8);
            v84 = (v83 | (v83 >> 16)) + 1;
            v85 = 8 * ((v83 | (v83 >> 16)) + 1);
          }
          j___libc_free_0(v142);
          v144 = v84;
          v86 = (_DWORD *)sub_22077B0(v85);
          v143 = 0;
          v53 = v86;
          v142 = v86;
          for ( i = &v86[2 * v144]; i != v86; v86 += 2 )
          {
            if ( v86 )
              *v86 = -1;
          }
          goto LABEL_63;
        }
      }
      else
      {
        if ( !HIDWORD(v143) )
          goto LABEL_63;
        v54 = v144;
        if ( v144 > 0x40 )
        {
          j___libc_free_0(v142);
          v53 = 0;
          v142 = 0;
          v143 = 0;
          v144 = 0;
          goto LABEL_63;
        }
      }
      v55 = &v142[2 * v54];
      if ( v55 != v142 )
      {
        do
        {
          *v53 = -1;
          v53 += 2;
        }
        while ( v55 != v53 );
        v53 = v142;
      }
      v143 = 0;
      goto LABEL_63;
    }
    v49 = v42 < v43;
    if ( v42 >= v43 )
      goto LABEL_54;
LABEL_85:
    v137[0] = v117;
    v137[1] = v116;
    if ( (unsigned __int8)sub_1F4B670(v128) )
    {
      v140[0] = a2;
      v138 = v140;
      v139 = 0x100000001LL;
      v135 = sub_1E80E60((unsigned int)v137, (unsigned int)v140, 1, 0, 0, (unsigned int)v137, 0, 0);
      v156 = 0x1000000000LL;
      v158 = v160;
      v159 = 0x1000000000LL;
      v155 = v157;
      sub_1DF1870(a1, (__int64 *)&v148, (__int64)&v155, (__int64)v160, v92, v93);
      sub_1DF1870(a1, (__int64 *)&v151, (__int64)&v158, v94, v95, v96);
      v97 = sub_1E80E60(
              (unsigned int)v137,
              (_DWORD)v138,
              v139,
              (_DWORD)v155,
              v156,
              (unsigned int)v137,
              (__int64)v158,
              (unsigned int)v159);
      if ( v158 != v160 )
        _libc_free((unsigned __int64)v158);
      if ( v155 != v157 )
        _libc_free((unsigned __int64)v155);
      if ( v138 != v140 )
        _libc_free((unsigned __int64)v138);
      if ( v135 < v97 )
        goto LABEL_54;
    }
    v75 = 0;
    v76 = *(_QWORD *)(a2 + 32);
    if ( v76 == v110 )
      goto LABEL_171;
    do
    {
      v76 = *(_QWORD *)(v76 + 8);
      ++v75;
    }
    while ( v76 != v110 );
    if ( dword_4FC5C60 >= (unsigned int)v75 )
    {
LABEL_171:
      v77 = v120;
    }
    else
    {
      v120 = v49;
      v77 = 1;
      v111 = v115;
    }
    v78 = *(_QWORD *)(a1 + 352);
    v158 = v160;
    v159 = 0x1000000000LL;
    if ( (_DWORD)v152 )
      sub_1DF1B30((__int64)&v158, (__int64)&v151, v75, v72, v73, v74);
    v155 = v157;
    v156 = 0x1000000000LL;
    if ( (_DWORD)v149 )
      sub_1DF1B30((__int64)&v155, (__int64)&v148, v75, v72, v73, v74);
    sub_1DF1C10(a2, (__int64 *)v123, (__int64)&v155, (__int64)&v158, v78, (__int64)v161, v77);
LABEL_114:
    if ( v155 != v157 )
      _libc_free((unsigned __int64)v155);
    if ( v158 != v160 )
      _libc_free((unsigned __int64)v158);
    j___libc_free_0(v142);
    if ( v151 != &v153 )
      _libc_free((unsigned __int64)v151);
    if ( v148 != v150 )
      _libc_free((unsigned __int64)v148);
    v119 = v145;
    v113 = v112;
LABEL_69:
    if ( v119 != (unsigned int *)v147 )
    {
      v15 = (unsigned __int64)v119;
LABEL_17:
      _libc_free(v15);
    }
LABEL_18:
    if ( v110 != v115 )
    {
      v123 = v115;
      continue;
    }
    break;
  }
  if ( (v113 & (unsigned __int8)v120) != 0 )
  {
    sub_1E80B50(*(_QWORD *)(a1 + 344), a2);
    v113 &= v120;
  }
LABEL_163:
  _libc_free(v163);
  if ( (_BYTE *)v161[0] != v162 )
    _libc_free(v161[0]);
  return v113;
}
