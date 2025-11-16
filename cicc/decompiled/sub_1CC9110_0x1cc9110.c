// Function: sub_1CC9110
// Address: 0x1cc9110
//
__int64 __fastcall sub_1CC9110(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r15
  unsigned int v4; // ebx
  __int64 v5; // r14
  __int64 v6; // rdi
  char v7; // al
  _QWORD *v8; // r15
  __int64 i; // r14
  unsigned __int8 v10; // bl
  unsigned __int64 v11; // rsi
  char v12; // al
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *j; // rdx
  __int64 v20; // r12
  __int64 v21; // r10
  int v22; // r13d
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rbx
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // ecx
  int v33; // edx
  __int64 v34; // rdi
  _QWORD *v35; // rbx
  int *v36; // r12
  int *v37; // rax
  _QWORD *v38; // r14
  _QWORD *v39; // rdx
  int *v40; // r8
  __int64 v41; // rsi
  __int64 v42; // rcx
  int *v43; // rax
  _QWORD *v44; // rdi
  int *v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  int *v49; // rax
  int *v50; // r8
  __int64 v51; // rsi
  __int64 v52; // rcx
  _QWORD *v53; // rax
  _BYTE *v54; // rsi
  int *v55; // rsi
  unsigned __int64 v56; // r8
  unsigned __int64 *v57; // rdx
  _QWORD *v58; // rax
  _QWORD *v59; // rbx
  char v60; // bl
  _BYTE *v61; // rsi
  __int64 v63; // r15
  _QWORD *v64; // rdi
  int *v65; // rax
  __int64 v66; // r13
  __int64 v67; // r14
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  _QWORD *m; // rsi
  unsigned __int64 v71; // rdx
  char v72; // al
  int *v73; // rax
  int *v74; // r8
  __int64 v75; // rcx
  __int64 v76; // rdx
  __int64 *v77; // r9
  int v78; // ecx
  int v79; // ecx
  int v80; // ecx
  __int64 v81; // rdi
  __int64 *v82; // r8
  int v83; // r9d
  unsigned int v84; // r14d
  __int64 v85; // rsi
  unsigned int v86; // ecx
  _QWORD *v87; // rdi
  unsigned int v88; // eax
  int v89; // eax
  unsigned __int64 v90; // rax
  unsigned __int64 v91; // rax
  int v92; // ebx
  __int64 v93; // r12
  _QWORD *v94; // rax
  __int64 v95; // rdx
  _QWORD *k; // rdx
  __int64 *v97; // r13
  int *v98; // rax
  int *v99; // rsi
  __int64 v100; // rdi
  __int64 v101; // rcx
  int *v102; // rax
  int *v103; // r13
  int *v104; // rcx
  __int64 v105; // rdi
  __int64 v106; // rsi
  _QWORD *v107; // r10
  int *v108; // rsi
  __int64 v109; // rdi
  __int64 v110; // rcx
  __int64 v111; // rax
  unsigned __int64 v112; // r8
  unsigned __int64 *v113; // rdx
  __int64 v114; // r12
  _QWORD *v115; // r15
  __int64 v116; // rsi
  int v117; // r14d
  __int64 *v118; // r9
  _QWORD *v119; // rax
  int *v120; // [rsp+8h] [rbp-198h]
  _QWORD *v121; // [rsp+28h] [rbp-178h]
  __int64 v122; // [rsp+30h] [rbp-170h]
  __int64 v123; // [rsp+38h] [rbp-168h]
  _QWORD *v124; // [rsp+48h] [rbp-158h]
  __int64 v125; // [rsp+50h] [rbp-150h]
  unsigned __int8 v126; // [rsp+5Fh] [rbp-141h]
  __int64 v127; // [rsp+60h] [rbp-140h]
  char v128; // [rsp+60h] [rbp-140h]
  _QWORD *v129; // [rsp+60h] [rbp-140h]
  __int64 v132; // [rsp+70h] [rbp-130h]
  _QWORD *v133; // [rsp+70h] [rbp-130h]
  int v134; // [rsp+70h] [rbp-130h]
  __int64 v135; // [rsp+70h] [rbp-130h]
  __int64 v136; // [rsp+78h] [rbp-128h]
  _QWORD *v137; // [rsp+78h] [rbp-128h]
  __int64 v138; // [rsp+78h] [rbp-128h]
  char v139; // [rsp+8Fh] [rbp-111h] BYREF
  _QWORD *v140; // [rsp+90h] [rbp-110h] BYREF
  _QWORD *v141; // [rsp+98h] [rbp-108h] BYREF
  unsigned __int64 v142; // [rsp+A0h] [rbp-100h] BYREF
  unsigned __int64 *v143; // [rsp+A8h] [rbp-F8h] BYREF
  _QWORD *v144; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned __int64 **v145; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 v146; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v147; // [rsp+C8h] [rbp-D8h]
  _BYTE *v148; // [rsp+D0h] [rbp-D0h]
  __int64 v149; // [rsp+E0h] [rbp-C0h] BYREF
  int v150; // [rsp+E8h] [rbp-B8h] BYREF
  int *v151; // [rsp+F0h] [rbp-B0h]
  int *v152; // [rsp+F8h] [rbp-A8h]
  int *v153; // [rsp+100h] [rbp-A0h]
  __int64 v154; // [rsp+108h] [rbp-98h]
  __int64 v155; // [rsp+110h] [rbp-90h] BYREF
  int v156; // [rsp+118h] [rbp-88h] BYREF
  int *v157; // [rsp+120h] [rbp-80h]
  int *v158; // [rsp+128h] [rbp-78h]
  int *v159; // [rsp+130h] [rbp-70h]
  __int64 v160; // [rsp+138h] [rbp-68h]
  unsigned __int64 v161; // [rsp+140h] [rbp-60h] BYREF
  int v162; // [rsp+148h] [rbp-58h] BYREF
  int *v163; // [rsp+150h] [rbp-50h]
  int *v164; // [rsp+158h] [rbp-48h]
  int *v165; // [rsp+160h] [rbp-40h]
  __int64 v166; // [rsp+168h] [rbp-38h]

  sub_1CC6C70(a1[10]);
  a1[10] = 0;
  a1[11] = a1 + 9;
  a1[12] = a1 + 9;
  a1[13] = 0;
  v2 = *(_QWORD *)(a2 + 80);
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v136 = a2 + 72;
  if ( v2 == a2 + 72 )
    return 0;
  v3 = a1 + 8;
  do
  {
    if ( !v2 )
    {
      v161 = 0;
      BUG();
    }
    v4 = 0;
    v161 = v2 - 24;
    v5 = *(_QWORD *)(v2 + 24);
    if ( v5 != v2 + 16 )
    {
      do
      {
        v6 = v5 - 24;
        if ( !v5 )
          v6 = 0;
        v7 = sub_1C30710(v6);
        v5 = *(_QWORD *)(v5 + 8);
        v4 -= (v7 == 0) - 1;
      }
      while ( v2 + 16 != v5 );
      if ( v4 > 1 )
      {
        if ( SLODWORD(qword_4FBF2C0[20]) > 1 )
        {
          v54 = v147;
          if ( v147 == v148 )
          {
            sub_1292090((__int64)&v146, v147, &v161);
          }
          else
          {
            if ( v147 )
            {
              *(_QWORD *)v147 = v161;
              v54 = v147;
            }
            v147 = v54 + 8;
          }
        }
        sub_1444990(v3, &v161);
      }
    }
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v136 != v2 );
  v8 = a1;
  if ( !a1[13] )
  {
    v13 = v146;
    v126 = 0;
    v61 = &v148[-v146];
    goto LABEL_98;
  }
  v126 = 0;
  for ( i = *(_QWORD *)(a2 + 80); v2 != i; i = *(_QWORD *)(a2 + 80) )
  {
    v10 = 0;
    do
    {
      v11 = i - 24;
      if ( !i )
        v11 = 0;
      v12 = sub_1CC7510(a1, v11);
      i = *(_QWORD *)(i + 8);
      v10 |= v12;
    }
    while ( v2 != i );
    if ( !v10 )
      break;
    v126 = v10;
  }
  v13 = v146;
  if ( SLODWORD(qword_4FBF2C0[20]) <= 1 )
    goto LABEL_97;
  v14 = (__int64)&v147[-v146] >> 3;
  if ( !(_DWORD)v14 )
    goto LABEL_97;
  v125 = 0;
  v122 = (__int64)(a1 + 14);
  v123 = 8LL * (unsigned int)v14;
  do
  {
    v15 = *(_QWORD *)(v13 + v125);
    ++v8[14];
    v127 = v15;
    v16 = *((_DWORD *)v8 + 32);
    if ( v16 )
    {
      v86 = 4 * v16;
      v17 = *((unsigned int *)v8 + 34);
      if ( (unsigned int)(4 * v16) < 0x40 )
        v86 = 64;
      if ( (unsigned int)v17 <= v86 )
      {
LABEL_26:
        v18 = (_QWORD *)v8[15];
        for ( j = &v18[2 * v17]; j != v18; v18 += 2 )
          *v18 = -8;
        v8[16] = 0;
        goto LABEL_29;
      }
      v87 = (_QWORD *)v8[15];
      v88 = v16 - 1;
      if ( v88 )
      {
        _BitScanReverse(&v88, v88);
        v89 = 1 << (33 - (v88 ^ 0x1F));
        if ( v89 < 64 )
          v89 = 64;
        if ( (_DWORD)v17 == v89 )
        {
          v8[16] = 0;
          v119 = &v87[2 * (unsigned int)v17];
          do
          {
            if ( v87 )
              *v87 = -8;
            v87 += 2;
          }
          while ( v119 != v87 );
          goto LABEL_29;
        }
        v90 = (4 * v89 / 3u + 1) | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1);
        v91 = ((v90 | (v90 >> 2)) >> 4) | v90 | (v90 >> 2) | ((((v90 | (v90 >> 2)) >> 4) | v90 | (v90 >> 2)) >> 8);
        v92 = (v91 | (v91 >> 16)) + 1;
        v93 = 16 * ((v91 | (v91 >> 16)) + 1);
      }
      else
      {
        v93 = 2048;
        v92 = 128;
      }
      j___libc_free_0(v87);
      *((_DWORD *)v8 + 34) = v92;
      v94 = (_QWORD *)sub_22077B0(v93);
      v95 = *((unsigned int *)v8 + 34);
      v8[16] = 0;
      v8[15] = v94;
      for ( k = &v94[2 * v95]; k != v94; v94 += 2 )
      {
        if ( v94 )
          *v94 = -8;
      }
    }
    else if ( *((_DWORD *)v8 + 33) )
    {
      v17 = *((unsigned int *)v8 + 34);
      if ( (unsigned int)v17 <= 0x40 )
        goto LABEL_26;
      j___libc_free_0(v8[15]);
      v8[15] = 0;
      v8[16] = 0;
      *((_DWORD *)v8 + 34) = 0;
    }
LABEL_29:
    v20 = *(_QWORD *)(v127 + 48);
    v21 = v127 + 40;
    v137 = (_QWORD *)(v127 + 40);
    if ( v20 != v127 + 40 )
    {
      v22 = 0;
      while ( 1 )
      {
        v27 = *((_DWORD *)v8 + 34);
        v28 = v20 - 24;
        if ( !v20 )
          v28 = 0;
        ++v22;
        if ( !v27 )
          break;
        v23 = v8[15];
        v24 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( v28 == *v25 )
        {
LABEL_32:
          *((_DWORD *)v25 + 2) = v22;
          v20 = *(_QWORD *)(v20 + 8);
          if ( v21 == v20 )
            goto LABEL_42;
        }
        else
        {
          v134 = 1;
          v77 = 0;
          while ( v26 != -8 )
          {
            if ( v26 == -16 && !v77 )
              v77 = v25;
            v24 = (v27 - 1) & (v134 + v24);
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v28 == *v25 )
              goto LABEL_32;
            ++v134;
          }
          v78 = *((_DWORD *)v8 + 32);
          if ( v77 )
            v25 = v77;
          ++v8[14];
          v33 = v78 + 1;
          if ( 4 * (v78 + 1) < 3 * v27 )
          {
            if ( v27 - *((_DWORD *)v8 + 33) - v33 <= v27 >> 3 )
            {
              v135 = v21;
              sub_14672C0(v122, v27);
              v79 = *((_DWORD *)v8 + 34);
              if ( !v79 )
              {
LABEL_240:
                ++*((_DWORD *)v8 + 32);
                BUG();
              }
              v80 = v79 - 1;
              v81 = v8[15];
              v82 = 0;
              v83 = 1;
              v84 = v80 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
              v21 = v135;
              v33 = *((_DWORD *)v8 + 32) + 1;
              v25 = (__int64 *)(v81 + 16LL * v84);
              v85 = *v25;
              if ( v28 != *v25 )
              {
                while ( v85 != -8 )
                {
                  if ( v85 == -16 && !v82 )
                    v82 = v25;
                  v84 = v80 & (v83 + v84);
                  v25 = (__int64 *)(v81 + 16LL * v84);
                  v85 = *v25;
                  if ( v28 == *v25 )
                    goto LABEL_39;
                  ++v83;
                }
                if ( v82 )
                  v25 = v82;
              }
            }
            goto LABEL_39;
          }
LABEL_37:
          v132 = v21;
          sub_14672C0(v122, 2 * v27);
          v29 = *((_DWORD *)v8 + 34);
          if ( !v29 )
            goto LABEL_240;
          v30 = v29 - 1;
          v31 = v8[15];
          v21 = v132;
          v32 = v30 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v33 = *((_DWORD *)v8 + 32) + 1;
          v25 = (__int64 *)(v31 + 16LL * v32);
          v34 = *v25;
          if ( v28 != *v25 )
          {
            v117 = 1;
            v118 = 0;
            while ( v34 != -8 )
            {
              if ( !v118 && v34 == -16 )
                v118 = v25;
              v32 = v30 & (v117 + v32);
              v25 = (__int64 *)(v31 + 16LL * v32);
              v34 = *v25;
              if ( v28 == *v25 )
                goto LABEL_39;
              ++v117;
            }
            if ( v118 )
              v25 = v118;
          }
LABEL_39:
          *((_DWORD *)v8 + 32) = v33;
          if ( *v25 != -8 )
            --*((_DWORD *)v8 + 33);
          *((_DWORD *)v25 + 2) = 0;
          *v25 = v28;
          *((_DWORD *)v25 + 2) = v22;
          v20 = *(_QWORD *)(v20 + 8);
          if ( v21 == v20 )
          {
LABEL_42:
            v35 = *(_QWORD **)(v127 + 48);
            goto LABEL_43;
          }
        }
      }
      ++v8[14];
      goto LABEL_37;
    }
    v35 = (_QWORD *)(v127 + 40);
LABEL_43:
    v150 = 0;
    v36 = &v150;
    v158 = &v156;
    v151 = 0;
    v152 = &v150;
    v153 = &v150;
    v154 = 0;
    v156 = 0;
    v157 = 0;
    v159 = &v156;
    v160 = 0;
    v162 = 0;
    v163 = 0;
    v164 = &v162;
    v165 = &v162;
    v166 = 0;
    v140 = 0;
    if ( v35 != v137 )
    {
      v37 = 0;
      v38 = 0;
      v133 = 0;
      while ( 1 )
      {
        v39 = v35 - 3;
        v40 = &v150;
        if ( !v35 )
          v39 = 0;
        v144 = v39;
        if ( !v37 )
          goto LABEL_54;
        do
        {
          while ( 1 )
          {
            v41 = *((_QWORD *)v37 + 2);
            v42 = *((_QWORD *)v37 + 3);
            if ( *((_QWORD *)v37 + 4) >= (unsigned __int64)v39 )
              break;
            v37 = (int *)*((_QWORD *)v37 + 3);
            if ( !v42 )
              goto LABEL_52;
          }
          v40 = v37;
          v37 = (int *)*((_QWORD *)v37 + 2);
        }
        while ( v41 );
LABEL_52:
        if ( v40 == &v150 || *((_QWORD *)v40 + 4) > (unsigned __int64)v39 )
        {
LABEL_54:
          v145 = &v144;
          v40 = (int *)sub_1CC72C0(&v149, v40, (unsigned __int64 **)&v145);
        }
        v43 = v157;
        *((_QWORD *)v40 + 5) = v38;
        if ( !v43 )
          break;
        v44 = v144;
        v45 = &v156;
        do
        {
          while ( 1 )
          {
            v46 = *((_QWORD *)v43 + 2);
            v47 = *((_QWORD *)v43 + 3);
            if ( *((_QWORD *)v43 + 4) >= (unsigned __int64)v144 )
              break;
            v43 = (int *)*((_QWORD *)v43 + 3);
            if ( !v47 )
              goto LABEL_60;
          }
          v45 = v43;
          v43 = (int *)*((_QWORD *)v43 + 2);
        }
        while ( v46 );
LABEL_60:
        if ( v45 == &v156 || *((_QWORD *)v45 + 4) > (unsigned __int64)v144 )
          goto LABEL_62;
LABEL_63:
        *((_QWORD *)v45 + 5) = v140;
        if ( (unsigned __int8)sub_1C30710((__int64)v44) )
          v38 = v144;
        if ( (unsigned __int8)sub_15F3040((__int64)v144) )
        {
          if ( v140 )
          {
            v49 = v163;
            v50 = &v162;
            if ( !v163 )
              goto LABEL_74;
            do
            {
              while ( 1 )
              {
                v51 = *((_QWORD *)v49 + 2);
                v52 = *((_QWORD *)v49 + 3);
                if ( *((_QWORD *)v49 + 4) >= (unsigned __int64)v140 )
                  break;
                v49 = (int *)*((_QWORD *)v49 + 3);
                if ( !v52 )
                  goto LABEL_72;
              }
              v50 = v49;
              v49 = (int *)*((_QWORD *)v49 + 2);
            }
            while ( v51 );
LABEL_72:
            if ( v50 == &v162 || *((_QWORD *)v50 + 4) > (unsigned __int64)v140 )
            {
LABEL_74:
              v145 = &v140;
              v50 = (int *)sub_1CC72C0(&v161, v50, (unsigned __int64 **)&v145);
            }
            v53 = v144;
            *((_QWORD *)v50 + 5) = v144;
          }
          else
          {
            v53 = v144;
            v133 = v144;
          }
          v140 = v53;
        }
        v35 = (_QWORD *)v35[1];
        if ( v35 == v137 )
        {
          v137 = *(_QWORD **)(v127 + 48);
          goto LABEL_103;
        }
        v37 = v151;
      }
      v45 = &v156;
LABEL_62:
      v145 = &v144;
      v48 = sub_1CC72C0(&v155, v45, (unsigned __int64 **)&v145);
      v44 = v144;
      v45 = (int *)v48;
      goto LABEL_63;
    }
    v133 = 0;
    v38 = 0;
LABEL_103:
    v139 = 0;
    v59 = (_QWORD *)(*(_QWORD *)(v127 + 40) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v59 == v137 )
    {
      sub_1CC6E40((__int64)v163);
      sub_1CC6E40((__int64)v157);
      sub_1CC6E40((__int64)v151);
      goto LABEL_96;
    }
    v124 = v38;
    v138 = (__int64)v8;
    v63 = v127;
    while ( 2 )
    {
      v64 = v59 - 3;
      if ( !v59 )
        v64 = 0;
      v141 = v64;
      if ( (unsigned __int8)sub_1C30710((__int64)v64) )
        goto LABEL_90;
      v65 = v151;
      if ( v151 )
      {
        v55 = v36;
        do
        {
          if ( *((_QWORD *)v65 + 4) < (unsigned __int64)v141 )
          {
            v65 = (int *)*((_QWORD *)v65 + 3);
          }
          else
          {
            v55 = v65;
            v65 = (int *)*((_QWORD *)v65 + 2);
          }
        }
        while ( v65 );
        if ( v55 != v36 && *((_QWORD *)v55 + 4) <= (unsigned __int64)v141 )
          goto LABEL_86;
      }
      else
      {
        v55 = v36;
      }
      v145 = &v141;
      v55 = (int *)sub_1CC72C0(&v149, v55, (unsigned __int64 **)&v145);
LABEL_86:
      v56 = *((_QWORD *)v55 + 5);
      if ( v56 )
      {
        v57 = *(unsigned __int64 **)(v56 + 32);
        if ( v57 )
          v57 -= 3;
        sub_1CC8CA0(v138, (__int64)v141, v57, &v139, v56, v63, (__int64)v133, &v149, &v155, &v161);
        goto LABEL_90;
      }
      if ( SLODWORD(qword_4FBF2C0[20]) <= 2 || *((_BYTE *)v141 + 16) == 77 )
      {
LABEL_92:
        v8 = (_QWORD *)v138;
        v128 = v139;
        goto LABEL_93;
      }
      if ( (unsigned __int8)sub_15F3040((__int64)v141) || !v141[1] )
        goto LABEL_90;
      v129 = v141;
      v66 = v141[1];
      v67 = v141[5];
      do
      {
        v68 = sub_1648700(v66);
        if ( *((_BYTE *)v68 + 16) > 0x17u && v67 == v68[5] )
          goto LABEL_90;
        v66 = *(_QWORD *)(v66 + 8);
      }
      while ( v66 );
      v69 = v124;
      v143 = 0;
      v142 = (unsigned __int64)v124;
      if ( !v124 )
        goto LABEL_90;
      for ( m = v129; ; m = v141 )
      {
        v71 = v69[4];
        if ( v71 )
          v71 -= 24LL;
        v143 = (unsigned __int64 *)v71;
        v72 = sub_1CC8920(v138, (__int64)m, v71, (__int64)v133, &v155, &v161);
        if ( v72 )
          break;
        v73 = v151;
        if ( !v151 )
        {
          v74 = v36;
LABEL_134:
          v145 = (unsigned __int64 **)&v142;
          v74 = (int *)sub_1CC72C0(&v149, v74, (unsigned __int64 **)&v145);
          goto LABEL_135;
        }
        v74 = v36;
        do
        {
          while ( 1 )
          {
            v75 = *((_QWORD *)v73 + 2);
            v76 = *((_QWORD *)v73 + 3);
            if ( *((_QWORD *)v73 + 4) >= v142 )
              break;
            v73 = (int *)*((_QWORD *)v73 + 3);
            if ( !v76 )
              goto LABEL_132;
          }
          v74 = v73;
          v73 = (int *)*((_QWORD *)v73 + 2);
        }
        while ( v75 );
LABEL_132:
        if ( v74 == v36 || *((_QWORD *)v74 + 4) > v142 )
          goto LABEL_134;
LABEL_135:
        v69 = (_QWORD *)*((_QWORD *)v74 + 5);
        v142 = (unsigned __int64)v69;
        if ( !v69 )
          goto LABEL_90;
      }
      v128 = v72;
      if ( !v142 )
      {
LABEL_90:
        v58 = *(_QWORD **)(v63 + 48);
LABEL_91:
        v59 = (_QWORD *)(*v59 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v59 == v58 )
          goto LABEL_92;
        continue;
      }
      break;
    }
    v59 = (_QWORD *)v59[1];
    sub_15F22F0(v141, (__int64)v143);
    v144 = v141;
    v145 = (unsigned __int64 **)v143;
    v97 = sub_1467480(v122, (__int64 *)&v145);
    *((_DWORD *)sub_1467480(v122, (__int64 *)&v144) + 2) = *((_DWORD *)v97 + 2);
    v98 = v151;
    if ( !v151 )
    {
      v99 = v36;
      goto LABEL_177;
    }
    v99 = v36;
    do
    {
      while ( 1 )
      {
        v100 = *((_QWORD *)v98 + 2);
        v101 = *((_QWORD *)v98 + 3);
        if ( *((_QWORD *)v98 + 4) >= (unsigned __int64)v141 )
          break;
        v98 = (int *)*((_QWORD *)v98 + 3);
        if ( !v101 )
          goto LABEL_175;
      }
      v99 = v98;
      v98 = (int *)*((_QWORD *)v98 + 2);
    }
    while ( v100 );
LABEL_175:
    if ( v99 == v36 || *((_QWORD *)v99 + 4) > (unsigned __int64)v141 )
    {
LABEL_177:
      v145 = &v141;
      v99 = (int *)sub_1CC72C0(&v149, v99, (unsigned __int64 **)&v145);
    }
    *((_QWORD *)v99 + 5) = v142;
    v102 = v157;
    if ( !v157 )
    {
      v103 = &v156;
      goto LABEL_185;
    }
    v103 = &v156;
    v104 = v157;
    do
    {
      while ( 1 )
      {
        v105 = *((_QWORD *)v104 + 2);
        v106 = *((_QWORD *)v104 + 3);
        if ( *((_QWORD *)v104 + 4) >= (unsigned __int64)v143 )
          break;
        v104 = (int *)*((_QWORD *)v104 + 3);
        if ( !v106 )
          goto LABEL_183;
      }
      v103 = v104;
      v104 = (int *)*((_QWORD *)v104 + 2);
    }
    while ( v105 );
LABEL_183:
    if ( v103 == &v156 || *((_QWORD *)v103 + 4) > (unsigned __int64)v143 )
    {
LABEL_185:
      v145 = &v143;
      v103 = (int *)sub_1CC72C0(&v155, v103, (unsigned __int64 **)&v145);
      v102 = v157;
      if ( v157 )
        goto LABEL_186;
      v108 = &v156;
LABEL_192:
      v145 = &v141;
      v111 = sub_1CC72C0(&v155, v108, (unsigned __int64 **)&v145);
      v107 = v141;
      v108 = (int *)v111;
    }
    else
    {
LABEL_186:
      v107 = v141;
      v108 = &v156;
      do
      {
        while ( 1 )
        {
          v109 = *((_QWORD *)v102 + 2);
          v110 = *((_QWORD *)v102 + 3);
          if ( *((_QWORD *)v102 + 4) >= (unsigned __int64)v141 )
            break;
          v102 = (int *)*((_QWORD *)v102 + 3);
          if ( !v110 )
            goto LABEL_190;
        }
        v108 = v102;
        v102 = (int *)*((_QWORD *)v102 + 2);
      }
      while ( v109 );
LABEL_190:
      if ( v108 == &v156 || *((_QWORD *)v108 + 4) > (unsigned __int64)v141 )
        goto LABEL_192;
    }
    v112 = v142;
    *((_QWORD *)v108 + 5) = *((_QWORD *)v103 + 5);
    if ( v107 )
    {
      v107 += 3;
      if ( !v112 )
      {
        v121 = 0;
        goto LABEL_196;
      }
      v121 = (_QWORD *)(v112 + 24);
      if ( v107 != (_QWORD *)(v112 + 24) )
      {
LABEL_196:
        v113 = v143;
        v120 = v36;
        v114 = v63;
        v115 = v107;
        while ( 1 )
        {
          v116 = (__int64)(v115 - 3);
          if ( !v115 )
            v116 = 0;
          v143 = sub_1CC8CA0(v138, v116, v113, &v139, v112, v114, (__int64)v133, &v149, &v155, &v161);
          v113 = v143;
          v115 = (_QWORD *)(*v115 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v115 == v121 )
            break;
          v112 = v142;
        }
        v63 = v114;
        v36 = v120;
      }
    }
    else
    {
      v121 = (_QWORD *)(v112 + 24);
      if ( v112 )
        goto LABEL_196;
    }
    v58 = *(_QWORD **)(v63 + 48);
    v139 = 1;
    if ( v59 != v58 )
      goto LABEL_91;
    v8 = (_QWORD *)v138;
LABEL_93:
    sub_1CC6E40((__int64)v163);
    sub_1CC6E40((__int64)v157);
    sub_1CC6E40((__int64)v151);
    v60 = v126;
    if ( v128 )
      v60 = v128;
    v126 = v60;
LABEL_96:
    v125 += 8;
    v13 = v146;
  }
  while ( v123 != v125 );
LABEL_97:
  v61 = &v148[-v13];
LABEL_98:
  if ( v13 )
    j_j___libc_free_0(v13, v61);
  return v126;
}
