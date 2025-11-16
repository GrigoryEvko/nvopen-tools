// Function: sub_317BE50
// Address: 0x317be50
//
__int64 __fastcall sub_317BE50(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 j; // r8
  __int64 v7; // r9
  char v8; // r14
  __int64 v9; // rsi
  int v10; // r11d
  __int64 v11; // r10
  unsigned int v12; // edx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // ebx
  __int64 *v17; // rax
  unsigned __int64 v18; // r14
  __int64 v19; // r13
  _BYTE *v20; // rsi
  __int64 *v21; // rdx
  _BYTE *v22; // rax
  _BYTE *v23; // rcx
  __int64 *v24; // rax
  unsigned int v25; // esi
  __int64 v26; // r13
  __int64 v27; // r12
  unsigned int v28; // esi
  _BYTE *v29; // rbx
  __int64 v30; // r9
  _QWORD *v31; // rdx
  int v32; // r11d
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // r8
  _DWORD *v36; // rdx
  int v37; // eax
  __int64 v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // r14
  __int64 v41; // rax
  bool v42; // zf
  __int64 v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 *v48; // rax
  unsigned __int8 *v49; // rsi
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 *v55; // r13
  __int64 v56; // r12
  __int64 *v57; // rbx
  __int64 *v58; // r12
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // rdi
  __int64 v63; // rax
  unsigned int v64; // esi
  __int64 *v65; // rdx
  __int64 v66; // r10
  unsigned __int8 *v67; // rdi
  __int64 v68; // rbx
  char v69; // al
  __int64 *v70; // rdi
  __int64 v71; // r14
  int v72; // eax
  unsigned __int64 v73; // rax
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 *v76; // r13
  __int64 v77; // r12
  __int64 *v78; // rbx
  unsigned int v79; // r13d
  __int64 v80; // rbx
  unsigned __int64 v81; // r12
  unsigned __int64 v82; // rdi
  unsigned __int64 v83; // rdi
  int v85; // r10d
  __int64 *v86; // rdx
  unsigned int v87; // edi
  __int64 v88; // rax
  __int64 v89; // rcx
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 *v92; // rax
  int v93; // eax
  int v94; // eax
  __int64 v95; // rax
  int v96; // ecx
  int v97; // edx
  unsigned int v98; // esi
  int v99; // r10d
  __int64 v100; // r8
  __int64 *v101; // rdi
  unsigned int v102; // ecx
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 *v105; // rax
  int v106; // eax
  int v107; // r8d
  __int64 v108; // rsi
  __int64 v109; // rax
  int v110; // edx
  __int64 v111; // rcx
  int v112; // r10d
  __int64 *v113; // r9
  int v114; // eax
  int v115; // eax
  int v116; // eax
  __int64 v117; // rdx
  __int64 v118; // rdi
  unsigned __int64 v119; // rcx
  __int64 v120; // r8
  __int64 *v121; // r9
  __int64 v122; // rbx
  _QWORD *v123; // r14
  __int64 *v124; // rax
  int v125; // ecx
  int v126; // ecx
  __int64 v127; // r11
  unsigned int v128; // esi
  __int64 v129; // r8
  int v130; // r10d
  _QWORD *v131; // r9
  __int64 v132; // rax
  __int64 v133; // rdi
  int v134; // r10d
  __int64 *v135; // r9
  int v136; // eax
  int v137; // eax
  __int64 v138; // rsi
  int v139; // r9d
  __int64 *v140; // r8
  __int64 v141; // r14
  __int64 v142; // rcx
  int v143; // ecx
  int v144; // ecx
  __int64 v145; // r11
  int v146; // r10d
  unsigned int v147; // esi
  __int64 v148; // r8
  __int64 *v149; // r8
  __int64 v150; // r14
  int v151; // r9d
  __int64 v152; // rsi
  __int64 v153; // rdi
  unsigned int v154; // r15d
  __int64 v155; // rdi
  _DWORD *v156; // rdx
  unsigned int v157; // ecx
  __int64 i; // r14
  int v159; // ecx
  int v160; // ecx
  __int64 v161; // r9
  int v162; // edi
  int v163; // edx
  __int64 v164; // rcx
  int v165; // r10d
  __int64 v166; // rdi
  int v167; // ecx
  unsigned int v168; // r9d
  __int64 v169; // [rsp+10h] [rbp-1840h]
  char v170; // [rsp+1Fh] [rbp-1831h]
  int v171; // [rsp+30h] [rbp-1820h]
  __int64 v172; // [rsp+30h] [rbp-1820h]
  __int64 v173; // [rsp+30h] [rbp-1820h]
  unsigned int v174; // [rsp+40h] [rbp-1810h]
  unsigned int v175; // [rsp+40h] [rbp-1810h]
  __int64 v176; // [rsp+48h] [rbp-1808h]
  __int64 *v177; // [rsp+48h] [rbp-1808h]
  __int64 v178; // [rsp+48h] [rbp-1808h]
  __int64 v179; // [rsp+50h] [rbp-1800h] BYREF
  __int64 v180; // [rsp+58h] [rbp-17F8h]
  __int64 v181; // [rsp+60h] [rbp-17F0h]
  unsigned int v182; // [rsp+68h] [rbp-17E8h]
  _BYTE *v183; // [rsp+70h] [rbp-17E0h] BYREF
  __int64 v184; // [rsp+78h] [rbp-17D8h]
  _BYTE v185[48]; // [rsp+80h] [rbp-17D0h] BYREF
  __int64 *v186; // [rsp+B0h] [rbp-17A0h] BYREF
  __int64 v187; // [rsp+B8h] [rbp-1798h]
  _BYTE v188[48]; // [rsp+C0h] [rbp-1790h] BYREF
  __int64 *v189; // [rsp+F0h] [rbp-1760h] BYREF
  __int64 *v190; // [rsp+F8h] [rbp-1758h]
  __int64 v191; // [rsp+100h] [rbp-1750h]
  int v192; // [rsp+108h] [rbp-1748h]
  char v193; // [rsp+10Ch] [rbp-1744h]
  _BYTE v194[256]; // [rsp+110h] [rbp-1740h] BYREF
  _BYTE *v195; // [rsp+210h] [rbp-1640h] BYREF
  __int64 v196; // [rsp+218h] [rbp-1638h]
  _BYTE v197[5680]; // [rsp+220h] [rbp-1630h] BYREF

  v195 = v197;
  v196 = 0x2000000000LL;
  v1 = *(_QWORD *)(a1 + 8);
  v179 = 0;
  v2 = *(_QWORD *)(v1 + 32);
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v176 = v1 + 24;
  if ( v2 == v1 + 24 )
  {
    v79 = 0;
    goto LABEL_100;
  }
  v171 = 0;
  v3 = a1;
  do
  {
    v4 = v2 - 56;
    if ( !v2 )
      v4 = 0;
    v8 = sub_31764A0(v3, v4);
    if ( !v8 )
      goto LABEL_16;
    v9 = *(unsigned int *)(v3 + 752);
    j = v3 + 728;
    if ( (_DWORD)v9 )
    {
      v5 = *(_QWORD *)(v3 + 736);
      v10 = 1;
      v11 = 0;
      v7 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      v12 = (v9 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v13 = v5 + 144LL * v12;
      v14 = *(_QWORD *)v13;
      if ( v4 == *(_QWORD *)v13 )
      {
LABEL_8:
        v8 = 0;
        goto LABEL_9;
      }
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v11 )
          v11 = v13;
        v12 = (v9 - 1) & (v10 + v12);
        v13 = v5 + 144LL * v12;
        v14 = *(_QWORD *)v13;
        if ( v4 == *(_QWORD *)v13 )
          goto LABEL_8;
        ++v10;
      }
      v115 = *(_DWORD *)(v3 + 744);
      if ( v11 )
        v13 = v11;
      ++*(_QWORD *)(v3 + 728);
      v116 = v115 + 1;
      if ( 4 * v116 < (unsigned int)(3 * v9) )
      {
        v117 = (unsigned int)(v9 - *(_DWORD *)(v3 + 748) - v116);
        if ( (unsigned int)v117 <= (unsigned int)v9 >> 3 )
        {
          v175 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
          sub_3178690(v3 + 728, v9);
          v163 = *(_DWORD *)(v3 + 752);
          if ( !v163 )
            goto LABEL_319;
          v117 = (unsigned int)(v163 - 1);
          v9 = 1;
          v164 = 0;
          j = *(_QWORD *)(v3 + 736);
          v165 = v117 & v175;
          v13 = j + 144LL * ((unsigned int)v117 & v175);
          v166 = *(_QWORD *)v13;
          v116 = *(_DWORD *)(v3 + 744) + 1;
          if ( v4 != *(_QWORD *)v13 )
          {
            while ( v166 != -4096 )
            {
              if ( !v164 && v166 == -8192 )
                v164 = v13;
              v168 = v9 + 1;
              v9 = (unsigned int)v117 & (v165 + (_DWORD)v9);
              v165 = v9;
              v13 = j + 144LL * (unsigned int)v9;
              v166 = *(_QWORD *)v13;
              if ( v4 == *(_QWORD *)v13 )
                goto LABEL_204;
              v9 = v168;
            }
            if ( v164 )
              v13 = v164;
          }
        }
        goto LABEL_204;
      }
    }
    else
    {
      ++*(_QWORD *)(v3 + 728);
    }
    v9 = (unsigned int)(2 * v9);
    sub_3178690(v3 + 728, v9);
    v159 = *(_DWORD *)(v3 + 752);
    if ( !v159 )
      goto LABEL_319;
    v160 = v159 - 1;
    v161 = *(_QWORD *)(v3 + 736);
    v117 = v160 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v13 = v161 + 144 * v117;
    j = *(_QWORD *)v13;
    v116 = *(_DWORD *)(v3 + 744) + 1;
    if ( v4 != *(_QWORD *)v13 )
    {
      v162 = 1;
      v9 = 0;
      while ( j != -4096 )
      {
        if ( j == -8192 && !v9 )
          v9 = v13;
        v117 = v160 & (unsigned int)(v117 + v162);
        v13 = v161 + 144 * v117;
        j = *(_QWORD *)v13;
        if ( v4 == *(_QWORD *)v13 )
          goto LABEL_204;
        ++v162;
      }
      if ( v9 )
        v13 = v9;
    }
LABEL_204:
    *(_DWORD *)(v3 + 744) = v116;
    if ( *(_QWORD *)v13 != -4096 )
      --*(_DWORD *)(v3 + 748);
    *(_QWORD *)v13 = v4;
    memset((void *)(v13 + 8), 0, 0x88u);
    v118 = v13 + 144;
    v189 = 0;
    v42 = *(_QWORD *)(v3 + 136) == 0;
    v190 = (__int64 *)v194;
    v191 = 32;
    v192 = 0;
    v193 = 1;
    if ( v42 )
LABEL_318:
      sub_4263D6(v118, v9, v117);
    v118 = v4;
    v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(v3 + 144))(v3 + 120, v4, v117, 0, j);
    sub_30AB9D0(v4, v9, &v189, v119, v120, v121);
    if ( *(_QWORD *)(v4 + 80) != v4 + 72 )
    {
      v170 = v8;
      v169 = v2;
      v122 = *(_QWORD *)(v4 + 80);
      do
      {
        v123 = 0;
        if ( v122 )
          v123 = (_QWORD *)(v122 - 24);
        if ( !*(_QWORD *)(v3 + 104) )
          goto LABEL_318;
        v124 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(v3 + 112))(v3 + 88, v4);
        v9 = (__int64)v123;
        v118 = v13 + 8;
        sub_30ABD80(v13 + 8, v123, v124, (__int64)&v189, 0, 0);
        v122 = *(_QWORD *)(v122 + 8);
      }
      while ( v122 != v4 + 72 );
      v8 = v170;
      v2 = v169;
    }
    if ( !v193 )
      _libc_free((unsigned __int64)v190);
LABEL_9:
    if ( (_BYTE)qword_5034E48 )
      goto LABEL_102;
    if ( !(_BYTE)qword_50344A8 )
    {
      if ( (unsigned __int8)sub_B2D610(v4, 31) )
      {
LABEL_102:
        if ( *(_BYTE *)(v13 + 10) )
          goto LABEL_16;
        v5 = *(unsigned int *)(v13 + 32);
        if ( (_DWORD)v5 )
          goto LABEL_16;
        if ( (_BYTE)qword_50344A8 )
          goto LABEL_115;
      }
      else
      {
        if ( *(_BYTE *)(v13 + 10) || *(_DWORD *)(v13 + 32) || (unsigned int)qword_50349E8 > *(__int64 *)(v13 + 24) )
          goto LABEL_16;
        if ( (_BYTE)qword_50344A8 )
        {
LABEL_115:
          v15 = *(_QWORD *)(v13 + 24);
          if ( !v8 )
            goto LABEL_15;
          goto LABEL_107;
        }
      }
      if ( v8 )
      {
        v15 = *(unsigned int *)(v13 + 24);
        goto LABEL_107;
      }
      if ( *(_BYTE *)(v13 + 9) )
      {
        v15 = *(unsigned int *)(v13 + 24);
        goto LABEL_15;
      }
      goto LABEL_16;
    }
    if ( !*(_BYTE *)(v13 + 10) && !*(_DWORD *)(v13 + 32) )
    {
      v15 = *(_QWORD *)(v13 + 24);
      if ( v15 >= (unsigned int)qword_50349E8 )
      {
        if ( !v8 )
        {
LABEL_15:
          v171 -= ((unsigned __int8)sub_317AF40(v3, v4, v15, (__int64)&v195, (__int64)&v179) == 0) - 1;
          goto LABEL_16;
        }
LABEL_107:
        if ( *(_BYTE *)(v13 + 9) )
        {
          v174 = v15;
          sub_3177130(v3, v4);
          v15 = v174;
        }
        goto LABEL_15;
      }
    }
LABEL_16:
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v176 != v2 );
  if ( v171 )
  {
    v184 = 0xC00000000LL;
    v16 = qword_5034D68 * v171;
    if ( (int)qword_5034D68 * v171 > (unsigned int)v196 )
      v16 = v196;
    v17 = (__int64 *)v185;
    v18 = v16 + 1;
    v183 = v185;
    v19 = 4LL * v16;
    if ( v16 == -1 )
    {
      v20 = v185;
      v22 = &v185[v19];
      goto LABEL_29;
    }
    v20 = v185;
    if ( v18 > 0xC )
    {
      sub_C8D5F0((__int64)&v183, v185, v16 + 1, 4u, j, v7);
      v20 = v183;
      v17 = (__int64 *)&v183[4 * (unsigned int)v184];
    }
    v21 = (__int64 *)&v20[4 * v18];
    if ( v21 != v17 )
    {
      do
      {
        if ( v17 )
          *(_DWORD *)v17 = 0;
        v17 = (__int64 *)((char *)v17 + 4);
      }
      while ( v21 != v17 );
      v20 = v183;
    }
    v22 = &v20[v19];
    LODWORD(v184) = v16 + 1;
    if ( &v20[v19] != v20 )
    {
LABEL_29:
      v23 = v22 - 4;
      v24 = 0;
      v5 = (unsigned __int64)(v23 - v20) >> 2;
      do
      {
        v21 = v24;
        *(_DWORD *)&v20[4 * (_QWORD)v24] = (_DWORD)v24;
        v24 = (__int64 *)((char *)v24 + 1);
      }
      while ( (__int64 *)v5 != v21 );
    }
    v25 = v196;
    if ( (unsigned int)v196 <= v16 )
      goto LABEL_32;
    v153 = (__int64)v183;
    if ( (unsigned __int64)v19 <= 4 )
    {
LABEL_254:
      v173 = v3;
      v154 = v16;
      while ( 1 )
      {
        *(_DWORD *)(v153 + 4LL * v16) = v154;
        v189 = (__int64 *)&v195;
        sub_3174DD0(
          (__int64)v183,
          ((4LL * (unsigned int)v184) >> 2) - 1,
          0,
          *(_DWORD *)&v183[4 * (unsigned int)v184 - 4],
          &v189);
        v5 = (unsigned int)v184;
        v21 = (__int64 *)(4LL * (unsigned int)v184);
        if ( (unsigned int)v184 <= 1uLL )
        {
          if ( ++v154 >= v25 )
            goto LABEL_259;
        }
        else
        {
          v155 = (__int64)v183;
          ++v154;
          v156 = (_DWORD *)((char *)v21 + (_QWORD)v183);
          v157 = *--v156;
          *v156 = *(_DWORD *)v183;
          sub_3174E80(v155, 0, ((__int64)v156 - v155) >> 2, v157, (__int64 *)&v195);
          if ( v154 >= v25 )
          {
LABEL_259:
            v3 = v173;
            goto LABEL_32;
          }
        }
        v153 = (__int64)v183;
      }
    }
    for ( i = ((v19 >> 2) - 2) / 2; ; --i )
    {
      sub_3174E80(v153, i, v19 >> 2, *(_DWORD *)(v153 + 4 * i), (__int64 *)&v195);
      if ( !i )
        break;
    }
    v25 = v196;
    if ( v16 < (unsigned int)v196 )
    {
      v153 = (__int64)v183;
      goto LABEL_254;
    }
LABEL_32:
    v189 = 0;
    v190 = (__int64 *)v194;
    v186 = (__int64 *)v188;
    v191 = 8;
    v192 = 0;
    v193 = 1;
    v187 = 0x600000000LL;
    if ( v16 )
    {
      v26 = 0;
      v172 = v3 + 760;
      v27 = 4LL * v16;
      while ( 1 )
      {
        v28 = *(_DWORD *)(v3 + 784);
        v29 = &v195[176 * *(unsigned int *)&v183[v26]];
        if ( v28 )
        {
          v30 = *(_QWORD *)(v3 + 768);
          v31 = 0;
          v32 = 1;
          v33 = (v28 - 1) & (((unsigned int)*(_QWORD *)v29 >> 9) ^ ((unsigned int)*(_QWORD *)v29 >> 4));
          v34 = (__int64 *)(v30 + 16LL * v33);
          v35 = *v34;
          if ( *(_QWORD *)v29 == *v34 )
          {
LABEL_36:
            v36 = v34 + 1;
            v37 = *((_DWORD *)v34 + 2);
            goto LABEL_37;
          }
          while ( v35 != -4096 )
          {
            if ( !v31 && v35 == -8192 )
              v31 = v34;
            v33 = (v28 - 1) & (v32 + v33);
            v34 = (__int64 *)(v30 + 16LL * v33);
            v35 = *v34;
            if ( *(_QWORD *)v29 == *v34 )
              goto LABEL_36;
            ++v32;
          }
          if ( !v31 )
            v31 = v34;
          v93 = *(_DWORD *)(v3 + 776);
          ++*(_QWORD *)(v3 + 760);
          v94 = v93 + 1;
          if ( 4 * v94 < 3 * v28 )
          {
            if ( v28 - *(_DWORD *)(v3 + 780) - v94 > v28 >> 3 )
              goto LABEL_135;
            sub_9E07A0(v172, v28);
            v143 = *(_DWORD *)(v3 + 784);
            if ( !v143 )
            {
LABEL_320:
              ++*(_DWORD *)(v3 + 776);
              BUG();
            }
            v144 = v143 - 1;
            v145 = *(_QWORD *)(v3 + 768);
            v131 = 0;
            v146 = 1;
            v147 = v144 & (((unsigned int)*(_QWORD *)v29 >> 9) ^ ((unsigned int)*(_QWORD *)v29 >> 4));
            v94 = *(_DWORD *)(v3 + 776) + 1;
            v31 = (_QWORD *)(v145 + 16LL * v147);
            v148 = *v31;
            if ( *(_QWORD *)v29 == *v31 )
              goto LABEL_135;
            while ( v148 != -4096 )
            {
              if ( v148 == -8192 && !v131 )
                v131 = v31;
              v147 = v144 & (v146 + v147);
              v31 = (_QWORD *)(v145 + 16LL * v147);
              v148 = *v31;
              if ( *(_QWORD *)v29 == *v31 )
                goto LABEL_135;
              ++v146;
            }
            goto LABEL_221;
          }
        }
        else
        {
          ++*(_QWORD *)(v3 + 760);
        }
        sub_9E07A0(v172, 2 * v28);
        v125 = *(_DWORD *)(v3 + 784);
        if ( !v125 )
          goto LABEL_320;
        v126 = v125 - 1;
        v127 = *(_QWORD *)(v3 + 768);
        v128 = v126 & (((unsigned int)*(_QWORD *)v29 >> 9) ^ ((unsigned int)*(_QWORD *)v29 >> 4));
        v94 = *(_DWORD *)(v3 + 776) + 1;
        v31 = (_QWORD *)(v127 + 16LL * v128);
        v129 = *v31;
        if ( *(_QWORD *)v29 == *v31 )
          goto LABEL_135;
        v130 = 1;
        v131 = 0;
        while ( v129 != -4096 )
        {
          if ( v129 == -8192 && !v131 )
            v131 = v31;
          v128 = v126 & (v130 + v128);
          v31 = (_QWORD *)(v127 + 16LL * v128);
          v129 = *v31;
          if ( *(_QWORD *)v29 == *v31 )
            goto LABEL_135;
          ++v130;
        }
LABEL_221:
        if ( v131 )
          v31 = v131;
LABEL_135:
        *(_DWORD *)(v3 + 776) = v94;
        if ( *v31 != -4096 )
          --*(_DWORD *)(v3 + 780);
        v95 = *(_QWORD *)v29;
        v36 = v31 + 1;
        *v36 = 0;
        *((_QWORD *)v36 - 1) = v95;
        v37 = 0;
LABEL_37:
        *v36 = *((_DWORD *)v29 + 27) + v37;
        v38 = sub_3176580(v3, *(_QWORD *)v29, (__int64)(v29 + 16));
        v39 = (__int64 *)*((_QWORD *)v29 + 14);
        *((_QWORD *)v29 + 1) = v38;
        v40 = v38;
        for ( j = (__int64)&v39[*((unsigned int *)v29 + 30)]; (__int64 *)j != v39; v40 = *((_QWORD *)v29 + 1) )
        {
          v41 = *v39;
          v42 = *(_QWORD *)(*v39 - 32) == 0;
          *(_QWORD *)(*v39 + 80) = *(_QWORD *)(v40 + 24);
          if ( !v42 )
          {
            v43 = *(_QWORD *)(v41 - 24);
            **(_QWORD **)(v41 - 16) = v43;
            if ( v43 )
              *(_QWORD *)(v43 + 16) = *(_QWORD *)(v41 - 16);
          }
          *(_QWORD *)(v41 - 32) = v40;
          v44 = *(_QWORD *)(v40 + 16);
          *(_QWORD *)(v41 - 24) = v44;
          if ( v44 )
          {
            v7 = v41 - 24;
            *(_QWORD *)(v44 + 16) = v41 - 24;
          }
          ++v39;
          *(_QWORD *)(v41 - 16) = v40 + 16;
          *(_QWORD *)(v40 + 16) = v41 - 32;
        }
        v45 = (unsigned int)v187;
        v5 = HIDWORD(v187);
        v46 = (unsigned int)v187 + 1LL;
        if ( v46 > HIDWORD(v187) )
        {
          sub_C8D5F0((__int64)&v186, v188, v46, 8u, j, v7);
          v45 = (unsigned int)v187;
        }
        v21 = v186;
        v186[v45] = v40;
        LODWORD(v187) = v187 + 1;
        v47 = *(_QWORD *)v29;
        if ( !v193 )
          goto LABEL_151;
        v48 = v190;
        v5 = HIDWORD(v191);
        v21 = &v190[HIDWORD(v191)];
        if ( v190 == v21 )
        {
LABEL_153:
          if ( HIDWORD(v191) >= (unsigned int)v191 )
          {
LABEL_151:
            v26 += 4;
            sub_C8CC70((__int64)&v189, v47, (__int64)v21, v5, j, v7);
            if ( v27 == v26 )
              break;
          }
          else
          {
            v5 = (unsigned int)(HIDWORD(v191) + 1);
            v26 += 4;
            ++HIDWORD(v191);
            *v21 = v47;
            v189 = (__int64 *)((char *)v189 + 1);
            if ( v27 == v26 )
              break;
          }
        }
        else
        {
          while ( v47 != *v48 )
          {
            if ( v21 == ++v48 )
              goto LABEL_153;
          }
          v26 += 4;
          if ( v27 == v26 )
            break;
        }
      }
    }
    v49 = (unsigned __int8 *)&v186;
    sub_2A72C20(*(__int64 **)v3, (__int64)&v186, (__int64)v21, v5, j, v7);
    v53 = v190;
    if ( v193 )
    {
      v54 = HIDWORD(v191);
      v55 = &v190[HIDWORD(v191)];
    }
    else
    {
      v54 = (unsigned int)v191;
      v55 = &v190[(unsigned int)v191];
    }
    if ( v190 != v55 )
    {
      while ( 1 )
      {
        v56 = *v53;
        v57 = v53;
        if ( (unsigned __int64)*v53 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v55 == ++v53 )
          goto LABEL_57;
      }
LABEL_117:
      if ( v55 == v57 )
        goto LABEL_57;
      if ( v182 )
      {
        v85 = 1;
        v86 = 0;
        v87 = (v182 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
        v88 = v180 + 16LL * v87;
        v89 = *(_QWORD *)v88;
        if ( v56 == *(_QWORD *)v88 )
        {
LABEL_120:
          v90 = 176LL * *(unsigned int *)(v88 + 12);
          v91 = 176LL * *(unsigned int *)(v88 + 8);
LABEL_121:
          v49 = (unsigned __int8 *)v56;
          sub_31776F0(v3, v56, (__int64)&v195[v91], (__int64)&v195[v90]);
          v92 = v57 + 1;
          if ( v57 + 1 == v55 )
            goto LABEL_57;
          while ( 1 )
          {
            v56 = *v92;
            v57 = v92;
            if ( (unsigned __int64)*v92 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_117;
            if ( v55 == ++v92 )
              goto LABEL_57;
          }
        }
        while ( v89 != -4096 )
        {
          if ( v89 == -8192 && !v86 )
            v86 = (__int64 *)v88;
          v87 = (v182 - 1) & (v85 + v87);
          v88 = v180 + 16LL * v87;
          v89 = *(_QWORD *)v88;
          if ( v56 == *(_QWORD *)v88 )
            goto LABEL_120;
          ++v85;
        }
        if ( !v86 )
          v86 = (__int64 *)v88;
        ++v179;
        v96 = v181 + 1;
        if ( 4 * ((int)v181 + 1) < 3 * v182 )
        {
          if ( v182 - HIDWORD(v181) - v96 <= v182 >> 3 )
          {
            sub_BF2390((__int64)&v179, v182);
            if ( !v182 )
            {
LABEL_321:
              LODWORD(v181) = v181 + 1;
              BUG();
            }
            v149 = 0;
            LODWORD(v150) = (v182 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
            v151 = 1;
            v96 = v181 + 1;
            v86 = (__int64 *)(v180 + 16LL * (unsigned int)v150);
            v152 = *v86;
            if ( v56 != *v86 )
            {
              while ( v152 != -4096 )
              {
                if ( !v149 && v152 == -8192 )
                  v149 = v86;
                v150 = (v182 - 1) & ((_DWORD)v150 + v151);
                v86 = (__int64 *)(v180 + 16 * v150);
                v152 = *v86;
                if ( v56 == *v86 )
                  goto LABEL_148;
                ++v151;
              }
              if ( v149 )
                v86 = v149;
            }
          }
          goto LABEL_148;
        }
      }
      else
      {
        ++v179;
      }
      sub_BF2390((__int64)&v179, 2 * v182);
      if ( !v182 )
        goto LABEL_321;
      LODWORD(v132) = (v182 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v96 = v181 + 1;
      v86 = (__int64 *)(v180 + 16LL * (unsigned int)v132);
      v133 = *v86;
      if ( v56 != *v86 )
      {
        v134 = 1;
        v135 = 0;
        while ( v133 != -4096 )
        {
          if ( v133 == -8192 && !v135 )
            v135 = v86;
          v132 = (v182 - 1) & ((_DWORD)v132 + v134);
          v86 = (__int64 *)(v180 + 16 * v132);
          v133 = *v86;
          if ( v56 == *v86 )
            goto LABEL_148;
          ++v134;
        }
        if ( v135 )
          v86 = v135;
      }
LABEL_148:
      LODWORD(v181) = v96;
      if ( *v86 != -4096 )
        --HIDWORD(v181);
      *v86 = v56;
      v90 = 0;
      v86[1] = 0;
      v91 = 0;
      goto LABEL_121;
    }
LABEL_57:
    if ( &v186[(unsigned int)v187] != v186 )
    {
      v177 = &v186[(unsigned int)v187];
      v58 = v186;
      v59 = 0x8000000000041LL;
      while ( 1 )
      {
LABEL_65:
        v68 = *v58;
        v54 = **(_QWORD **)(*(_QWORD *)(*v58 + 24) + 16LL);
        v69 = *(_BYTE *)(v54 + 8);
        if ( v69 == 7 )
          goto LABEL_64;
        v70 = *(__int64 **)v3;
        if ( v69 != 15 )
          break;
        v49 = (unsigned __int8 *)*v58;
        if ( !(unsigned __int8)sub_2A65580(v70, *v58, v54) )
          goto LABEL_64;
LABEL_68:
        v71 = *(_QWORD *)(v68 + 16);
        if ( !v71 )
          goto LABEL_64;
        do
        {
          v49 = *(unsigned __int8 **)(v71 + 24);
          v72 = *v49;
          if ( (unsigned __int8)v72 > 0x1Cu )
          {
            v73 = (unsigned int)(v72 - 34);
            if ( (unsigned __int8)v73 <= 0x33u )
            {
              if ( _bittest64(&v59, v73) )
              {
                v74 = *((_QWORD *)v49 - 4);
                if ( v74 )
                {
                  if ( !*(_BYTE *)v74 && *(_QWORD *)(v74 + 24) == *((_QWORD *)v49 + 10) && v68 == v74 )
                    sub_2A72F60(*(__int64 **)v3, v49);
                }
              }
            }
          }
          v71 = *(_QWORD *)(v71 + 8);
        }
        while ( v71 );
        if ( v177 == ++v58 )
          goto LABEL_79;
      }
      v60 = sub_2A654E0(v70);
      v61 = *(_QWORD *)(v60 + 8);
      v62 = v60;
      v63 = *(unsigned int *)(v60 + 24);
      if ( (_DWORD)v63 )
      {
        v64 = (v63 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
        v65 = (__int64 *)(v61 + 16LL * v64);
        v66 = *v65;
        if ( v68 == *v65 )
        {
LABEL_61:
          v49 = *(unsigned __int8 **)(v62 + 32);
          if ( v65 != (__int64 *)(v61 + 16 * v63) )
          {
            v67 = &v49[48 * *((unsigned int *)v65 + 2)];
LABEL_63:
            if ( !(unsigned __int8)sub_2A62E90(v67 + 8) )
              goto LABEL_68;
LABEL_64:
            if ( v177 == ++v58 )
              goto LABEL_79;
            goto LABEL_65;
          }
LABEL_159:
          v67 = &v49[48 * *(unsigned int *)(v62 + 40)];
          goto LABEL_63;
        }
        v97 = 1;
        while ( v66 != -4096 )
        {
          v167 = v97 + 1;
          v64 = (v63 - 1) & (v97 + v64);
          v65 = (__int64 *)(v61 + 16LL * v64);
          v66 = *v65;
          if ( v68 == *v65 )
            goto LABEL_61;
          v97 = v167;
        }
      }
      v49 = *(unsigned __int8 **)(v62 + 32);
      goto LABEL_159;
    }
LABEL_79:
    sub_2A72F50(*(__int64 **)v3, v49, v54, v50, v51, v52);
    v75 = v190;
    if ( v193 )
      v76 = &v190[HIDWORD(v191)];
    else
      v76 = &v190[(unsigned int)v191];
    if ( v190 != v76 )
    {
      while ( 1 )
      {
        v77 = *v75;
        v78 = v75;
        if ( (unsigned __int64)*v75 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v76 == ++v75 )
          goto LABEL_84;
      }
      if ( v76 != v75 )
      {
        v98 = *(_DWORD *)(v3 + 752);
        v178 = v3 + 728;
        if ( !v98 )
          goto LABEL_171;
LABEL_162:
        v99 = 1;
        v100 = *(_QWORD *)(v3 + 736);
        v101 = 0;
        v102 = (v98 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
        v103 = v100 + 144LL * v102;
        v104 = *(_QWORD *)v103;
        if ( v77 == *(_QWORD *)v103 )
        {
LABEL_163:
          if ( *(_BYTE *)(v103 + 9) )
            sub_3177130(v3, v77);
          goto LABEL_165;
        }
        while ( v104 != -4096 )
        {
          if ( !v101 && v104 == -8192 )
            v101 = (__int64 *)v103;
          v102 = (v98 - 1) & (v99 + v102);
          v103 = v100 + 144LL * v102;
          v104 = *(_QWORD *)v103;
          if ( v77 == *(_QWORD *)v103 )
            goto LABEL_163;
          ++v99;
        }
        if ( !v101 )
          v101 = (__int64 *)v103;
        v114 = *(_DWORD *)(v3 + 744);
        ++*(_QWORD *)(v3 + 728);
        v110 = v114 + 1;
        if ( 4 * (v114 + 1) < 3 * v98 )
        {
          if ( v98 - *(_DWORD *)(v3 + 748) - v110 > v98 >> 3 )
            goto LABEL_189;
          sub_3178690(v178, v98);
          v136 = *(_DWORD *)(v3 + 752);
          if ( v136 )
          {
            v137 = v136 - 1;
            v138 = *(_QWORD *)(v3 + 736);
            v139 = 1;
            v140 = 0;
            LODWORD(v141) = v137 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
            v101 = (__int64 *)(v138 + 144LL * (unsigned int)v141);
            v110 = *(_DWORD *)(v3 + 744) + 1;
            v142 = *v101;
            if ( *v101 != v77 )
            {
              while ( v142 != -4096 )
              {
                if ( v142 == -8192 && !v140 )
                  v140 = v101;
                v141 = v137 & (unsigned int)(v141 + v139);
                v101 = (__int64 *)(v138 + 144 * v141);
                v142 = *v101;
                if ( v77 == *v101 )
                  goto LABEL_189;
                ++v139;
              }
              if ( v140 )
                v101 = v140;
            }
            goto LABEL_189;
          }
LABEL_319:
          ++*(_DWORD *)(v3 + 744);
          BUG();
        }
        while ( 1 )
        {
          sub_3178690(v178, 2 * v98);
          v106 = *(_DWORD *)(v3 + 752);
          if ( !v106 )
            goto LABEL_319;
          v107 = v106 - 1;
          v108 = *(_QWORD *)(v3 + 736);
          LODWORD(v109) = (v106 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v101 = (__int64 *)(v108 + 144LL * (unsigned int)v109);
          v110 = *(_DWORD *)(v3 + 744) + 1;
          v111 = *v101;
          if ( v77 != *v101 )
          {
            v112 = 1;
            v113 = 0;
            while ( v111 != -4096 )
            {
              if ( !v113 && v111 == -8192 )
                v113 = v101;
              v109 = v107 & (unsigned int)(v109 + v112);
              v101 = (__int64 *)(v108 + 144 * v109);
              v111 = *v101;
              if ( v77 == *v101 )
                goto LABEL_189;
              ++v112;
            }
            if ( v113 )
              v101 = v113;
          }
LABEL_189:
          *(_DWORD *)(v3 + 744) = v110;
          if ( *v101 != -4096 )
            --*(_DWORD *)(v3 + 748);
          *v101 = v77;
          memset(v101 + 1, 0, 0x88u);
LABEL_165:
          v105 = v78 + 1;
          if ( v78 + 1 == v76 )
            goto LABEL_84;
          v77 = *v105;
          ++v78;
          if ( (unsigned __int64)*v105 >= 0xFFFFFFFFFFFFFFFELL )
            break;
LABEL_169:
          if ( v78 == v76 )
            goto LABEL_84;
          v98 = *(_DWORD *)(v3 + 752);
          if ( v98 )
            goto LABEL_162;
LABEL_171:
          ++*(_QWORD *)(v3 + 728);
        }
        while ( v76 != ++v105 )
        {
          v77 = *v105;
          v78 = v105;
          if ( (unsigned __int64)*v105 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_169;
        }
      }
    }
LABEL_84:
    if ( v186 != (__int64 *)v188 )
      _libc_free((unsigned __int64)v186);
    if ( !v193 )
      _libc_free((unsigned __int64)v190);
    if ( v183 != v185 )
      _libc_free((unsigned __int64)v183);
    v79 = 1;
  }
  else
  {
    v79 = 0;
  }
  v80 = (__int64)v195;
  v81 = (unsigned __int64)&v195[176 * (unsigned int)v196];
  if ( v195 != (_BYTE *)v81 )
  {
    do
    {
      v81 -= 176LL;
      v82 = *(_QWORD *)(v81 + 112);
      if ( v82 != v81 + 128 )
        _libc_free(v82);
      v83 = *(_QWORD *)(v81 + 24);
      if ( v83 != v81 + 40 )
        _libc_free(v83);
    }
    while ( v80 != v81 );
    v81 = (unsigned __int64)v195;
  }
  if ( (_BYTE *)v81 != v197 )
    _libc_free(v81);
LABEL_100:
  sub_C7D6A0(v180, 16LL * v182, 8);
  return v79;
}
