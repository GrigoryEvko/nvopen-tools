// Function: sub_2EFDF10
// Address: 0x2efdf10
//
__int64 __fastcall sub_2EFDF10(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  int v9; // ecx
  __int64 v10; // r13
  int v11; // r14d
  __int64 v12; // rax
  __int64 result; // rax
  char v14; // dl
  char v15; // cl
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rsi
  unsigned int v23; // edi
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 j; // r9
  __int16 v27; // cx
  unsigned int v28; // r8d
  __int64 *v29; // rcx
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // rsi
  __m128i *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // edi
  unsigned int v37; // eax
  __int64 v38; // rdx
  bool v39; // r11
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r14
  __int16 *v44; // r13
  unsigned int v45; // ebx
  int v46; // edx
  int v47; // eax
  __int64 v48; // rdx
  unsigned int v49; // ecx
  int v50; // esi
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rax
  __int16 *v54; // rcx
  __int16 v55; // dx
  __int16 *v56; // rcx
  __int16 v57; // si
  unsigned __int16 v58; // dx
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdi
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rax
  char *v66; // rcx
  char *v67; // rax
  __int64 v68; // rdx
  char *v69; // rdx
  __int64 v70; // rcx
  unsigned __int64 v71; // rax
  __int64 i; // r10
  __int16 v73; // dx
  unsigned int v74; // r10d
  __int64 *v75; // rdx
  __int64 v76; // rcx
  __int16 *v77; // rcx
  __int16 v78; // dx
  __int16 *v79; // rcx
  __int16 v80; // si
  unsigned __int16 v81; // dx
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rax
  __int16 *v85; // rcx
  __int16 v86; // dx
  __int16 *v87; // rcx
  __int16 v88; // si
  unsigned __int16 v89; // dx
  __int16 *v90; // rcx
  __int16 *v91; // rsi
  int v92; // r11d
  __int64 v93; // rax
  __int64 v94; // r13
  char v95; // r12
  int v96; // esi
  __int16 *v97; // rax
  int v98; // edx
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rcx
  __int64 v102; // r10
  __int64 v103; // r11
  int v104; // edi
  int v105; // edx
  __int64 v106; // rdx
  _QWORD *v107; // rax
  __int64 v108; // r13
  __int64 v109; // r15
  __int64 v110; // r12
  unsigned __int64 v111; // rbx
  unsigned __int64 v112; // r13
  _QWORD *v113; // r8
  __int64 *v114; // rax
  __int64 v115; // r10
  unsigned int v116; // r9d
  __int64 v117; // r13
  __int64 v118; // r14
  __int64 v119; // r12
  _QWORD *v120; // rbx
  int v121; // ecx
  int v122; // r10d
  unsigned int v123; // esi
  int v124; // edi
  int v125; // ecx
  __int64 *v126; // rax
  __int64 v127; // rcx
  unsigned int v128; // edi
  unsigned int v129; // esi
  __int64 v130; // rdx
  __int128 v131; // rdi
  __int64 v132; // rdx
  __int64 v133; // rax
  _QWORD *v134; // rax
  __int64 v135; // rsi
  _QWORD *v136; // rcx
  int v137; // eax
  int v138; // eax
  unsigned int v139; // edx
  int v140; // edi
  unsigned int v141; // ecx
  char *v142; // rsi
  __int64 v143; // rdx
  char v144; // al
  int v145; // ecx
  int v146; // r8d
  __int64 v147; // rax
  int v148; // ecx
  unsigned int v149; // esi
  unsigned int v150; // r11d
  _DWORD *v151; // rax
  int v152; // edx
  _DWORD *v153; // r10
  int v154; // eax
  _DWORD *v155; // rax
  __int64 v156; // rdx
  __int64 v157; // rdx
  int v158; // edx
  __int64 v159; // rdi
  __int64 v160; // rcx
  unsigned __int64 v161; // rcx
  signed __int64 v162; // rdx
  int v163; // r10d
  int v164; // r10d
  __int64 v165; // rt1
  __int64 v166; // [rsp-10h] [rbp-B0h]
  __int64 v167; // [rsp+0h] [rbp-A0h]
  __int128 v168; // [rsp+8h] [rbp-98h]
  unsigned int v169; // [rsp+1Ch] [rbp-84h]
  int v170; // [rsp+20h] [rbp-80h]
  __int64 v171; // [rsp+20h] [rbp-80h]
  __int64 v172; // [rsp+20h] [rbp-80h]
  __int128 v173; // [rsp+20h] [rbp-80h]
  __int64 v174; // [rsp+28h] [rbp-78h]
  __int64 v175; // [rsp+28h] [rbp-78h]
  __int64 v176; // [rsp+28h] [rbp-78h]
  _QWORD *v177; // [rsp+28h] [rbp-78h]
  unsigned __int64 v178; // [rsp+30h] [rbp-70h]
  __int64 v179; // [rsp+30h] [rbp-70h]
  __int64 v180; // [rsp+30h] [rbp-70h]
  int v181; // [rsp+30h] [rbp-70h]
  __int64 v182; // [rsp+30h] [rbp-70h]
  unsigned __int16 v183; // [rsp+38h] [rbp-68h]
  __int64 v184; // [rsp+38h] [rbp-68h]
  int v185; // [rsp+40h] [rbp-60h]
  __int64 v186; // [rsp+40h] [rbp-60h]
  __int64 v187; // [rsp+48h] [rbp-58h]
  unsigned int v188; // [rsp+48h] [rbp-58h]
  _DWORD *v189; // [rsp+58h] [rbp-48h] BYREF
  __int64 v190; // [rsp+60h] [rbp-40h] BYREF
  __int64 v191; // [rsp+68h] [rbp-38h]

  v7 = a2;
  v8 = a1;
  v9 = *(_DWORD *)a2;
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(_DWORD *)(a2 + 8);
  v185 = (*(_DWORD *)a2 >> 8) & 0xFFF;
  v12 = *(_QWORD *)(a1 + 640);
  v183 = (*(_DWORD *)a2 >> 8) & 0xFFF;
  if ( !v12 || v11 >= 0 )
    goto LABEL_3;
  v31 = v11 & 0x7FFFFFFF;
  if ( *(_DWORD *)(v12 + 160) <= (v11 & 0x7FFFFFFFu)
    || (v32 = v31, (v187 = *(_QWORD *)(*(_QWORD *)(v12 + 152) + 8LL * v31)) == 0) )
  {
    sub_2EF0A60(a1, "Virtual register has no live interval", v7, a3, 0);
LABEL_3:
    result = *(unsigned __int8 *)(v7 + 3);
    v187 = 0;
    v14 = result & 0x10;
    goto LABEL_4;
  }
  result = *(unsigned __int8 *)(v7 + 3);
  v14 = *(_BYTE *)(v7 + 3) & 0x10;
  if ( v185 )
  {
    if ( v14 )
    {
      a6 = *(unsigned int *)(v187 + 8);
      if ( !(_DWORD)a6 )
      {
        if ( (*(_BYTE *)(v7 + 4) & 1) != 0 )
          goto LABEL_9;
        if ( (*(_BYTE *)(v7 + 4) & 2) != 0 )
          goto LABEL_58;
        goto LABEL_8;
      }
      if ( *(_QWORD *)(v187 + 104) )
      {
        v15 = *(_BYTE *)(v7 + 4);
        if ( (v15 & 1) != 0 )
          goto LABEL_9;
        goto LABEL_5;
      }
    }
    else
    {
      if ( (*(_BYTE *)(v7 + 4) & 1) != 0 )
        return result;
      if ( !*(_DWORD *)(v187 + 8) || *(_QWORD *)(v187 + 104) )
      {
        v16 = *(_BYTE *)(v7 + 4) & 2;
        if ( (*(_BYTE *)(v7 + 4) & 2) != 0 )
          return result;
        goto LABEL_36;
      }
    }
    v159 = *(_QWORD *)(a1 + 64);
    v160 = *(_QWORD *)(16 * v32 + *(_QWORD *)(v159 + 56));
    if ( v160 )
    {
      if ( (v160 & 4) == 0 )
      {
        v161 = v160 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v161 )
        {
          if ( *(_BYTE *)(v159 + 48) && *(_BYTE *)(v161 + 43) )
          {
            sub_2EF0A60(v8, "Live interval for subreg operand has no subranges", v7, a3, 0);
            result = *(unsigned __int8 *)(v7 + 3);
            v14 = *(_BYTE *)(v7 + 3) & 0x10;
          }
        }
      }
    }
  }
LABEL_4:
  v15 = *(_BYTE *)(v7 + 4);
  if ( (v15 & 1) != 0 )
    goto LABEL_58;
LABEL_5:
  v16 = v15 & 2;
  if ( (_DWORD)v16 )
    goto LABEL_58;
  if ( v14 )
  {
    v9 = *(_DWORD *)v7;
LABEL_8:
    v16 = v9 & 0xFFF00;
    if ( !(_DWORD)v16 )
      goto LABEL_9;
    goto LABEL_37;
  }
LABEL_36:
  if ( (result & 0x40) != 0 )
  {
    v82 = *(unsigned int *)(v8 + 472);
    v16 = *(unsigned int *)(v8 + 476);
    v83 = v8 + 464;
    if ( v82 + 1 > v16 )
    {
      sub_C8D5F0(v83, (const void *)(v8 + 480), v82 + 1, 4u, a5, a6);
      v82 = *(unsigned int *)(v8 + 472);
      v83 = v8 + 464;
    }
    *(_DWORD *)(*(_QWORD *)(v8 + 464) + 4 * v82) = v11;
    v84 = (unsigned int)(*(_DWORD *)(v8 + 472) + 1);
    *(_DWORD *)(v8 + 472) = v84;
    if ( (unsigned int)(v11 - 1) <= 0x3FFFFFFE )
    {
      v85 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 56LL)
                      + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 8LL) + 24LL * (unsigned int)v11 + 4));
      v86 = *v85;
      v87 = v85 + 1;
      v88 = v86;
      v89 = v11 + v86;
      if ( !v88 )
        v87 = 0;
      sub_2EEE7C0(v83, (char *)(*(_QWORD *)(v8 + 464) + 4 * v84), v89, v87, v89, 0);
      goto LABEL_39;
    }
  }
LABEL_37:
  v33 = *(__m128i **)(v8 + 632);
  if ( !v33
    || v11 >= 0
    || (((*(_BYTE *)(v7 + 3) & 0x40) != 0) & ((*(_BYTE *)(v7 + 3) >> 4) ^ 1)) == 0
    || (*(_BYTE *)(v10 + 44) & 4) != 0 )
  {
    goto LABEL_39;
  }
  v65 = sub_2E29D60(v33, v11, (*(_BYTE *)(v7 + 3) >> 4) ^ 1u, v16, a5, a6);
  v66 = *(char **)(v65 + 40);
  v67 = *(char **)(v65 + 32);
  v68 = (v66 - v67) >> 5;
  if ( v68 > 0 )
  {
    v69 = &v67[32 * v68];
    while ( v10 != *(_QWORD *)v67 )
    {
      if ( v10 == *((_QWORD *)v67 + 1) )
      {
        v67 += 8;
        goto LABEL_90;
      }
      if ( v10 == *((_QWORD *)v67 + 2) )
      {
        v67 += 16;
        goto LABEL_90;
      }
      if ( v10 == *((_QWORD *)v67 + 3) )
      {
        v67 += 24;
        goto LABEL_90;
      }
      v67 += 32;
      if ( v69 == v67 )
        goto LABEL_257;
    }
    goto LABEL_90;
  }
LABEL_257:
  v162 = v66 - v67;
  if ( v66 - v67 == 16 )
    goto LABEL_270;
  if ( v162 == 24 )
  {
    if ( v10 == *(_QWORD *)v67 )
      goto LABEL_90;
    v67 += 8;
LABEL_270:
    if ( v10 == *(_QWORD *)v67 )
      goto LABEL_90;
    v67 += 8;
    goto LABEL_260;
  }
  if ( v162 != 8 )
    goto LABEL_91;
LABEL_260:
  if ( v10 != *(_QWORD *)v67 )
    goto LABEL_91;
LABEL_90:
  if ( v66 == v67 )
LABEL_91:
    sub_2EF0A60(v8, "Kill missing from LiveVariables", v7, a3, 0);
LABEL_39:
  v34 = *(_QWORD *)(v8 + 640);
  if ( !v34 )
    goto LABEL_55;
  v35 = *(_QWORD *)(v34 + 32);
  v36 = *(_DWORD *)(v35 + 144);
  a5 = *(_QWORD *)(v35 + 128);
  if ( !v36 )
    goto LABEL_55;
  a6 = v36 - 1;
  v37 = a6 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v38 = *(_QWORD *)(a5 + 16LL * v37);
  if ( v10 != v38 )
  {
    v145 = 1;
    while ( v38 != -4096 )
    {
      v37 = a6 & (v145 + v37);
      v38 = *(_QWORD *)(a5 + 16LL * v37);
      if ( v10 == v38 )
        goto LABEL_42;
      ++v145;
    }
    goto LABEL_55;
  }
LABEL_42:
  v39 = *(_WORD *)(v10 + 68) == 0 || *(_WORD *)(v10 + 68) == 68;
  if ( v39 )
  {
    v40 = *(_QWORD *)(*(_QWORD *)(v35 + 152)
                    + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL * (a3 + 1) + 24) + 24LL)
                    + 8);
    if ( ((v40 >> 1) & 3) != 0 )
      v178 = v40 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v40 >> 1) & 3) - 1));
    else
      v178 = *(_QWORD *)(v40 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
  }
  else
  {
    v70 = v10;
    v71 = v10;
    if ( (*(_DWORD *)(v10 + 44) & 4) != 0 )
    {
      do
        v71 = *(_QWORD *)v71 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v71 + 44) & 4) != 0 );
    }
    if ( (*(_DWORD *)(v10 + 44) & 8) != 0 )
    {
      do
        v70 = *(_QWORD *)(v70 + 8);
      while ( (*(_BYTE *)(v70 + 44) & 8) != 0 );
    }
    for ( i = *(_QWORD *)(v70 + 8); i != v71; v71 = *(_QWORD *)(v71 + 8) )
    {
      v73 = *(_WORD *)(v71 + 68);
      if ( (unsigned __int16)(v73 - 14) > 4u && v73 != 24 )
        break;
    }
    v74 = a6 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
    v75 = (__int64 *)(a5 + 16LL * v74);
    v76 = *v75;
    if ( *v75 != v71 )
    {
      v158 = 1;
      while ( v76 != -4096 )
      {
        v74 = a6 & (v158 + v74);
        v181 = v158 + 1;
        v75 = (__int64 *)(a5 + 16LL * v74);
        v76 = *v75;
        if ( *v75 == v71 )
          goto LABEL_101;
        v158 = v181;
      }
      v75 = (__int64 *)(a5 + 16LL * v36);
    }
LABEL_101:
    v178 = v75[1];
  }
  if ( (unsigned int)(v11 - 1) <= 0x3FFFFFFE )
  {
    if ( (unsigned int)v11 >= *(_DWORD *)(v8 + 264)
      || (*(_QWORD *)(*(_QWORD *)(v8 + 200) + 8LL * ((unsigned int)v11 >> 6)) & (1LL << v11)) == 0 )
    {
      v41 = *(_QWORD *)(v8 + 56);
      v174 = v10;
      v42 = 3LL * (unsigned int)v11;
      v170 = v11;
      v43 = v8;
      LODWORD(v42) = *(_DWORD *)(*(_QWORD *)(v41 + 8) + 8 * v42 + 16);
      a6 = v42 & 0xFFF;
      v44 = (__int16 *)(*(_QWORD *)(v41 + 56) + 2LL * ((unsigned int)v42 >> 12));
      v45 = v42 & 0xFFF;
      do
      {
        if ( !v44 )
          break;
        if ( !(unsigned __int8)sub_2EBFC90(*(_QWORD **)(v43 + 64), v45) )
        {
          a5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 640) + 424LL) + 8LL * v45);
          if ( a5 )
            sub_2EF0B60(v43, v7, a3, v178, a5, v45, 0, 0);
        }
        v46 = *v44++;
        v45 += v46;
      }
      while ( (_WORD)v46 );
      v8 = v43;
      v10 = v174;
      v11 = v170;
    }
    goto LABEL_55;
  }
  if ( v11 >= 0 )
    goto LABEL_55;
  if ( !v39 )
    goto LABEL_133;
  v106 = *(_QWORD *)(*(_QWORD *)(v35 + 152)
                   + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL * (a3 + 1) + 24) + 24LL)
                   + 8);
  if ( ((v106 >> 1) & 3) == 0 )
  {
    v178 = *(_QWORD *)(v106 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
LABEL_133:
    sub_2EF0B60(v8, v7, a3, v178, v187, v11, 0, 0);
    goto LABEL_134;
  }
  v178 = v106 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v106 >> 1) & 3) - 1));
  sub_2EF0B60(v8, v7, a3, v178, v187, v11, 0, 0);
LABEL_134:
  a6 = v166;
  a5 = *(_QWORD *)(v187 + 104);
  if ( !a5 || (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
    goto LABEL_55;
  if ( v185 )
  {
    v107 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 272LL) + 16LL * v183);
    *((_QWORD *)&v173 + 1) = *v107;
    *(_QWORD *)&v173 = v107[1];
LABEL_138:
    v167 = v10;
    v108 = a5;
    v168 = 0u;
    v169 = a3;
    v109 = v7;
    v110 = v8;
    v111 = v178 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      if ( *(_QWORD *)(v108 + 112) & *((_QWORD *)&v173 + 1) | *(_QWORD *)(v108 + 120) & (unsigned __int64)v173 )
      {
        sub_2EF0B60(v110, v109, v169, v178, v108, v11, *(_QWORD *)(v108 + 112), *(_QWORD *)(v108 + 120));
        v126 = (__int64 *)sub_2E09D00((__int64 *)v108, v111);
        v127 = *(_QWORD *)v108 + 24LL * *(unsigned int *)(v108 + 8);
        if ( v126 != (__int64 *)v127 )
          break;
      }
LABEL_140:
      v108 = *(_QWORD *)(v108 + 104);
      if ( !v108 )
      {
        v8 = v110;
        v7 = v109;
        v10 = v167;
        a3 = v169;
        if ( (v173 & v168) == 0 )
          goto LABEL_142;
        goto LABEL_143;
      }
    }
    v128 = *(_DWORD *)(v111 + 24);
    v129 = *(_DWORD *)((*v126 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v129 | (*v126 >> 1) & 3) > v128 )
    {
      a5 = 0;
      v130 = 0;
    }
    else
    {
      a5 = v126[1];
      v130 = v126[2];
      if ( v111 == (a5 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        a6 = 0;
        if ( (__int64 *)v127 == v126 + 3 )
        {
LABEL_176:
          if ( v130
            || (*(_WORD *)(v167 + 68) == 68 || !*(_WORD *)(v167 + 68))
            && (a5 = ((unsigned __int8)a5 ^ 6) & 6, (_DWORD)a5)
            && a6 )
          {
            *((_QWORD *)&v131 + 1) = *(_QWORD *)(v108 + 112);
            *(_QWORD *)&v131 = *(_QWORD *)(v108 + 120);
            v168 |= v131;
          }
          goto LABEL_140;
        }
        v129 = *(_DWORD *)((v126[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v126 += 3;
      }
      if ( v111 == *(_QWORD *)(v130 + 8) )
        v130 = 0;
    }
    a6 = 0;
    if ( v129 <= v128 )
    {
      a6 = v126[2];
      a5 = v126[1];
    }
    goto LABEL_176;
  }
  *((_QWORD *)&v173 + 1) = sub_2EBF1E0(*(_QWORD *)(v8 + 64), v11);
  *(_QWORD *)&v173 = v156;
  a5 = *(_QWORD *)(v187 + 104);
  if ( a5 )
    goto LABEL_138;
  v168 = 0u;
LABEL_142:
  sub_2EF0A60(v8, "No live subrange at use", v7, a3, 0);
  sub_2EEF440(*(_QWORD *)(v8 + 16), v187);
  sub_2EEF640(*(_QWORD *)(v8 + 16), v178);
LABEL_143:
  if ( (!*(_WORD *)(v10 + 68) || *(_WORD *)(v10 + 68) == 68) && v168 != v173 )
  {
    sub_2EF0A60(v8, "Not all lanes of PHI source live at use", v7, a3, 0);
    sub_2EEF440(*(_QWORD *)(v8 + 16), v187);
    sub_2EEF640(*(_QWORD *)(v8 + 16), v178);
  }
LABEL_55:
  v47 = *(_DWORD *)(v8 + 296);
  v48 = *(_QWORD *)(v8 + 280);
  if ( v47 )
  {
    v49 = (v47 - 1) & (37 * v11);
    v50 = *(_DWORD *)(v48 + 4LL * v49);
    if ( v11 == v50 )
      goto LABEL_57;
    a5 = 1;
    while ( v50 != -1 )
    {
      a6 = (unsigned int)(a5 + 1);
      v49 = (v47 - 1) & (a5 + v49);
      v50 = *(_DWORD *)(v48 + 4LL * v49);
      if ( v11 == v50 )
        goto LABEL_57;
      a5 = (unsigned int)a6;
    }
  }
  if ( (unsigned int)(v11 - 1) > 0x3FFFFFFE )
  {
    v132 = *(_QWORD *)(v8 + 64);
    if ( v11 < 0 )
      v133 = *(_QWORD *)(*(_QWORD *)(v132 + 56) + 16LL * (v11 & 0x7FFFFFFF) + 8);
    else
      v133 = *(_QWORD *)(*(_QWORD *)(v132 + 304) + 8LL * (unsigned int)v11);
    if ( !v133
      || (*(_BYTE *)(v133 + 3) & 0x10) == 0
      && ((v147 = *(_QWORD *)(v133 + 32)) == 0 || (*(_BYTE *)(v147 + 3) & 0x10) == 0) )
    {
      v141 = a3;
      v142 = "Reading virtual register without a def";
      v143 = v7;
      goto LABEL_185;
    }
    v190 = *(_QWORD *)(v10 + 24);
    v134 = sub_2EEFC50(v8 + 600, &v190);
    a5 = (__int64)&v190;
    v135 = v134[6];
    v136 = v134;
    v137 = *((_DWORD *)v134 + 16);
    if ( v137 )
    {
      v138 = v137 - 1;
      v139 = v138 & (37 * v11);
      v140 = *(_DWORD *)(v135 + 4LL * v139);
      if ( v11 == v140 )
      {
LABEL_184:
        v141 = a3;
        v142 = "Using a killed virtual register";
        v143 = v7;
        goto LABEL_185;
      }
      v163 = 1;
      while ( v140 != -1 )
      {
        a6 = (unsigned int)(v163 + 1);
        v139 = v138 & (v163 + v139);
        v140 = *(_DWORD *)(v135 + 4LL * v139);
        if ( v11 == v140 )
          goto LABEL_184;
        ++v163;
      }
    }
    if ( !*(_WORD *)(v10 + 68) || *(_WORD *)(v10 + 68) == 68 )
      goto LABEL_57;
    LODWORD(v190) = v11;
    v191 = v10;
    v149 = *((_DWORD *)v136 + 8);
    if ( v149 )
    {
      v150 = (v149 - 1) & (37 * v11);
      v180 = v136[2];
      v151 = (_DWORD *)(v180 + 16LL * v150);
      v152 = *v151;
      if ( v11 == *v151 )
        goto LABEL_57;
      a6 = 1;
      v153 = 0;
      while ( v152 != -1 )
      {
        if ( v152 == -2 && !v153 )
          v153 = v151;
        v150 = (v149 - 1) & (a6 + v150);
        v151 = (_DWORD *)(v180 + 16LL * v150);
        v152 = *v151;
        if ( v11 == *v151 )
          goto LABEL_57;
        a6 = (unsigned int)(a6 + 1);
      }
      if ( v153 )
        v151 = v153;
      ++v136[1];
      v189 = v151;
      v154 = *((_DWORD *)v136 + 6) + 1;
      a6 = (unsigned int)(4 * v154);
      if ( (unsigned int)a6 < 3 * v149 )
      {
        if ( v149 - *((_DWORD *)v136 + 7) - v154 > v149 >> 3 )
        {
LABEL_232:
          v155 = v189;
          ++*((_DWORD *)v136 + 6);
          if ( *v155 != -1 )
            --*((_DWORD *)v136 + 7);
          *v155 = v190;
          *((_QWORD *)v155 + 1) = v191;
          goto LABEL_57;
        }
        v177 = v136;
LABEL_281:
        v182 = (__int64)(v136 + 1);
        sub_2EFDD30((__int64)(v136 + 1), v149);
        sub_2EF9170(v182, (int *)&v190, &v189);
        v136 = v177;
        goto LABEL_232;
      }
    }
    else
    {
      ++v136[1];
      v189 = 0;
    }
    v149 *= 2;
    v177 = v136;
    goto LABEL_281;
  }
  if ( (unsigned int)v11 < *(_DWORD *)(v8 + 264) )
  {
    a5 = *(_QWORD *)(v8 + 200);
    if ( (*(_QWORD *)(a5 + 8LL * ((unsigned int)v11 >> 6)) & (1LL << v11)) != 0 )
      goto LABEL_57;
  }
  v179 = 24LL * (unsigned int)v11;
  v90 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 56LL)
                  + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 8LL) + v179 + 4));
  v91 = v90 + 1;
  LODWORD(v90) = *v90;
  v92 = v11 + (_DWORD)v90;
  if ( (_WORD)v90 )
  {
    v121 = (unsigned __int16)v92;
    a6 = (__int64)v91;
    v122 = v47 - 1;
    if ( !v47 )
      goto LABEL_166;
LABEL_163:
    v123 = v122 & (37 * v121);
    v124 = *(_DWORD *)(v48 + 4LL * v123);
    if ( v121 == v124 )
      goto LABEL_57;
    a5 = 1;
    while ( v124 != -1 )
    {
      v123 = v122 & (a5 + v123);
      v124 = *(_DWORD *)(v48 + 4LL * v123);
      if ( v121 == v124 )
        goto LABEL_57;
      a5 = (unsigned int)(a5 + 1);
    }
LABEL_166:
    while ( 1 )
    {
      v125 = *(__int16 *)a6;
      a6 += 2;
      if ( !(_WORD)v125 )
        break;
      v92 += v125;
      v121 = (unsigned __int16)v92;
      if ( v47 )
        goto LABEL_163;
    }
  }
  v171 = *(_QWORD *)(v10 + 32);
  v175 = v171 + 40LL * (*(_DWORD *)(v10 + 40) & 0xFFFFFF);
  a5 = v171 + 40LL * (unsigned int)sub_2E88FE0(v10);
  v93 = v175;
  if ( v175 != a5 )
  {
    v176 = v10;
    v94 = v93;
    v172 = v7;
    v95 = 1;
    do
    {
      if ( !*(_BYTE *)a5 && (*(_BYTE *)(a5 + 3) & 0x20) != 0 )
      {
        v99 = *(unsigned int *)(a5 + 8);
        if ( (unsigned int)(v99 - 1) <= 0x3FFFFFFE && v11 != (_DWORD)v99 )
        {
          v100 = *(_QWORD *)(v8 + 56);
          v101 = *(_QWORD *)(v100 + 8);
          v102 = *(_QWORD *)(v100 + 56);
          v103 = v101 + 24 * v99;
          v104 = *(_DWORD *)(v101 + v179 + 16) & 0xFFF;
          a6 = v102 + 2LL * (*(_DWORD *)(v101 + v179 + 16) >> 12);
LABEL_123:
          if ( a6 )
          {
            v98 = *(_DWORD *)(v103 + 16) & 0xFFF;
            v97 = (__int16 *)(v102 + 2LL * (*(_DWORD *)(v103 + 16) >> 12));
            while ( v97 )
            {
              if ( v104 == v98 )
              {
                v105 = *(__int16 *)a6;
                a6 += 2;
                v104 += v105;
                if ( (_WORD)v105 )
                  goto LABEL_123;
                goto LABEL_128;
              }
              v96 = *v97++;
              v98 += v96;
              if ( !(_WORD)v96 )
                break;
            }
          }
          else
          {
LABEL_128:
            v95 = 0;
          }
        }
      }
      a5 += 40;
    }
    while ( v94 != a5 );
    v144 = v95;
    v10 = v176;
    v7 = v172;
    if ( !v144 )
    {
LABEL_57:
      result = *(unsigned __int8 *)(v7 + 3);
      v14 = *(_BYTE *)(v7 + 3) & 0x10;
      goto LABEL_58;
    }
  }
  v141 = a3;
  v142 = "Using an undefined physical register";
  v143 = v7;
LABEL_185:
  sub_2EF0A60(v8, v142, v143, v141, 0);
  result = *(unsigned __int8 *)(v7 + 3);
  v14 = *(_BYTE *)(v7 + 3) & 0x10;
LABEL_58:
  if ( !v14 )
    return result;
LABEL_9:
  if ( (((result & 0x10) != 0) & ((unsigned __int8)result >> 6)) != 0 )
  {
    v17 = *(unsigned int *)(v8 + 392);
    v18 = v8 + 384;
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 396) )
    {
      sub_C8D5F0(v18, (const void *)(v8 + 400), v17 + 1, 4u, a5, a6);
      v17 = *(unsigned int *)(v8 + 392);
      v18 = v8 + 384;
    }
    *(_DWORD *)(*(_QWORD *)(v8 + 384) + 4 * v17) = v11;
    v19 = (unsigned int)(*(_DWORD *)(v8 + 392) + 1);
    *(_DWORD *)(v8 + 392) = v19;
    if ( (unsigned int)(v11 - 1) <= 0x3FFFFFFE )
    {
      v77 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 56LL)
                      + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 8LL) + 24LL * (unsigned int)v11 + 4));
      v78 = *v77;
      v79 = v77 + 1;
      v80 = v78;
      v81 = v11 + v78;
      if ( !v80 )
        v79 = 0;
      sub_2EEE7C0(v18, (char *)(*(_QWORD *)(v8 + 384) + 4 * v19), v81, v79, v81, 0);
      goto LABEL_14;
    }
  }
  else
  {
    v51 = *(unsigned int *)(v8 + 312);
    v52 = v8 + 304;
    if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 316) )
    {
      sub_C8D5F0(v52, (const void *)(v8 + 320), v51 + 1, 4u, a5, a6);
      v51 = *(unsigned int *)(v8 + 312);
      v52 = v8 + 304;
    }
    *(_DWORD *)(*(_QWORD *)(v8 + 304) + 4 * v51) = v11;
    v53 = (unsigned int)(*(_DWORD *)(v8 + 312) + 1);
    *(_DWORD *)(v8 + 312) = v53;
    if ( (unsigned int)(v11 - 1) <= 0x3FFFFFFE )
    {
      v54 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 56LL)
                      + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 8LL) + 24LL * (unsigned int)v11 + 4));
      v55 = *v54;
      v56 = v54 + 1;
      v57 = v55;
      v58 = v11 + v55;
      if ( !v57 )
        v56 = 0;
      sub_2EEE7C0(v52, (char *)(*(_QWORD *)(v8 + 304) + 4 * v53), v58, v56, v58, 0);
      goto LABEL_14;
    }
  }
  v20 = *(_QWORD **)(v8 + 64);
  if ( (*(_BYTE *)(*v20 + 344LL) & 1) != 0 && v11 < 0 )
  {
    v59 = *(_QWORD *)(v20[7] + 16LL * (v11 & 0x7FFFFFFF) + 8);
    if ( v59 )
    {
      if ( (*(_BYTE *)(v59 + 3) & 0x10) != 0 || (v59 = *(_QWORD *)(v59 + 32)) != 0 && (*(_BYTE *)(v59 + 3) & 0x10) != 0 )
      {
        v60 = v59;
        v61 = 0;
        do
        {
          v60 = *(_QWORD *)(v60 + 32);
          ++v61;
        }
        while ( v60 && (*(_BYTE *)(v60 + 3) & 0x10) != 0 );
        if ( v61 > 2 )
          goto LABEL_79;
        if ( v61 == 2 )
        {
          v62 = *(_QWORD *)(v59 + 32);
          if ( v62 && (*(_BYTE *)(v62 + 3) & 0x10) == 0 )
            v62 = 0;
          v63 = *(_QWORD *)(v59 + 16);
          v64 = *(_QWORD *)(v62 + 16);
          if ( *(_WORD *)(v63 + 68) == 21 )
          {
            if ( *(_WORD *)(v64 + 68) == 21 )
              goto LABEL_79;
          }
          else
          {
            if ( *(_WORD *)(v64 + 68) != 21 )
            {
LABEL_79:
              sub_2EF0A60(v8, "Multiple virtual register defs in SSA form", v7, a3, 0);
              goto LABEL_14;
            }
            v64 = *(_QWORD *)(v59 + 16);
            v63 = *(_QWORD *)(v62 + 16);
            v165 = v62;
            v62 = v59;
            v59 = v165;
          }
          if ( (*(_BYTE *)(v63 + 44) & 4) != 0 )
            goto LABEL_79;
          for ( ; (*(_BYTE *)(v64 + 44) & 4) != 0; v64 = *(_QWORD *)v64 & 0xFFFFFFFFFFFFFFF8LL )
            ;
          if ( v63 != v64 || (*(_BYTE *)(v59 + 3) & 0x20) == 0 || (*(_BYTE *)(v62 + 3) & 0x20) != 0 )
            goto LABEL_79;
        }
      }
    }
  }
LABEL_14:
  result = *(_QWORD *)(v8 + 640);
  if ( result )
  {
    result = *(_QWORD *)(result + 32);
    v21 = *(_DWORD *)(result + 144);
    v22 = *(_QWORD *)(result + 128);
    if ( v21 )
    {
      v23 = v21 - 1;
      result = (v21 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v24 = *(_QWORD *)(v22 + 16 * result);
      if ( v10 == v24 )
      {
LABEL_17:
        v25 = v10;
        if ( (*(_DWORD *)(v10 + 44) & 4) != 0 )
        {
          do
            v25 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v25 + 44) & 4) != 0 );
        }
        for ( ; (*(_BYTE *)(v10 + 44) & 8) != 0; v10 = *(_QWORD *)(v10 + 8) )
          ;
        for ( j = *(_QWORD *)(v10 + 8); j != v25; v25 = *(_QWORD *)(v25 + 8) )
        {
          v27 = *(_WORD *)(v25 + 68);
          if ( (unsigned __int16)(v27 - 14) > 4u && v27 != 24 )
            break;
        }
        v28 = v23 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = (__int64 *)(v22 + 16LL * v28);
        v30 = *v29;
        if ( v25 != *v29 )
        {
          v148 = 1;
          while ( v30 != -4096 )
          {
            v164 = v148 + 1;
            v28 = v23 & (v148 + v28);
            v29 = (__int64 *)(v22 + 16LL * v28);
            v30 = *v29;
            if ( v25 == *v29 )
              goto LABEL_26;
            v148 = v164;
          }
          v29 = (__int64 *)(v22 + 16LL * v21);
        }
LABEL_26:
        result = (*(_BYTE *)(v7 + 4) & 4) == 0 ? 4LL : 2LL;
        if ( v11 < 0 )
        {
          v112 = result | v29[1] & 0xFFFFFFFFFFFFFFF8LL;
          sub_2EF0D40(v8, v7, a3, v112, v187, v11, 0, 0, 0);
          result = v187;
          v113 = *(_QWORD **)(v187 + 104);
          if ( v113 )
          {
            if ( v185 )
            {
              v114 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v8 + 56) + 272LL) + 16LL * v183);
              v115 = *v114;
              result = v114[1];
            }
            else
            {
              v115 = sub_2EBF1E0(*(_QWORD *)(v8 + 64), v11);
              result = v157;
              v113 = *(_QWORD **)(v187 + 104);
              if ( !v113 )
                return result;
            }
            v186 = v112;
            v116 = v11;
            v117 = result;
            v118 = v7;
            v184 = v8;
            v119 = v115;
            v120 = v113;
            do
            {
              if ( v117 & v120[15] | v119 & v120[14] )
              {
                v188 = v116;
                result = sub_2EF0D40(v184, v118, a3, v186, (__int64)v120, v116, 1, v120[14], v120[15]);
                v116 = v188;
              }
              v120 = (_QWORD *)v120[13];
            }
            while ( v120 );
          }
        }
      }
      else
      {
        v146 = 1;
        while ( v24 != -4096 )
        {
          result = v23 & (v146 + (_DWORD)result);
          v24 = *(_QWORD *)(v22 + 16LL * (unsigned int)result);
          if ( v10 == v24 )
            goto LABEL_17;
          ++v146;
        }
      }
    }
  }
  return result;
}
