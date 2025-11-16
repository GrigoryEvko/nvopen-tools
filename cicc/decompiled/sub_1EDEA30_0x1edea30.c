// Function: sub_1EDEA30
// Address: 0x1edea30
//
__int64 __fastcall sub_1EDEA30(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, int a5)
{
  char v6; // cl
  int v7; // edx
  int v8; // r15d
  __int64 v9; // r14
  int v10; // r9d
  unsigned __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r13
  __int64 v14; // r10
  __int64 v15; // r12
  unsigned int v16; // edx
  __int64 v17; // r8
  __int64 v18; // r13
  __int64 v19; // rcx
  unsigned __int64 i; // rdx
  unsigned int v21; // edi
  __int64 v22; // rsi
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r10
  unsigned __int64 v26; // rbx
  __int64 *v27; // rdx
  __int64 *v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // r15
  unsigned int v32; // r14d
  unsigned int v34; // r13d
  __int64 v35; // rcx
  unsigned int v36; // r15d
  __int64 v37; // r12
  int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // rcx
  _QWORD *v41; // rdx
  _QWORD *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rsi
  _QWORD *v45; // rdx
  _QWORD *v46; // rax
  __int64 v47; // rcx
  unsigned int v48; // esi
  __int64 v49; // r8
  _BYTE *v50; // r9
  __int64 v51; // rax
  __int64 v52; // r11
  __int64 *v53; // r9
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 *v56; // r10
  __int64 v57; // rcx
  __int64 v58; // rbx
  __int64 *v59; // rdi
  int v60; // r9d
  unsigned int v61; // esi
  unsigned int v62; // edx
  unsigned int v63; // ecx
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // r9
  __int64 v70; // r11
  __int64 v71; // r8
  int v72; // eax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // r8
  __int64 v76; // rax
  __int64 v77; // r13
  unsigned __int64 v78; // r14
  __int64 v79; // rbx
  _QWORD *v80; // r12
  unsigned __int64 *v81; // rsi
  unsigned __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // r8
  __int64 j; // r15
  __int64 v87; // r14
  unsigned __int64 v88; // rbx
  __int64 *v89; // rax
  __int64 v90; // rsi
  __int64 v91; // r15
  __int64 v92; // r14
  __int64 *v93; // r12
  __int64 *v94; // rax
  __int64 **v95; // rax
  __int64 *v96; // rbx
  __int64 v97; // r15
  __int64 v98; // r14
  __int64 v99; // r12
  __int64 **v100; // rax
  _DWORD *v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rax
  __int64 v104; // rbx
  __int64 v105; // rax
  __int64 v106; // rbx
  __int64 v107; // r13
  unsigned __int64 v108; // r14
  __int64 v109; // r15
  __int64 *v110; // rsi
  unsigned __int64 v111; // rax
  int v112; // esi
  __int64 v113; // rax
  __int64 v114; // rcx
  int v115; // r8d
  __int128 v116; // [rsp-20h] [rbp-E0h]
  __int64 v117; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v118; // [rsp+8h] [rbp-B8h]
  __int64 v119; // [rsp+10h] [rbp-B0h]
  __int64 v120; // [rsp+10h] [rbp-B0h]
  __int64 v121; // [rsp+10h] [rbp-B0h]
  __int64 v122; // [rsp+10h] [rbp-B0h]
  __int64 v123; // [rsp+10h] [rbp-B0h]
  __int64 v124; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v125; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v126; // [rsp+18h] [rbp-A8h]
  __int64 v127; // [rsp+18h] [rbp-A8h]
  __int64 v128; // [rsp+18h] [rbp-A8h]
  __int64 v129; // [rsp+20h] [rbp-A0h]
  __int64 v130; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v131; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v132; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v133; // [rsp+20h] [rbp-A0h]
  int v134; // [rsp+28h] [rbp-98h]
  __int64 v135; // [rsp+28h] [rbp-98h]
  __int64 v136; // [rsp+28h] [rbp-98h]
  __int64 v137; // [rsp+30h] [rbp-90h]
  unsigned __int64 v138; // [rsp+30h] [rbp-90h]
  __int64 v139; // [rsp+30h] [rbp-90h]
  __int64 v140; // [rsp+30h] [rbp-90h]
  __int64 v141; // [rsp+30h] [rbp-90h]
  __int64 v142; // [rsp+30h] [rbp-90h]
  __int64 v143; // [rsp+30h] [rbp-90h]
  __int64 v144; // [rsp+30h] [rbp-90h]
  int v145; // [rsp+30h] [rbp-90h]
  __int64 v146; // [rsp+38h] [rbp-88h]
  __int64 v147; // [rsp+38h] [rbp-88h]
  __int64 v148; // [rsp+40h] [rbp-80h]
  int v149; // [rsp+40h] [rbp-80h]
  __int64 k; // [rsp+48h] [rbp-78h]
  __int64 v151; // [rsp+48h] [rbp-78h]
  __int64 *v153; // [rsp+50h] [rbp-70h]
  unsigned __int8 v154; // [rsp+50h] [rbp-70h]
  unsigned int v156; // [rsp+68h] [rbp-58h] BYREF
  unsigned int v157; // [rsp+6Ch] [rbp-54h] BYREF
  __int64 **v158; // [rsp+70h] [rbp-50h] BYREF
  __int64 v159; // [rsp+78h] [rbp-48h]
  __int64 (__fastcall *v160)(__m128i **, const __m128i **, int); // [rsp+80h] [rbp-40h]
  __int64 (__fastcall *v161)(__int64 *, __int64); // [rsp+88h] [rbp-38h]

  v6 = *(_BYTE *)(a2 + 26);
  v7 = *(_DWORD *)(a2 + 8);
  v8 = *(_DWORD *)(a2 + 12);
  v9 = *(_QWORD *)(a1 + 272);
  v10 = v7;
  if ( !v6 )
    v10 = *(_DWORD *)(a2 + 12);
  v11 = *(unsigned int *)(v9 + 408);
  v12 = v10 & 0x7FFFFFFF;
  v13 = v10 & 0x7FFFFFFF;
  v14 = 8 * v13;
  if ( (v10 & 0x7FFFFFFFu) >= (unsigned int)v11 || (v15 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * v12)) == 0 )
  {
    v36 = v12 + 1;
    if ( (unsigned int)v11 < v12 + 1 )
    {
      v39 = v36;
      if ( v36 < v11 )
      {
        *(_DWORD *)(v9 + 408) = v36;
      }
      else if ( v36 > v11 )
      {
        if ( v36 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
        {
          v149 = v10;
          v147 = 8LL * (v10 & 0x7FFFFFFF);
          sub_16CD150(v9 + 400, (const void *)(v9 + 416), v36, 8, a5, v10);
          v11 = *(unsigned int *)(v9 + 408);
          v14 = v147;
          v10 = v149;
          v39 = v36;
        }
        v37 = *(_QWORD *)(v9 + 400);
        v40 = *(_QWORD *)(v9 + 416);
        v41 = (_QWORD *)(v37 + 8 * v39);
        v42 = (_QWORD *)(v37 + 8 * v11);
        if ( v41 != v42 )
        {
          do
            *v42++ = v40;
          while ( v41 != v42 );
          v37 = *(_QWORD *)(v9 + 400);
        }
        *(_DWORD *)(v9 + 408) = v36;
        goto LABEL_30;
      }
    }
    v37 = *(_QWORD *)(v9 + 400);
LABEL_30:
    *(_QWORD *)(v14 + v37) = sub_1DBA290(v10);
    v15 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8 * v13);
    sub_1DBB110((_QWORD *)v9, v15);
    v7 = *(_DWORD *)(a2 + 8);
    v8 = *(_DWORD *)(a2 + 12);
    v6 = *(_BYTE *)(a2 + 26);
    v9 = *(_QWORD *)(a1 + 272);
    v11 = *(unsigned int *)(v9 + 408);
  }
  if ( !v6 )
    v8 = v7;
  v16 = v8 & 0x7FFFFFFF;
  v17 = v8 & 0x7FFFFFFF;
  if ( (v8 & 0x7FFFFFFFu) >= (unsigned int)v11 || (v18 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * v16)) == 0 )
  {
    v34 = v16 + 1;
    if ( v16 + 1 > (unsigned int)v11 )
    {
      v43 = v34;
      if ( v34 < v11 )
      {
        *(_DWORD *)(v9 + 408) = v34;
      }
      else if ( v34 > v11 )
      {
        if ( v34 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
        {
          sub_16CD150(v9 + 400, (const void *)(v9 + 416), v34, 8, v17, v10);
          v11 = *(unsigned int *)(v9 + 408);
          v17 = v8 & 0x7FFFFFFF;
          v43 = v34;
        }
        v35 = *(_QWORD *)(v9 + 400);
        v44 = *(_QWORD *)(v9 + 416);
        v45 = (_QWORD *)(v35 + 8 * v43);
        v46 = (_QWORD *)(v35 + 8 * v11);
        if ( v45 != v46 )
        {
          do
            *v46++ = v44;
          while ( v45 != v46 );
          v35 = *(_QWORD *)(v9 + 400);
        }
        *(_DWORD *)(v9 + 408) = v34;
        goto LABEL_27;
      }
    }
    v35 = *(_QWORD *)(v9 + 400);
LABEL_27:
    v151 = v17;
    *(_QWORD *)(v35 + 8LL * (v8 & 0x7FFFFFFF)) = sub_1DBA290(v8);
    v18 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8 * v151);
    sub_1DBB110((_QWORD *)v9, v18);
    v9 = *(_QWORD *)(a1 + 272);
  }
  v19 = *(_QWORD *)(v9 + 272);
  for ( i = a3; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v21 = *(_DWORD *)(v19 + 384);
  v22 = *(_QWORD *)(v19 + 368);
  if ( !v21 )
  {
LABEL_33:
    v24 = (__int64 *)(v22 + 16LL * v21);
    goto LABEL_13;
  }
  v23 = (v21 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v24 = (__int64 *)(v22 + 16LL * v23);
  v25 = *v24;
  if ( i != *v24 )
  {
    v38 = 1;
    while ( v25 != -8 )
    {
      v60 = v38 + 1;
      v23 = (v21 - 1) & (v38 + v23);
      v24 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( i == *v24 )
        goto LABEL_13;
      v38 = v60;
    }
    goto LABEL_33;
  }
LABEL_13:
  v26 = v24[1] & 0xFFFFFFFFFFFFFFF8LL;
  v148 = v26 | 4;
  v27 = (__int64 *)sub_1DB3C70((__int64 *)v18, v26 | 4);
  if ( v27 == (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8))
    || (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) > (*(_DWORD *)(v26 + 24) | 2u) )
  {
    k = 0;
  }
  else
  {
    k = v27[2];
  }
  v146 = v26 | 2;
  v28 = (__int64 *)sub_1DB3C70((__int64 *)v15, v26 | 2);
  if ( v28 == (__int64 *)(*(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8))
    || (*(_DWORD *)((*v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v28 >> 1) & 3) > (*(_DWORD *)(v26 + 24) | 1u) )
  {
    BUG();
  }
  v29 = *(_QWORD *)(v28[2] + 8);
  if ( (v29 & 6) == 0 )
    return 0;
  v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v30 )
    return 0;
  v31 = *(_QWORD *)(v30 + 16);
  v137 = v28[2];
  if ( !v31 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v31 + 16) + 10LL) & 0x20) != 0
    && (v48 = sub_1E16810(v31, *(_DWORD *)(v15 + 112), 0, 0, 0), v51 = *(_QWORD *)(v31 + 32) + 40LL * v48,
                                                                 !*(_BYTE *)v51)
    && (*(_BYTE *)(v51 + 3) & 0x10) != 0
    && (*(_WORD *)(v51 + 2) & 0xFF0) != 0
    && (v156 = sub_1E16AB0(v31, v48, 5LL * v48, v47, v49, v50),
        v157 = -1,
        (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, unsigned int *, unsigned int *))(**(_QWORD **)(a1 + 264)
                                                                                           + 176LL))(
          *(_QWORD *)(a1 + 264),
          v31,
          &v156,
          &v157))
    && (v134 = *(_DWORD *)(*(_QWORD *)(v31 + 32) + 40LL * v157 + 8), *(_DWORD *)(v18 + 112) == v134)
    && (sub_1E86030((__int64)&v158, v18, *(_QWORD *)(v137 + 8)), v32 = (unsigned __int8)v161, (_BYTE)v161)
    && !(unsigned __int8)sub_1DBCAA0(*(_QWORD *)(a1 + 272), v15, v137) )
  {
    v52 = v137;
    if ( *(_QWORD *)v15 != *(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8) )
    {
      v53 = *(__int64 **)v15;
      v138 = v26;
      do
      {
        if ( v52 == v53[2] )
        {
          v54 = *(__int64 **)v18;
          v55 = 24LL * *(unsigned int *)(v18 + 8);
          v56 = (__int64 *)(*(_QWORD *)v18 + v55);
          v57 = 0xAAAAAAAAAAAAAAABLL * (v55 >> 3);
          if ( v55 )
          {
            v58 = *(_QWORD *)v18;
            do
            {
              v59 = (__int64 *)(v58 + 8 * ((v57 >> 1) + (v57 & 0xFFFFFFFFFFFFFFFELL)));
              if ( (*(_DWORD *)((*v53 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v53 >> 1) & 3)) >= (*(_DWORD *)((*v59 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v59 >> 1) & 3) )
              {
                v58 = (__int64)(v59 + 3);
                v57 = v57 - (v57 >> 1) - 1;
              }
              else
              {
                v57 >>= 1;
              }
            }
            while ( v57 > 0 );
            if ( (__int64 *)v58 != v54 )
              v54 = (__int64 *)(v58 - 24);
          }
          if ( v56 != v54 )
          {
            v61 = *(_DWORD *)((v53[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v53[1] >> 1) & 3;
            do
            {
              v62 = *(_DWORD *)((*v54 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v54 >> 1) & 3;
              if ( v61 < v62 )
                break;
              if ( k != v54[2] )
              {
                v63 = *(_DWORD *)((*v53 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v53 >> 1) & 3;
                if ( v62 <= v63 )
                {
                  if ( v63 < (*(_DWORD *)((v54[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v54[1] >> 1) & 3) )
                    return 0;
                }
                else if ( v61 > v62 )
                {
                  return 0;
                }
              }
              v54 += 3;
            }
            while ( v56 != v54 );
          }
        }
        v53 += 3;
      }
      while ( (__int64 *)(*(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8)) != v53 );
      v26 = v138;
      v32 = (unsigned __int8)v32;
    }
    v64 = *(_QWORD *)(a1 + 248);
    v65 = *(unsigned int *)(v15 + 112);
    if ( (int)v65 < 0 )
      v66 = *(_QWORD *)(*(_QWORD *)(v64 + 24) + 16 * (v65 & 0x7FFFFFFF) + 8);
    else
      v66 = *(_QWORD *)(*(_QWORD *)(v64 + 272) + 8 * v65);
    while ( v66 )
    {
      if ( (*(_BYTE *)(v66 + 3) & 0x10) == 0 && (*(_BYTE *)(v66 + 4) & 8) == 0 )
      {
        v123 = v31;
        v118 = v32;
        v133 = v26;
        v106 = v52;
        v127 = v18;
        v107 = v66;
LABEL_149:
        v108 = *(_QWORD *)(v107 + 16);
        v143 = *(_QWORD *)(v108 + 32);
        v109 = sub_1E85F30(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), v108);
        v110 = (__int64 *)sub_1DB3C70((__int64 *)v15, v109);
        if ( v110 != (__int64 *)(*(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8))
          && (*(_DWORD *)((*v110 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v110 >> 1) & 3)) <= (*(_DWORD *)((v109 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v109 >> 1) & 3)
          && v110[2] == v106 )
        {
          v111 = *(_QWORD *)(v108 + 32) + 0xFFFFFFF800000008LL * (unsigned int)((v107 - v143) >> 3);
          if ( !*(_BYTE *)v111 && (*(_BYTE *)(v111 + 3) & 0x10) == 0 && (*(_WORD *)(v111 + 2) & 0xFF0) != 0 )
            return 0;
        }
        while ( 1 )
        {
          v107 = *(_QWORD *)(v107 + 32);
          if ( !v107 )
            break;
          if ( (*(_BYTE *)(v107 + 3) & 0x10) == 0 && (*(_BYTE *)(v107 + 4) & 8) == 0 )
            goto LABEL_149;
        }
        v52 = v106;
        v18 = v127;
        v26 = v133;
        v31 = v123;
        v32 = v118;
        break;
      }
      v66 = *(_QWORD *)(v66 + 32);
    }
    v139 = v52;
    v129 = *(_QWORD *)(v31 + 24);
    v67 = sub_1F3ADB0(*(_QWORD *)(a1 + 264), v31, 0, v156, v157);
    v70 = v139;
    v71 = v67;
    if ( !v67 )
      return 0;
    v72 = *(_DWORD *)(v15 + 112);
    if ( v72 < 0 )
    {
      v112 = *(_DWORD *)(v18 + 112);
      if ( v112 < 0 )
      {
        v128 = v139;
        v144 = v71;
        v113 = sub_1E69410(
                 *(__int64 **)(a1 + 248),
                 v112,
                 *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16LL * (v72 & 0x7FFFFFFF))
               & 0xFFFFFFFFFFFFFFF8LL,
                 0);
        v71 = v144;
        v70 = v128;
        if ( !v113 )
          return 0;
      }
    }
    if ( v31 != v71 )
    {
      v119 = v70;
      v124 = v71;
      sub_1EDE6E0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), v31, v71);
      v140 = v129 + 16;
      sub_1DD5BA0((__int64 *)(v129 + 16), v124);
      v70 = v119;
      v73 = *(_QWORD *)v124;
      v74 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v124 + 8) = v31;
      *(_QWORD *)v124 = v74 | v73 & 7;
      *(_QWORD *)(v74 + 8) = v124;
      v75 = *(_QWORD *)v31 & 7LL | v124;
      v76 = v31;
      *(_QWORD *)v31 = v75;
      if ( (v75 & 4) == 0 && (*(_BYTE *)(v31 + 46) & 8) != 0 )
      {
        do
          v76 = *(_QWORD *)(v76 + 8);
        while ( (*(_BYTE *)(v76 + 46) & 8) != 0 );
      }
      if ( v31 != *(_QWORD *)(v76 + 8) )
      {
        v130 = v18;
        v77 = *(_QWORD *)(v76 + 8);
        v125 = v32;
        v78 = v26;
        v79 = v15;
        do
        {
          v80 = (_QWORD *)v31;
          v31 = *(_QWORD *)(v31 + 8);
          v120 = v70;
          sub_1DD5BC0(v140, (__int64)v80);
          v81 = (unsigned __int64 *)v80[1];
          v82 = *v80 & 0xFFFFFFFFFFFFFFF8LL;
          *v81 = v82 | *v81 & 7;
          *(_QWORD *)(v82 + 8) = v81;
          *v80 &= 7uLL;
          v80[1] = 0;
          sub_1DD5C20(v140);
          v70 = v120;
        }
        while ( v31 != v77 );
        v15 = v79;
        v18 = v130;
        v26 = v78;
        v32 = v125;
      }
    }
    v83 = *(_QWORD *)(a1 + 248);
    v84 = *(unsigned int *)(v15 + 112);
    if ( (int)v84 < 0 )
      v85 = *(_QWORD *)(*(_QWORD *)(v83 + 24) + 16 * (v84 & 0x7FFFFFFF) + 8);
    else
      v85 = *(_QWORD *)(*(_QWORD *)(v83 + 272) + 8 * v84);
    if ( v85 )
    {
      if ( (*(_BYTE *)(v85 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v85 = *(_QWORD *)(v85 + 32);
          if ( !v85 )
            break;
          if ( (*(_BYTE *)(v85 + 3) & 0x10) == 0 )
            goto LABEL_103;
        }
      }
      else
      {
LABEL_103:
        v131 = v26;
        v141 = v70;
        v126 = v32;
        while ( 1 )
        {
          for ( j = *(_QWORD *)(v85 + 32); j; j = *(_QWORD *)(j + 32) )
          {
            if ( (*(_BYTE *)(j + 3) & 0x10) == 0 )
              break;
          }
          if ( (*(_BYTE *)(v85 + 4) & 1) == 0 )
          {
            v87 = *(_QWORD *)(v85 + 16);
            if ( **(_WORD **)(v87 + 16) == 12 )
            {
              sub_1E310D0(v85, v134);
            }
            else
            {
              v121 = v85;
              v88 = sub_1E85F30(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), *(_QWORD *)(v85 + 16))
                  & 0xFFFFFFFFFFFFFFF8LL;
              v89 = (__int64 *)sub_1DB3C70((__int64 *)v15, v88 | 2);
              v85 = v121;
              v90 = *(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8);
              if ( v89 != (__int64 *)v90
                && (*(_DWORD *)((*v89 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v89 >> 1) & 3) <= (*(_DWORD *)(v88 + 24) | 1u) )
              {
                v90 = (__int64)v89;
              }
              if ( *(_QWORD *)(v90 + 16) == v141 )
              {
                *(_BYTE *)(v121 + 3) &= ~0x40u;
                if ( v134 <= 0 )
                  sub_1E310D0(v121, v134);
                else
                  sub_1E311F0(v121, (unsigned int)v134, *(_QWORD *)(a1 + 256));
                if ( a3 != v87 && **(_WORD **)(v87 + 16) == 15 )
                {
                  v101 = *(_DWORD **)(v87 + 32);
                  if ( *(_DWORD *)(v18 + 112) == v101[2] && (*v101 & 0xFFF00) == 0 )
                  {
                    v122 = v88 | 4;
                    v102 = sub_1EDA640(v18, v88 | 4);
                    if ( v102 )
                    {
                      v103 = sub_1DB4840(v18, v102, k);
                      v104 = *(_QWORD *)(v18 + 104);
                      for ( k = v103; v104; v104 = *(_QWORD *)(v104 + 104) )
                      {
                        v117 = sub_1EDA640(v104, v122);
                        if ( v117 )
                        {
                          v105 = sub_1EDA640(v104, v148);
                          sub_1DB4840(v104, v117, v105);
                        }
                      }
                      sub_1ED8E30(a1, v87);
                    }
                  }
                }
              }
            }
          }
          if ( !j )
            break;
          v85 = j;
        }
        v26 = v131;
        v70 = v141;
        v32 = v126;
      }
    }
    if ( *(_QWORD *)(v18 + 104) )
    {
      v91 = *(_QWORD *)(v15 + 104);
      v153 = (__int64 *)(*(_QWORD *)(a1 + 272) + 296LL);
      if ( !v91 )
      {
        v136 = v70;
        v145 = sub_1E69F40(*(_QWORD *)(a1 + 248), *(_DWORD *)(v15 + 112));
        v91 = sub_145CBF0(v153, 120, 16);
        *(_QWORD *)v91 = v91 + 16;
        *(_QWORD *)(v91 + 64) = v91 + 80;
        *(_QWORD *)(v91 + 8) = 0x200000000LL;
        *(_QWORD *)(v91 + 72) = 0x200000000LL;
        *(_QWORD *)(v91 + 96) = 0;
        sub_1EDCA90(v91, (__int64 *)v15, v153, v114, v115);
        v70 = v136;
        *(_DWORD *)(v91 + 112) = v145;
        *(_QWORD *)(v91 + 104) = *(_QWORD *)(v15 + 104);
        *(_QWORD *)(v15 + 104) = v91;
      }
      v142 = v15;
      v135 = v70;
      v132 = v32;
      v92 = v91;
      do
      {
        v93 = 0;
        v94 = (__int64 *)sub_1DB3C70((__int64 *)v92, v146);
        if ( v94 != (__int64 *)(*(_QWORD *)v92 + 24LL * *(unsigned int *)(v92 + 8))
          && (*(_DWORD *)((*v94 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v94 >> 1) & 3) <= (*(_DWORD *)(v26 + 24)
                                                                                                 | 1u) )
        {
          v93 = (__int64 *)v94[2];
        }
        v160 = 0;
        v95 = (__int64 **)sub_22077B0(32);
        if ( v95 )
        {
          v95[1] = (__int64 *)v92;
          v95[3] = v93;
          *v95 = v153;
          v95[2] = (__int64 *)v148;
        }
        v158 = v95;
        v161 = sub_1EDD620;
        v160 = sub_1ED8380;
        sub_1DB5D80(v18, (__int64)v153, *(unsigned int *)(v92 + 112), (__int64)&v158);
        if ( v160 )
          v160((__m128i **)&v158, (const __m128i **)&v158, 3);
        v92 = *(_QWORD *)(v92 + 104);
      }
      while ( v92 );
      v15 = v142;
      v70 = v135;
      v32 = v132;
    }
    *(_QWORD *)(k + 8) = *(_QWORD *)(v70 + 8);
    v96 = *(__int64 **)v15;
    v97 = *(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8);
    if ( *(_QWORD *)v15 != v97 )
    {
      v154 = v32;
      v98 = v15;
      v99 = v70;
      do
      {
        if ( v99 == v96[2] )
        {
          v100 = (__int64 **)*v96;
          v159 = v96[1];
          v160 = (__int64 (__fastcall *)(__m128i **, const __m128i **, int))k;
          *((_QWORD *)&v116 + 1) = v159;
          *(_QWORD *)&v116 = v100;
          v158 = v100;
          sub_1DB8610(v18, k, v159, v68, v85, v69, v116, k);
        }
        v96 += 3;
      }
      while ( (__int64 *)v97 != v96 );
      v70 = v99;
      v15 = v98;
      v32 = v154;
    }
    sub_1DBEA10(*(_QWORD *)(a1 + 272), v15, *(_QWORD *)(v70 + 8));
  }
  else
  {
    return 0;
  }
  return v32;
}
