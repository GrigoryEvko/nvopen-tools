// Function: sub_1B205A0
// Address: 0x1b205a0
//
__int64 __fastcall sub_1B205A0(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // r15
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // edx
  _QWORD *v7; // r12
  __int64 v8; // rax
  unsigned int *v9; // r12
  __int64 v10; // r8
  __int64 v11; // r11
  unsigned int *v12; // r15
  __int64 v13; // rdi
  unsigned int v14; // ecx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // r13
  int v19; // esi
  int v20; // esi
  __int64 v21; // r9
  unsigned int v22; // ecx
  int v23; // edx
  __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // r13
  unsigned int v27; // esi
  _QWORD *v28; // rcx
  _QWORD *v29; // r14
  _QWORD *v30; // r15
  unsigned int v31; // eax
  _QWORD *v32; // r12
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rcx
  __int64 v36; // r9
  unsigned int v37; // edx
  _QWORD *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r14
  __int64 v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rsi
  int v44; // eax
  _QWORD *v45; // r10
  int v46; // edi
  int v47; // ecx
  int v48; // ecx
  __int64 v49; // rdi
  _QWORD *v50; // r9
  unsigned int v51; // r14d
  int v52; // r10d
  __int64 v53; // rsi
  __int64 v54; // rcx
  _QWORD *v55; // rbx
  unsigned __int64 v56; // rdi
  int v58; // r14d
  _QWORD *v59; // rdi
  int v60; // eax
  int v61; // edx
  __int64 v62; // rax
  int v63; // r10d
  _QWORD *v64; // r8
  _QWORD *v65; // r9
  int v66; // r10d
  unsigned int v67; // edx
  __int64 v68; // rcx
  int v69; // r9d
  _QWORD *v70; // r8
  int v71; // eax
  int v72; // eax
  int v73; // eax
  __int64 v74; // rsi
  unsigned int v75; // eax
  __int64 v76; // rcx
  int v77; // r11d
  _QWORD *v78; // r10
  int v79; // r10d
  int v80; // r10d
  __int64 v81; // r8
  unsigned int v82; // edx
  __int64 v83; // rcx
  int v84; // edi
  _QWORD *v85; // rsi
  int v86; // eax
  __int64 v87; // rsi
  int v88; // ecx
  int v89; // r11d
  unsigned int v90; // eax
  int v91; // r9d
  int v92; // r9d
  __int64 v93; // rdi
  _QWORD *v94; // rcx
  unsigned int v95; // r13d
  int v96; // esi
  __int64 v97; // rdx
  _QWORD *v98; // r13
  _QWORD *v99; // rax
  _QWORD *v100; // r12
  __int64 v101; // rax
  __int64 *v102; // r9
  _BYTE *v103; // r10
  unsigned int v104; // r14d
  unsigned int v105; // esi
  __int64 v106; // rdx
  __int64 v107; // rdi
  unsigned int v108; // eax
  _QWORD *v109; // r14
  __int64 v110; // rcx
  size_t v111; // rdx
  int v112; // eax
  int v113; // r8d
  __int64 v114; // r11
  unsigned int v115; // ecx
  int v116; // eax
  __int64 v117; // rsi
  _QWORD *v118; // rdi
  _QWORD *v119; // r11
  int v120; // eax
  int v121; // eax
  int v122; // r8d
  __int64 v123; // r11
  unsigned int v124; // ecx
  __int64 v125; // rsi
  int v126; // r14d
  _QWORD *v127; // r10
  int v128; // edi
  __int64 *v129; // r14
  int v130; // r10d
  __int64 v131; // [rsp+8h] [rbp-E8h]
  __int64 v132; // [rsp+10h] [rbp-E0h]
  __int64 v133; // [rsp+10h] [rbp-E0h]
  __int64 v134; // [rsp+18h] [rbp-D8h]
  int v135; // [rsp+18h] [rbp-D8h]
  __int64 v136; // [rsp+18h] [rbp-D8h]
  __int64 v137; // [rsp+20h] [rbp-D0h]
  __int64 v138; // [rsp+30h] [rbp-C0h]
  __int64 *v139; // [rsp+30h] [rbp-C0h]
  __int64 *v140; // [rsp+30h] [rbp-C0h]
  __int64 *v141; // [rsp+30h] [rbp-C0h]
  __int64 v142; // [rsp+38h] [rbp-B8h]
  __int64 *v143; // [rsp+38h] [rbp-B8h]
  _BYTE *v144; // [rsp+38h] [rbp-B8h]
  int v145; // [rsp+38h] [rbp-B8h]
  int v146; // [rsp+38h] [rbp-B8h]
  _BYTE *v147; // [rsp+38h] [rbp-B8h]
  int v148; // [rsp+38h] [rbp-B8h]
  __int64 *v149; // [rsp+40h] [rbp-B0h]
  __int64 v150; // [rsp+48h] [rbp-A8h]
  __int64 v151; // [rsp+48h] [rbp-A8h]
  __int64 v152; // [rsp+48h] [rbp-A8h]
  __int64 *v153; // [rsp+58h] [rbp-98h] BYREF
  __int64 v154; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v155; // [rsp+68h] [rbp-88h]
  __int64 v156; // [rsp+70h] [rbp-80h]
  unsigned int v157; // [rsp+78h] [rbp-78h]
  __int64 v158; // [rsp+80h] [rbp-70h]
  _BYTE *v159; // [rsp+88h] [rbp-68h] BYREF
  __int64 v160; // [rsp+90h] [rbp-60h]
  _BYTE dest[88]; // [rsp+98h] [rbp-58h] BYREF

  v150 = *(_QWORD *)(*(_QWORD *)(a1 + 488) + 8LL);
  v149 = (__int64 *)sub_157E9C0(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v153 = v149;
  v138 = sub_161C490(&v153, (__int64)"LVerDomain", 10, 0);
  v2 = *(_QWORD *)(v150 + 152);
  v131 = a1 + 424;
  v142 = v2 + 48LL * *(unsigned int *)(v150 + 160);
  if ( v2 == v142 )
    goto LABEL_18;
  v137 = v150;
  v3 = *(_QWORD *)(v150 + 152);
  do
  {
    v4 = *(_DWORD *)(a1 + 448);
    if ( v4 )
    {
      v5 = *(_QWORD *)(a1 + 432);
      v6 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v7 = (_QWORD *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == v3 )
        goto LABEL_5;
      v69 = 1;
      v70 = 0;
      while ( v8 != -8 )
      {
        if ( v70 || v8 != -16 )
          v7 = v70;
        v6 = (v4 - 1) & (v69 + v6);
        v8 = *(_QWORD *)(v5 + 16LL * v6);
        if ( v8 == v3 )
        {
          v7 = (_QWORD *)(v5 + 16LL * v6);
          goto LABEL_5;
        }
        ++v69;
        v70 = v7;
        v7 = (_QWORD *)(v5 + 16LL * v6);
      }
      v71 = *(_DWORD *)(a1 + 440);
      if ( v70 )
        v7 = v70;
      ++*(_QWORD *)(a1 + 424);
      v72 = v71 + 1;
      if ( 4 * v72 < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(a1 + 444) - v72 <= v4 >> 3 )
        {
          sub_1B1FA60(v131, v4);
          v91 = *(_DWORD *)(a1 + 448);
          if ( !v91 )
          {
LABEL_246:
            ++*(_DWORD *)(a1 + 440);
            BUG();
          }
          v92 = v91 - 1;
          v93 = *(_QWORD *)(a1 + 432);
          v94 = 0;
          v95 = v92 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v96 = 1;
          v72 = *(_DWORD *)(a1 + 440) + 1;
          v7 = (_QWORD *)(v93 + 16LL * v95);
          v97 = *v7;
          if ( v3 != *v7 )
          {
            while ( v97 != -8 )
            {
              if ( v97 == -16 && !v94 )
                v94 = v7;
              v95 = v92 & (v96 + v95);
              v7 = (_QWORD *)(v93 + 16LL * v95);
              v97 = *v7;
              if ( *v7 == v3 )
                goto LABEL_84;
              ++v96;
            }
            if ( v94 )
              v7 = v94;
          }
        }
        goto LABEL_84;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 424);
    }
    sub_1B1FA60(v131, 2 * v4);
    v79 = *(_DWORD *)(a1 + 448);
    if ( !v79 )
      goto LABEL_246;
    v80 = v79 - 1;
    v81 = *(_QWORD *)(a1 + 432);
    v82 = v80 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v72 = *(_DWORD *)(a1 + 440) + 1;
    v7 = (_QWORD *)(v81 + 16LL * v82);
    v83 = *v7;
    if ( *v7 != v3 )
    {
      v84 = 1;
      v85 = 0;
      while ( v83 != -8 )
      {
        if ( !v85 && v83 == -16 )
          v85 = v7;
        v82 = v80 & (v84 + v82);
        v7 = (_QWORD *)(v81 + 16LL * v82);
        v83 = *v7;
        if ( *v7 == v3 )
          goto LABEL_84;
        ++v84;
      }
      if ( v85 )
        v7 = v85;
    }
LABEL_84:
    *(_DWORD *)(a1 + 440) = v72;
    if ( *v7 != -8 )
      --*(_DWORD *)(a1 + 444);
    *v7 = v3;
    v7[1] = 0;
LABEL_5:
    v7[1] = sub_161C490(&v153, 0, 0, v138);
    v9 = *(unsigned int **)(v3 + 24);
    v151 = a1 + 392;
    if ( &v9[*(unsigned int *)(v3 + 32)] == v9 )
      goto LABEL_17;
    v10 = v3;
    v11 = v137;
    v12 = &v9[*(unsigned int *)(v3 + 32)];
    do
    {
      while ( 1 )
      {
        v17 = *(_DWORD *)(a1 + 416);
        v18 = *(_QWORD *)(*(_QWORD *)(v11 + 8) + ((unsigned __int64)*v9 << 6) + 16);
        if ( !v17 )
        {
          ++*(_QWORD *)(a1 + 392);
LABEL_11:
          v132 = v10;
          v134 = v11;
          sub_1B200C0(v151, 2 * v17);
          v19 = *(_DWORD *)(a1 + 416);
          if ( !v19 )
            goto LABEL_245;
          v20 = v19 - 1;
          v21 = *(_QWORD *)(a1 + 400);
          v11 = v134;
          v10 = v132;
          v22 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v23 = *(_DWORD *)(a1 + 408) + 1;
          v15 = (_QWORD *)(v21 + 16LL * v22);
          v24 = *v15;
          if ( v18 != *v15 )
          {
            v126 = 1;
            v127 = 0;
            while ( v24 != -8 )
            {
              if ( v24 != -16 || v127 )
                v15 = v127;
              v22 = v20 & (v126 + v22);
              v24 = *(_QWORD *)(v21 + 16LL * v22);
              if ( v18 == v24 )
              {
                v15 = (_QWORD *)(v21 + 16LL * v22);
                goto LABEL_13;
              }
              ++v126;
              v127 = v15;
              v15 = (_QWORD *)(v21 + 16LL * v22);
            }
            if ( v127 )
              v15 = v127;
          }
          goto LABEL_13;
        }
        v13 = *(_QWORD *)(a1 + 400);
        v14 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v15 = (_QWORD *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( v18 != *v15 )
          break;
LABEL_8:
        ++v9;
        v15[1] = v10;
        if ( v12 == v9 )
          goto LABEL_16;
      }
      v135 = 1;
      v45 = 0;
      while ( v16 != -8 )
      {
        if ( !v45 && v16 == -16 )
          v45 = v15;
        v14 = (v17 - 1) & (v135 + v14);
        v15 = (_QWORD *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( v18 == *v15 )
          goto LABEL_8;
        ++v135;
      }
      v46 = *(_DWORD *)(a1 + 408);
      if ( v45 )
        v15 = v45;
      ++*(_QWORD *)(a1 + 392);
      v23 = v46 + 1;
      if ( 4 * (v46 + 1) >= 3 * v17 )
        goto LABEL_11;
      if ( v17 - *(_DWORD *)(a1 + 412) - v23 <= v17 >> 3 )
      {
        v133 = v10;
        v136 = v11;
        sub_1B200C0(v151, v17);
        v47 = *(_DWORD *)(a1 + 416);
        if ( !v47 )
        {
LABEL_245:
          ++*(_DWORD *)(a1 + 408);
          BUG();
        }
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 400);
        v50 = 0;
        v51 = v48 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v11 = v136;
        v10 = v133;
        v52 = 1;
        v23 = *(_DWORD *)(a1 + 408) + 1;
        v15 = (_QWORD *)(v49 + 16LL * v51);
        v53 = *v15;
        if ( v18 != *v15 )
        {
          while ( v53 != -8 )
          {
            if ( v53 == -16 && !v50 )
              v50 = v15;
            v51 = v48 & (v52 + v51);
            v15 = (_QWORD *)(v49 + 16LL * v51);
            v53 = *v15;
            if ( v18 == *v15 )
              goto LABEL_13;
            ++v52;
          }
          if ( v50 )
            v15 = v50;
        }
      }
LABEL_13:
      *(_DWORD *)(a1 + 408) = v23;
      if ( *v15 != -8 )
        --*(_DWORD *)(a1 + 412);
      ++v9;
      v15[1] = 0;
      *v15 = v18;
      v15[1] = v10;
    }
    while ( v12 != v9 );
LABEL_16:
    v3 = v10;
LABEL_17:
    v3 += 48;
  }
  while ( v142 != v3 );
LABEL_18:
  v25 = *(unsigned int *)(a1 + 104);
  v26 = *(_QWORD **)(a1 + 96);
  v157 = 0;
  v27 = 0;
  v154 = 0;
  v28 = 0;
  v29 = 0;
  v25 *= 16;
  v156 = 0;
  v155 = 0;
  v30 = (_QWORD *)((char *)v26 + v25);
  if ( (_QWORD *)((char *)v26 + v25) == v26 )
    return j___libc_free_0(v29);
  while ( 2 )
  {
    if ( !v27 )
    {
      ++v154;
      goto LABEL_30;
    }
    v31 = (v27 - 1) & (((unsigned int)*v26 >> 9) ^ ((unsigned int)*v26 >> 4));
    v32 = &v28[7 * v31];
    v33 = *v32;
    if ( *v26 != *v32 )
    {
      v63 = 1;
      v64 = 0;
      while ( v33 != -8 )
      {
        if ( !v64 && v33 == -16 )
          v64 = v32;
        v31 = (v27 - 1) & (v63 + v31);
        v32 = &v28[7 * v31];
        v33 = *v32;
        if ( *v26 == *v32 )
          goto LABEL_21;
        ++v63;
      }
      if ( v64 )
        v32 = v64;
      ++v154;
      v44 = v156 + 1;
      if ( 4 * ((int)v156 + 1) >= 3 * v27 )
      {
LABEL_30:
        sub_1B20280((__int64)&v154, 2 * v27);
        if ( !v157 )
          goto LABEL_244;
        v42 = (v157 - 1) & (((unsigned int)*v26 >> 9) ^ ((unsigned int)*v26 >> 4));
        v32 = &v155[7 * v42];
        v43 = *v32;
        v44 = v156 + 1;
        if ( *v26 == *v32 )
          goto LABEL_32;
        v130 = 1;
        v65 = 0;
        while ( v43 != -8 )
        {
          if ( v43 == -16 && !v65 )
            v65 = v32;
          v42 = (v157 - 1) & (v42 + v130);
          v32 = &v155[7 * v42];
          v43 = *v32;
          if ( *v26 == *v32 )
            goto LABEL_32;
          ++v130;
        }
      }
      else
      {
        if ( v27 - (v44 + HIDWORD(v156)) > v27 >> 3 )
          goto LABEL_32;
        sub_1B20280((__int64)&v154, v27);
        if ( !v157 )
        {
LABEL_244:
          LODWORD(v156) = v156 + 1;
          BUG();
        }
        v65 = 0;
        v66 = 1;
        v67 = (v157 - 1) & (((unsigned int)*v26 >> 9) ^ ((unsigned int)*v26 >> 4));
        v32 = &v155[7 * v67];
        v68 = *v32;
        v44 = v156 + 1;
        if ( *v26 == *v32 )
          goto LABEL_32;
        while ( v68 != -8 )
        {
          if ( v68 == -16 && !v65 )
            v65 = v32;
          v67 = (v157 - 1) & (v67 + v66);
          v32 = &v155[7 * v67];
          v68 = *v32;
          if ( *v26 == *v32 )
            goto LABEL_32;
          ++v66;
        }
      }
      if ( v65 )
        v32 = v65;
LABEL_32:
      LODWORD(v156) = v44;
      if ( *v32 != -8 )
        --HIDWORD(v156);
      *v32 = *v26;
      v32[1] = v32 + 3;
      v32[2] = 0x400000000LL;
    }
LABEL_21:
    v34 = *(_DWORD *)(a1 + 448);
    if ( v34 )
    {
      v35 = v26[1];
      v36 = *(_QWORD *)(a1 + 432);
      v37 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v38 = (_QWORD *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( v35 == *v38 )
      {
        v40 = v38[1];
        goto LABEL_24;
      }
      v58 = 1;
      v59 = 0;
      while ( v39 != -8 )
      {
        if ( v39 != -16 || v59 )
          v38 = v59;
        v128 = v58 + 1;
        v37 = (v34 - 1) & (v58 + v37);
        v129 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v129;
        if ( v35 == *v129 )
        {
          v40 = v129[1];
          goto LABEL_24;
        }
        v58 = v128;
        v59 = v38;
        v38 = (_QWORD *)(v36 + 16LL * v37);
      }
      if ( !v59 )
        v59 = v38;
      v60 = *(_DWORD *)(a1 + 440);
      ++*(_QWORD *)(a1 + 424);
      v61 = v60 + 1;
      if ( 4 * (v60 + 1) < 3 * v34 )
      {
        if ( v34 - *(_DWORD *)(a1 + 444) - v61 <= v34 >> 3 )
        {
          sub_1B1FA60(a1 + 424, v34);
          v86 = *(_DWORD *)(a1 + 448);
          if ( !v86 )
          {
LABEL_248:
            ++*(_DWORD *)(a1 + 440);
            BUG();
          }
          v87 = v26[1];
          v88 = v86 - 1;
          v89 = 1;
          v78 = 0;
          v36 = *(_QWORD *)(a1 + 432);
          v90 = (v86 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
          v61 = *(_DWORD *)(a1 + 440) + 1;
          v59 = (_QWORD *)(v36 + 16LL * v90);
          v39 = *v59;
          if ( v87 != *v59 )
          {
            while ( v39 != -8 )
            {
              if ( !v78 && v39 == -16 )
                v78 = v59;
              v90 = v88 & (v89 + v90);
              v59 = (_QWORD *)(v36 + 16LL * v90);
              v39 = *v59;
              if ( v87 == *v59 )
                goto LABEL_63;
              ++v89;
            }
            goto LABEL_113;
          }
        }
        goto LABEL_63;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 424);
    }
    sub_1B1FA60(a1 + 424, 2 * v34);
    v73 = *(_DWORD *)(a1 + 448);
    if ( !v73 )
      goto LABEL_248;
    v74 = v26[1];
    LODWORD(v36) = v73 - 1;
    v39 = *(_QWORD *)(a1 + 432);
    v75 = (v73 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
    v61 = *(_DWORD *)(a1 + 440) + 1;
    v59 = (_QWORD *)(v39 + 16LL * v75);
    v76 = *v59;
    if ( v74 != *v59 )
    {
      v77 = 1;
      v78 = 0;
      while ( v76 != -8 )
      {
        if ( !v78 && v76 == -16 )
          v78 = v59;
        v75 = v36 & (v77 + v75);
        v59 = (_QWORD *)(v39 + 16LL * v75);
        v76 = *v59;
        if ( v74 == *v59 )
          goto LABEL_63;
        ++v77;
      }
LABEL_113:
      if ( v78 )
        v59 = v78;
    }
LABEL_63:
    *(_DWORD *)(a1 + 440) = v61;
    if ( *v59 != -8 )
      --*(_DWORD *)(a1 + 444);
    v62 = v26[1];
    v40 = 0;
    v59[1] = 0;
    *v59 = v62;
LABEL_24:
    v41 = *((unsigned int *)v32 + 4);
    if ( (unsigned int)v41 >= *((_DWORD *)v32 + 5) )
    {
      sub_16CD150((__int64)(v32 + 1), v32 + 3, 0, 8, v39, v36);
      v41 = *((unsigned int *)v32 + 4);
    }
    v26 += 2;
    *(_QWORD *)(v32[1] + 8 * v41) = v40;
    ++*((_DWORD *)v32 + 4);
    if ( v26 != v30 )
    {
      v28 = v155;
      v27 = v157;
      continue;
    }
    break;
  }
  v29 = v155;
  v54 = v157;
  if ( !(_DWORD)v156 )
    goto LABEL_48;
  v98 = &v155[7 * v157];
  if ( v98 == v155 )
    goto LABEL_48;
  v99 = v155;
  while ( 1 )
  {
    v100 = v99;
    if ( *v99 != -16 && *v99 != -8 )
      break;
    v99 += 7;
    if ( v98 == v99 )
      goto LABEL_48;
  }
  if ( v99 != v98 )
  {
    v152 = a1 + 456;
    while ( 1 )
    {
      v101 = *v100;
      v102 = 0;
      v159 = dest;
      v103 = dest;
      v158 = v101;
      v160 = 0x400000000LL;
      v104 = *((_DWORD *)v100 + 4);
      if ( !v104 || &v159 == v100 + 1 )
      {
        v105 = *(_DWORD *)(a1 + 480);
        if ( !v105 )
          goto LABEL_145;
      }
      else
      {
        v102 = (__int64 *)v104;
        v111 = 8LL * v104;
        if ( v104 <= 4
          || (sub_16CD150((__int64)&v159, dest, v104, 8, v39, v104),
              v103 = v159,
              v102 = (__int64 *)v104,
              (v111 = 8LL * *((unsigned int *)v100 + 4)) != 0) )
        {
          v143 = v102;
          memcpy(v103, (const void *)v100[1], v111);
          v103 = v159;
          v102 = v143;
        }
        v105 = *(_DWORD *)(a1 + 480);
        LODWORD(v160) = v104;
        if ( !v105 )
        {
LABEL_145:
          ++*(_QWORD *)(a1 + 456);
          goto LABEL_146;
        }
      }
      v106 = v158;
      v107 = *(_QWORD *)(a1 + 464);
      v108 = (v105 - 1) & (((unsigned int)v158 >> 9) ^ ((unsigned int)v158 >> 4));
      v109 = (_QWORD *)(v107 + 16LL * v108);
      v110 = *v109;
      if ( v158 != *v109 )
        break;
LABEL_132:
      v109[1] = sub_1627350(v149, (__int64 *)v103, v102, 0, 1);
      if ( v159 != dest )
        _libc_free((unsigned __int64)v159);
      v100 += 7;
      if ( v100 != v98 )
      {
        while ( *v100 == -8 || *v100 == -16 )
        {
          v100 += 7;
          if ( v98 == v100 )
            goto LABEL_138;
        }
        if ( v100 != v98 )
          continue;
      }
LABEL_138:
      v54 = v157;
      v29 = v155;
      goto LABEL_48;
    }
    v146 = 1;
    v119 = 0;
    while ( v110 != -8 )
    {
      if ( v119 || v110 != -16 )
        v109 = v119;
      v108 = (v105 - 1) & (v146 + v108);
      v141 = (__int64 *)(v107 + 16LL * v108);
      v110 = *v141;
      if ( v158 == *v141 )
      {
        v109 = (_QWORD *)(v107 + 16LL * v108);
        goto LABEL_132;
      }
      ++v146;
      v119 = v109;
      v109 = (_QWORD *)(v107 + 16LL * v108);
    }
    v120 = *(_DWORD *)(a1 + 472);
    if ( v119 )
      v109 = v119;
    ++*(_QWORD *)(a1 + 456);
    v116 = v120 + 1;
    if ( 4 * v116 >= 3 * v105 )
    {
LABEL_146:
      v139 = v102;
      v144 = v103;
      sub_1B1FA60(v152, 2 * v105);
      v112 = *(_DWORD *)(a1 + 480);
      if ( v112 )
      {
        v106 = v158;
        v113 = v112 - 1;
        v114 = *(_QWORD *)(a1 + 464);
        v103 = v144;
        v102 = v139;
        v115 = (v112 - 1) & (((unsigned int)v158 >> 9) ^ ((unsigned int)v158 >> 4));
        v116 = *(_DWORD *)(a1 + 472) + 1;
        v109 = (_QWORD *)(v114 + 16LL * v115);
        v117 = *v109;
        if ( *v109 == v158 )
        {
LABEL_161:
          *(_DWORD *)(a1 + 472) = v116;
          if ( *v109 != -8 )
            --*(_DWORD *)(a1 + 476);
          *v109 = v106;
          v109[1] = 0;
          goto LABEL_132;
        }
        v145 = 1;
        v118 = 0;
        while ( v117 != -8 )
        {
          if ( v117 == -16 && !v118 )
            v118 = v109;
          v115 = v113 & (v145 + v115);
          v109 = (_QWORD *)(v114 + 16LL * v115);
          v117 = *v109;
          if ( v158 == *v109 )
            goto LABEL_161;
          ++v145;
        }
LABEL_150:
        if ( v118 )
          v109 = v118;
        goto LABEL_161;
      }
    }
    else
    {
      if ( v105 - *(_DWORD *)(a1 + 476) - v116 > v105 >> 3 )
        goto LABEL_161;
      v140 = v102;
      v147 = v103;
      sub_1B1FA60(v152, v105);
      v121 = *(_DWORD *)(a1 + 480);
      if ( v121 )
      {
        v106 = v158;
        v122 = v121 - 1;
        v123 = *(_QWORD *)(a1 + 464);
        v103 = v147;
        v102 = v140;
        v124 = (v121 - 1) & (((unsigned int)v158 >> 9) ^ ((unsigned int)v158 >> 4));
        v116 = *(_DWORD *)(a1 + 472) + 1;
        v109 = (_QWORD *)(v123 + 16LL * v124);
        v125 = *v109;
        if ( *v109 == v158 )
          goto LABEL_161;
        v148 = 1;
        v118 = 0;
        while ( v125 != -8 )
        {
          if ( !v118 && v125 == -16 )
            v118 = v109;
          v124 = v122 & (v148 + v124);
          v109 = (_QWORD *)(v123 + 16LL * v124);
          v125 = *v109;
          if ( v158 == *v109 )
            goto LABEL_161;
          ++v148;
        }
        goto LABEL_150;
      }
    }
    ++*(_DWORD *)(a1 + 472);
    BUG();
  }
LABEL_48:
  if ( (_DWORD)v54 )
  {
    v55 = &v29[7 * v54];
    do
    {
      if ( *v29 != -8 && *v29 != -16 )
      {
        v56 = v29[1];
        if ( (_QWORD *)v56 != v29 + 3 )
          _libc_free(v56);
      }
      v29 += 7;
    }
    while ( v55 != v29 );
    v29 = v155;
  }
  return j___libc_free_0(v29);
}
