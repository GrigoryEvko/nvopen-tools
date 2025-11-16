// Function: sub_2109140
// Address: 0x2109140
//
__int64 __fastcall sub_2109140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r11
  __int64 v7; // rbx
  unsigned int v8; // esi
  int v9; // r8d
  __int64 v10; // rdi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  bool v15; // zf
  unsigned int v16; // eax
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // r8
  _QWORD *v27; // r9
  int v28; // edx
  __int64 v29; // r11
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rbx
  int v33; // r15d
  __int64 v34; // r12
  __int64 *v35; // rbx
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 *v38; // r13
  __int64 v39; // rdi
  unsigned int v40; // edx
  _QWORD *v41; // rax
  __int64 v42; // r11
  __int64 v43; // r14
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // eax
  int v47; // edi
  __int64 v48; // rsi
  unsigned int v49; // eax
  int v50; // edx
  __int64 v51; // rax
  __int64 v53; // rax
  __int64 v54; // r8
  int v55; // esi
  unsigned int v56; // r14d
  __int64 v57; // rdi
  unsigned int v58; // ecx
  _QWORD *v59; // rbx
  __int64 v60; // rdx
  __int64 v61; // rax
  unsigned int v62; // esi
  __int64 v63; // r12
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdi
  unsigned int v67; // ecx
  int v68; // edx
  __int64 v69; // rsi
  __int64 v70; // rax
  int v71; // edx
  __int64 v72; // rsi
  int v73; // edx
  unsigned int v74; // ecx
  __int64 *v75; // rax
  __int64 v76; // r9
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rdx
  int v80; // eax
  __int64 v81; // rdx
  int v82; // r11d
  _QWORD *v83; // r10
  int v84; // edi
  int v85; // ecx
  int v86; // ecx
  __int64 v87; // rdi
  _QWORD *v88; // r9
  int v89; // r10d
  unsigned int v90; // eax
  __int64 v91; // rsi
  int v92; // edx
  int v93; // r14d
  int v94; // eax
  int v95; // eax
  int v96; // edi
  __int64 v97; // rsi
  _QWORD *v98; // r10
  int v99; // r11d
  unsigned int v100; // eax
  __int64 v101; // rax
  int v102; // r10d
  int v103; // edi
  int v104; // r11d
  int v105; // r14d
  _QWORD *v106; // r10
  int v107; // edi
  int v108; // edx
  int v109; // eax
  int v110; // r13d
  __int64 v111; // r10
  __int64 v112; // rdi
  int v113; // esi
  _QWORD *v114; // rcx
  int v115; // r10d
  int v116; // r10d
  unsigned int v117; // r13d
  int v118; // esi
  __int64 v119; // rdi
  __int64 v121; // [rsp+40h] [rbp-320h]
  __int64 v122; // [rsp+48h] [rbp-318h]
  __int64 v123; // [rsp+48h] [rbp-318h]
  __int64 v124; // [rsp+48h] [rbp-318h]
  __int64 v125; // [rsp+48h] [rbp-318h]
  __int64 v126; // [rsp+50h] [rbp-310h]
  __int64 v127; // [rsp+50h] [rbp-310h]
  int v128; // [rsp+50h] [rbp-310h]
  __int64 v129; // [rsp+50h] [rbp-310h]
  unsigned int v130; // [rsp+50h] [rbp-310h]
  __int64 v131; // [rsp+50h] [rbp-310h]
  __int64 v132; // [rsp+50h] [rbp-310h]
  __int64 v133; // [rsp+50h] [rbp-310h]
  __int64 v134; // [rsp+50h] [rbp-310h]
  __int64 v135; // [rsp+50h] [rbp-310h]
  __int64 *v136; // [rsp+58h] [rbp-308h]
  __int64 v137; // [rsp+58h] [rbp-308h]
  _BYTE *v138; // [rsp+60h] [rbp-300h] BYREF
  __int64 v139; // [rsp+68h] [rbp-2F8h]
  _BYTE v140[80]; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE *v141; // [rsp+C0h] [rbp-2A0h] BYREF
  __int64 v142; // [rsp+C8h] [rbp-298h]
  _BYTE v143[80]; // [rsp+D0h] [rbp-290h] BYREF
  _BYTE *v144; // [rsp+120h] [rbp-240h] BYREF
  __int64 v145; // [rsp+128h] [rbp-238h]
  _BYTE v146[560]; // [rsp+130h] [rbp-230h] BYREF

  v138 = v140;
  v139 = 0xA00000000LL;
  v144 = v146;
  v145 = 0x4000000000LL;
  v136 = (__int64 *)(a1 + 56);
  v4 = sub_145CBF0((__int64 *)(a1 + 56), 64, 16);
  v6 = a1;
  *(_QWORD *)v4 = a2;
  v7 = v4;
  v8 = *(_DWORD *)(a1 + 48);
  *(_DWORD *)(v4 + 8) = 0;
  *(_QWORD *)(v4 + 16) = 0;
  *(_DWORD *)(v4 + 24) = 0;
  *(_QWORD *)(v4 + 32) = 0;
  *(_DWORD *)(v4 + 40) = 0;
  *(_QWORD *)(v4 + 48) = 0;
  *(_QWORD *)(v4 + 56) = 0;
  v121 = a1 + 24;
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = *(_QWORD *)(a1 + 32);
    LODWORD(v11) = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (_QWORD *)(v10 + 16LL * (unsigned int)v11);
    v13 = *v12;
    if ( *v12 == a2 )
      goto LABEL_3;
    v105 = 1;
    v106 = 0;
    while ( v13 != -8 )
    {
      if ( v13 == -16 && !v106 )
        v106 = v12;
      LODWORD(v5) = v105 + 1;
      v11 = v9 & (unsigned int)(v11 + v105);
      v12 = (_QWORD *)(v10 + 16 * v11);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v105;
    }
    v107 = *(_DWORD *)(v6 + 40);
    if ( v106 )
      v12 = v106;
    ++*(_QWORD *)(v6 + 24);
    v108 = v107 + 1;
    if ( 4 * (v107 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(v6 + 44) - v108 > v8 >> 3 )
        goto LABEL_141;
      v135 = v6;
      sub_2107BB0(v121, v8);
      v6 = v135;
      v115 = *(_DWORD *)(v135 + 48);
      if ( !v115 )
        goto LABEL_182;
      v116 = v115 - 1;
      v5 = *(_QWORD *)(v135 + 32);
      v114 = 0;
      v117 = v116 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v108 = *(_DWORD *)(v135 + 40) + 1;
      v118 = 1;
      v12 = (_QWORD *)(v5 + 16LL * v117);
      v119 = *v12;
      if ( *v12 == a2 )
        goto LABEL_141;
      while ( v119 != -8 )
      {
        if ( !v114 && v119 == -16 )
          v114 = v12;
        v9 = v118 + 1;
        v117 = v116 & (v118 + v117);
        v12 = (_QWORD *)(v5 + 16LL * v117);
        v119 = *v12;
        if ( *v12 == a2 )
          goto LABEL_141;
        ++v118;
      }
      goto LABEL_154;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  v134 = v6;
  sub_2107BB0(v121, 2 * v8);
  v6 = v134;
  v109 = *(_DWORD *)(v134 + 48);
  if ( !v109 )
    goto LABEL_182;
  v110 = v109 - 1;
  v111 = *(_QWORD *)(v134 + 32);
  LODWORD(v5) = (v109 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v108 = *(_DWORD *)(v134 + 40) + 1;
  v12 = (_QWORD *)(v111 + 16LL * (unsigned int)v5);
  v112 = *v12;
  if ( *v12 == a2 )
    goto LABEL_141;
  v113 = 1;
  v114 = 0;
  while ( v112 != -8 )
  {
    if ( !v114 && v112 == -16 )
      v114 = v12;
    v9 = v113 + 1;
    LODWORD(v5) = v110 & (v113 + v5);
    v12 = (_QWORD *)(v111 + 16LL * (unsigned int)v5);
    v112 = *v12;
    if ( *v12 == a2 )
      goto LABEL_141;
    ++v113;
  }
LABEL_154:
  if ( v114 )
    v12 = v114;
LABEL_141:
  *(_DWORD *)(v6 + 40) = v108;
  if ( *v12 != -8 )
    --*(_DWORD *)(v6 + 44);
  *v12 = a2;
  v12[1] = 0;
LABEL_3:
  v12[1] = v7;
  v14 = (unsigned int)v145;
  if ( (unsigned int)v145 >= HIDWORD(v145) )
  {
    v133 = v6;
    sub_16CD150((__int64)&v144, v146, 0, 8, v9, v5);
    v14 = (unsigned int)v145;
    v6 = v133;
  }
  *(_QWORD *)&v144[8 * v14] = v7;
  v141 = v143;
  v15 = (_DWORD)v145 == -1;
  v16 = v145 + 1;
  v142 = 0xA00000000LL;
  LODWORD(v145) = v145 + 1;
  if ( !v15 )
  {
    v17 = v6;
    while ( 1 )
    {
      v18 = v16--;
      v19 = *(_QWORD *)&v144[8 * v18 - 8];
      LODWORD(v145) = v16;
      LODWORD(v142) = 0;
      v20 = *(__int64 **)(*(_QWORD *)v19 + 64LL);
      v21 = *(__int64 **)(*(_QWORD *)v19 + 72LL);
      if ( v20 == v21 )
      {
        *(_DWORD *)(v19 + 40) = 0;
        goto LABEL_14;
      }
      v22 = 0;
      v23 = v19;
      do
      {
        if ( HIDWORD(v142) <= (unsigned int)v22 )
        {
          sub_16CD150((__int64)&v141, v143, 0, 8, v19, v5);
          v22 = (unsigned int)v142;
        }
        v24 = *v20++;
        *(_QWORD *)&v141[8 * v22] = v24;
        v22 = (unsigned int)(v142 + 1);
        LODWORD(v142) = v142 + 1;
      }
      while ( v21 != v20 );
      *(_DWORD *)(v23 + 40) = v22;
      v19 = v23;
      if ( !(_DWORD)v22 )
      {
        v16 = v145;
LABEL_14:
        *(_QWORD *)(v19 + 48) = 0;
        goto LABEL_15;
      }
      v53 = sub_145CBF0(v136, 8LL * (unsigned int)v22, 8);
      v54 = v23;
      v55 = *(_DWORD *)(v23 + 40);
      *(_QWORD *)(v23 + 48) = v53;
      if ( !v55 )
        goto LABEL_71;
      v56 = 0;
LABEL_57:
      v62 = *(_DWORD *)(v17 + 48);
      v63 = *(_QWORD *)&v141[8 * v56];
      if ( !v62 )
        break;
      LODWORD(v5) = v62 - 1;
      v57 = *(_QWORD *)(v17 + 32);
      v58 = (v62 - 1) & (((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9));
      v59 = (_QWORD *)(v57 + 16LL * v58);
      v60 = *v59;
      if ( v63 == *v59 )
        goto LABEL_54;
      v82 = 1;
      v83 = 0;
      while ( 1 )
      {
        if ( v60 == -8 )
        {
          v84 = *(_DWORD *)(v17 + 40);
          if ( v83 )
            v59 = v83;
          ++*(_QWORD *)(v17 + 24);
          v68 = v84 + 1;
          if ( 4 * (v84 + 1) < 3 * v62 )
          {
            if ( v62 - *(_DWORD *)(v17 + 44) - v68 > v62 >> 3 )
              goto LABEL_61;
            v123 = v54;
            v130 = ((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9);
            sub_2107BB0(v121, v62);
            v85 = *(_DWORD *)(v17 + 48);
            if ( v85 )
            {
              v86 = v85 - 1;
              v87 = *(_QWORD *)(v17 + 32);
              v88 = 0;
              v54 = v123;
              v89 = 1;
              v90 = v86 & v130;
              v68 = *(_DWORD *)(v17 + 40) + 1;
              v59 = (_QWORD *)(v87 + 16LL * (v86 & v130));
              v91 = *v59;
              if ( v63 != *v59 )
              {
                while ( v91 != -8 )
                {
                  if ( !v88 && v91 == -16 )
                    v88 = v59;
                  v90 = v86 & (v89 + v90);
                  v59 = (_QWORD *)(v87 + 16LL * v90);
                  v91 = *v59;
                  if ( v63 == *v59 )
                    goto LABEL_61;
                  ++v89;
                }
LABEL_88:
                if ( v88 )
                  v59 = v88;
              }
LABEL_61:
              *(_DWORD *)(v17 + 40) = v68;
              if ( *v59 != -8 )
                --*(_DWORD *)(v17 + 44);
              *v59 = v63;
              v59[1] = 0;
              goto LABEL_64;
            }
LABEL_181:
            v6 = v17;
LABEL_182:
            ++*(_DWORD *)(v6 + 40);
            BUG();
          }
LABEL_59:
          v127 = v54;
          sub_2107BB0(v121, 2 * v62);
          v64 = *(_DWORD *)(v17 + 48);
          if ( v64 )
          {
            v65 = v64 - 1;
            v66 = *(_QWORD *)(v17 + 32);
            v54 = v127;
            v67 = v65 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v68 = *(_DWORD *)(v17 + 40) + 1;
            v59 = (_QWORD *)(v66 + 16LL * v67);
            v69 = *v59;
            if ( v63 != *v59 )
            {
              v102 = 1;
              v88 = 0;
              while ( v69 != -8 )
              {
                if ( v69 == -16 && !v88 )
                  v88 = v59;
                v67 = v65 & (v102 + v67);
                v59 = (_QWORD *)(v66 + 16LL * v67);
                v69 = *v59;
                if ( v63 == *v59 )
                  goto LABEL_61;
                ++v102;
              }
              goto LABEL_88;
            }
            goto LABEL_61;
          }
          goto LABEL_181;
        }
        if ( v83 || v60 != -16 )
          v59 = v83;
        v58 = v5 & (v82 + v58);
        v60 = *(_QWORD *)(v57 + 16LL * v58);
        if ( v63 == v60 )
          break;
        ++v82;
        v83 = v59;
        v59 = (_QWORD *)(v57 + 16LL * v58);
      }
      v59 = (_QWORD *)(v57 + 16LL * v58);
LABEL_54:
      v61 = v59[1];
      if ( v61 )
      {
        *(_QWORD *)(*(_QWORD *)(v54 + 48) + 8LL * v56) = v61;
        goto LABEL_56;
      }
LABEL_64:
      v70 = *(_QWORD *)(v17 + 8);
      v71 = *(_DWORD *)(v70 + 24);
      if ( v71 )
      {
        v72 = *(_QWORD *)(v70 + 8);
        v73 = v71 - 1;
        v74 = v73 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
        v75 = (__int64 *)(v72 + 16LL * v74);
        v76 = *v75;
        if ( v63 == *v75 )
        {
LABEL_66:
          v122 = v54;
          v128 = *((_DWORD *)v75 + 2);
          v77 = sub_145CBF0(v136, 64, 16);
          v54 = v122;
          *(_QWORD *)v77 = v63;
          v78 = v77;
          *(_DWORD *)(v77 + 8) = v128;
          if ( v128 )
            goto LABEL_67;
          goto LABEL_75;
        }
        v80 = 1;
        while ( v76 != -8 )
        {
          v103 = v80 + 1;
          v74 = v73 & (v80 + v74);
          v75 = (__int64 *)(v72 + 16LL * v74);
          v76 = *v75;
          if ( v63 == *v75 )
            goto LABEL_66;
          v80 = v103;
        }
      }
      v129 = v54;
      v77 = sub_145CBF0(v136, 64, 16);
      *(_QWORD *)v77 = v63;
      *(_DWORD *)(v77 + 8) = 0;
      v54 = v129;
LABEL_75:
      v78 = 0;
LABEL_67:
      *(_QWORD *)(v77 + 16) = v78;
      *(_DWORD *)(v77 + 24) = 0;
      *(_QWORD *)(v77 + 32) = 0;
      *(_DWORD *)(v77 + 40) = 0;
      *(_QWORD *)(v77 + 48) = 0;
      *(_QWORD *)(v77 + 56) = 0;
      v59[1] = v77;
      *(_QWORD *)(*(_QWORD *)(v54 + 48) + 8LL * v56) = v77;
      if ( !*(_DWORD *)(v77 + 8) )
      {
        v79 = (unsigned int)v145;
        if ( (unsigned int)v145 >= HIDWORD(v145) )
        {
          v124 = v54;
          v131 = v77;
          sub_16CD150((__int64)&v144, v146, 0, 8, v54, v5);
          v79 = (unsigned int)v145;
          v54 = v124;
          v77 = v131;
        }
        ++v56;
        *(_QWORD *)&v144[8 * v79] = v77;
        LODWORD(v145) = v145 + 1;
        if ( *(_DWORD *)(v54 + 40) == v56 )
          goto LABEL_71;
        goto LABEL_57;
      }
      v81 = (unsigned int)v139;
      if ( (unsigned int)v139 >= HIDWORD(v139) )
      {
        v125 = v54;
        v132 = v77;
        sub_16CD150((__int64)&v138, v140, 0, 8, v54, v5);
        v81 = (unsigned int)v139;
        v54 = v125;
        v77 = v132;
      }
      *(_QWORD *)&v138[8 * v81] = v77;
      LODWORD(v139) = v139 + 1;
LABEL_56:
      if ( *(_DWORD *)(v54 + 40) != ++v56 )
        goto LABEL_57;
LABEL_71:
      v16 = v145;
LABEL_15:
      if ( !v16 )
      {
        v6 = v17;
        goto LABEL_17;
      }
    }
    ++*(_QWORD *)(v17 + 24);
    goto LABEL_59;
  }
LABEL_17:
  v126 = v6;
  v25 = sub_145CBF0(v136, 64, 16);
  v28 = v139;
  v29 = v126;
  *(_QWORD *)v25 = 0;
  v30 = v25;
  *(_DWORD *)(v25 + 8) = 0;
  *(_QWORD *)(v25 + 16) = 0;
  *(_DWORD *)(v25 + 24) = 0;
  *(_QWORD *)(v25 + 32) = 0;
  *(_DWORD *)(v25 + 40) = 0;
  *(_QWORD *)(v25 + 48) = 0;
  *(_QWORD *)(v25 + 56) = 0;
  v31 = (unsigned int)v145;
  if ( v28 )
  {
    do
    {
      v32 = *(_QWORD *)&v138[8 * v28 - 8];
      LODWORD(v139) = v28 - 1;
      *(_QWORD *)(v32 + 32) = v30;
      *(_DWORD *)(v32 + 24) = -1;
      if ( (unsigned int)v31 >= HIDWORD(v145) )
      {
        sub_16CD150((__int64)&v144, v146, 0, 8, v26, (int)v27);
        v31 = (unsigned int)v145;
      }
      *(_QWORD *)&v144[8 * v31] = v32;
      v28 = v139;
      v31 = (unsigned int)(v145 + 1);
      LODWORD(v145) = v145 + 1;
    }
    while ( (_DWORD)v139 );
    v29 = v126;
  }
  v33 = 1;
  if ( (_DWORD)v31 )
  {
    v137 = v30;
    v34 = v29;
    while ( 1 )
    {
      v35 = *(__int64 **)&v144[8 * v31 - 8];
      if ( *((_DWORD *)v35 + 6) == -2 )
      {
        v92 = *((_DWORD *)v35 + 2);
        *((_DWORD *)v35 + 6) = v33;
        if ( !v92 )
        {
          v101 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v101 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v26, (int)v27);
            v101 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v101) = v35;
          ++*(_DWORD *)(a3 + 8);
        }
        ++v33;
        v31 = (unsigned int)(v145 - 1);
        LODWORD(v145) = v145 - 1;
        goto LABEL_42;
      }
      v36 = *v35;
      *((_DWORD *)v35 + 6) = -2;
      v37 = *(__int64 **)(v36 + 88);
      v38 = *(__int64 **)(v36 + 96);
      if ( v38 != v37 )
        break;
LABEL_41:
      v31 = (unsigned int)v145;
LABEL_42:
      if ( !(_DWORD)v31 )
      {
        v30 = v137;
        goto LABEL_44;
      }
    }
    while ( 1 )
    {
      v45 = *(_DWORD *)(v34 + 48);
      if ( !v45 )
        break;
      LODWORD(v26) = v45 - 1;
      v39 = *(_QWORD *)(v34 + 32);
      v40 = (v45 - 1) & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
      v41 = (_QWORD *)(v39 + 16LL * v40);
      v42 = *v41;
      if ( *v41 == *v37 )
      {
LABEL_28:
        v43 = v41[1];
        if ( v43 && !*(_DWORD *)(v43 + 24) )
        {
          *(_DWORD *)(v43 + 24) = -1;
          v44 = (unsigned int)v145;
          if ( (unsigned int)v145 >= HIDWORD(v145) )
          {
            sub_16CD150((__int64)&v144, v146, 0, 8, v26, (int)v27);
            v44 = (unsigned int)v145;
          }
          *(_QWORD *)&v144[8 * v44] = v43;
          LODWORD(v145) = v145 + 1;
        }
        if ( v38 == ++v37 )
          goto LABEL_41;
      }
      else
      {
        v93 = 1;
        v27 = 0;
        while ( v42 != -8 )
        {
          if ( v42 == -16 && !v27 )
            v27 = v41;
          v40 = v26 & (v93 + v40);
          v41 = (_QWORD *)(v39 + 16LL * v40);
          v42 = *v41;
          if ( *v37 == *v41 )
            goto LABEL_28;
          ++v93;
        }
        if ( !v27 )
          v27 = v41;
        v94 = *(_DWORD *)(v34 + 40);
        ++*(_QWORD *)(v34 + 24);
        v50 = v94 + 1;
        if ( 4 * (v94 + 1) < 3 * v45 )
        {
          if ( v45 - *(_DWORD *)(v34 + 44) - v50 > v45 >> 3 )
            goto LABEL_38;
          sub_2107BB0(v121, v45);
          v95 = *(_DWORD *)(v34 + 48);
          if ( !v95 )
          {
LABEL_183:
            ++*(_DWORD *)(v34 + 40);
            BUG();
          }
          v96 = v95 - 1;
          v97 = *(_QWORD *)(v34 + 32);
          v98 = 0;
          v99 = 1;
          v100 = (v95 - 1) & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
          v50 = *(_DWORD *)(v34 + 40) + 1;
          v27 = (_QWORD *)(v97 + 16LL * v100);
          v26 = *v27;
          if ( *v27 == *v37 )
            goto LABEL_38;
          while ( v26 != -8 )
          {
            if ( !v98 && v26 == -16 )
              v98 = v27;
            v100 = v96 & (v99 + v100);
            v27 = (_QWORD *)(v97 + 16LL * v100);
            v26 = *v27;
            if ( *v37 == *v27 )
              goto LABEL_38;
            ++v99;
          }
          goto LABEL_102;
        }
LABEL_36:
        sub_2107BB0(v121, 2 * v45);
        v46 = *(_DWORD *)(v34 + 48);
        if ( !v46 )
          goto LABEL_183;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(v34 + 32);
        v49 = (v46 - 1) & (((unsigned int)*v37 >> 9) ^ ((unsigned int)*v37 >> 4));
        v50 = *(_DWORD *)(v34 + 40) + 1;
        v27 = (_QWORD *)(v48 + 16LL * v49);
        v26 = *v27;
        if ( *v27 == *v37 )
          goto LABEL_38;
        v104 = 1;
        v98 = 0;
        while ( v26 != -8 )
        {
          if ( !v98 && v26 == -16 )
            v98 = v27;
          v49 = v47 & (v104 + v49);
          v27 = (_QWORD *)(v48 + 16LL * v49);
          v26 = *v27;
          if ( *v37 == *v27 )
            goto LABEL_38;
          ++v104;
        }
LABEL_102:
        if ( v98 )
          v27 = v98;
LABEL_38:
        *(_DWORD *)(v34 + 40) = v50;
        if ( *v27 != -8 )
          --*(_DWORD *)(v34 + 44);
        v51 = *v37++;
        v27[1] = 0;
        *v27 = v51;
        if ( v38 == v37 )
          goto LABEL_41;
      }
    }
    ++*(_QWORD *)(v34 + 24);
    goto LABEL_36;
  }
LABEL_44:
  *(_DWORD *)(v30 + 24) = v33;
  if ( v141 != v143 )
    _libc_free((unsigned __int64)v141);
  if ( v144 != v146 )
    _libc_free((unsigned __int64)v144);
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
  return v30;
}
