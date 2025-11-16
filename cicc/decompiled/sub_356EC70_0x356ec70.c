// Function: sub_356EC70
// Address: 0x356ec70
//
__int64 __fastcall sub_356EC70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // r9
  __int64 v9; // r8
  int v10; // r11d
  _QWORD *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  __int64 *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  bool v18; // zf
  int v19; // eax
  unsigned __int64 i; // rdx
  int v21; // eax
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdi
  const void *v25; // r13
  unsigned __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rsi
  unsigned __int64 v29; // rax
  int v30; // esi
  unsigned int v31; // r13d
  int v32; // r11d
  _QWORD *v33; // rdx
  unsigned int v34; // edi
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 *v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // r12
  int v41; // esi
  int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // ecx
  int v45; // eax
  __int64 v46; // rdi
  __int64 v47; // rax
  int v48; // esi
  __int64 v49; // rdi
  int v50; // esi
  unsigned int v51; // ecx
  __int64 *v52; // rax
  __int64 v53; // r10
  __int64 v54; // rax
  __int64 *v55; // rdx
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  int v60; // eax
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // rdi
  _QWORD *v64; // r9
  unsigned int v65; // r15d
  int v66; // r11d
  __int64 v67; // rsi
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  int v70; // eax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 *v74; // r9
  __int64 v75; // r14
  int v76; // edx
  __int64 j; // rax
  __int64 v78; // r12
  unsigned __int64 v79; // rcx
  int v80; // r13d
  _DWORD *v81; // r12
  __int64 v82; // rax
  __int64 *v83; // r12
  __int64 *v84; // r14
  __int64 v85; // rdi
  int v86; // r15d
  unsigned int v87; // edx
  __int64 *v88; // rax
  __int64 v89; // r11
  __int64 v90; // r15
  __int64 v91; // rax
  unsigned __int64 v92; // rcx
  unsigned int v93; // esi
  int v94; // eax
  int v95; // edi
  __int64 v96; // rsi
  unsigned int v97; // eax
  int v98; // edx
  __int64 v99; // rax
  int v100; // eax
  int v102; // eax
  int v103; // eax
  int v104; // edi
  __int64 v105; // rsi
  __int64 *v106; // r10
  int v107; // r11d
  unsigned int v108; // eax
  int v109; // edx
  __int64 v110; // rax
  int v111; // r11d
  _QWORD *v112; // r10
  int v113; // r9d
  int v114; // r11d
  int v115; // edi
  int v116; // edx
  int v117; // r11d
  int v118; // r11d
  __int64 v119; // r10
  __int64 v120; // rdi
  int v121; // esi
  _QWORD *v122; // rcx
  int v123; // r10d
  int v124; // r10d
  unsigned int v125; // r14d
  int v126; // esi
  __int64 v127; // rdi
  __int64 v129; // [rsp+38h] [rbp-328h]
  __int64 *v130; // [rsp+40h] [rbp-320h]
  __int64 v131; // [rsp+48h] [rbp-318h]
  __int64 *v132; // [rsp+50h] [rbp-310h]
  __int64 v133; // [rsp+50h] [rbp-310h]
  __int64 v134; // [rsp+58h] [rbp-308h]
  __int64 v135; // [rsp+58h] [rbp-308h]
  __int64 v136; // [rsp+58h] [rbp-308h]
  int v137; // [rsp+58h] [rbp-308h]
  __int64 v138; // [rsp+58h] [rbp-308h]
  __int64 *v139; // [rsp+58h] [rbp-308h]
  __int64 v140; // [rsp+58h] [rbp-308h]
  __int64 v141; // [rsp+58h] [rbp-308h]
  __int64 v142; // [rsp+58h] [rbp-308h]
  __int64 v143; // [rsp+58h] [rbp-308h]
  _BYTE *v144; // [rsp+60h] [rbp-300h] BYREF
  __int64 v145; // [rsp+68h] [rbp-2F8h]
  _BYTE v146[80]; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE *v147; // [rsp+C0h] [rbp-2A0h] BYREF
  __int64 v148; // [rsp+C8h] [rbp-298h]
  _BYTE v149[80]; // [rsp+D0h] [rbp-290h] BYREF
  _BYTE *v150; // [rsp+120h] [rbp-240h] BYREF
  __int64 v151; // [rsp+128h] [rbp-238h]
  _BYTE v152[560]; // [rsp+130h] [rbp-230h] BYREF

  v144 = v146;
  v145 = 0xA00000000LL;
  v150 = v152;
  v151 = 0x4000000000LL;
  v130 = (__int64 *)(a1 + 56);
  v5 = sub_A777F0(0x40u, (__int64 *)(a1 + 56));
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = a2;
    *(_DWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_DWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
  }
  v7 = *(_DWORD *)(a1 + 48);
  v129 = a1 + 24;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_158;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)(a1 + 32);
  v10 = 1;
  v11 = 0;
  LODWORD(v12) = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v9 + 16LL * (unsigned int)v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_5:
    v15 = v13 + 1;
    goto LABEL_6;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (unsigned int)v8 & ((_DWORD)v12 + v10);
    v13 = (_QWORD *)(v9 + 16 * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_5;
    ++v10;
  }
  v115 = *(_DWORD *)(a1 + 40);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)(a1 + 24);
  v116 = v115 + 1;
  if ( 4 * (v115 + 1) >= 3 * v7 )
  {
LABEL_158:
    sub_356EA90(v129, 2 * v7);
    v117 = *(_DWORD *)(a1 + 48);
    if ( v117 )
    {
      v118 = v117 - 1;
      v119 = *(_QWORD *)(a1 + 32);
      v8 = v118 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v116 = *(_DWORD *)(a1 + 40) + 1;
      v11 = (_QWORD *)(v119 + 16 * v8);
      v120 = *v11;
      if ( *v11 == a2 )
        goto LABEL_149;
      v121 = 1;
      v122 = 0;
      while ( v120 != -4096 )
      {
        if ( !v122 && v120 == -8192 )
          v122 = v11;
        v9 = (unsigned int)(v121 + 1);
        v8 = v118 & (unsigned int)(v121 + v8);
        v11 = (_QWORD *)(v119 + 16LL * (unsigned int)v8);
        v120 = *v11;
        if ( *v11 == a2 )
          goto LABEL_149;
        ++v121;
      }
LABEL_162:
      if ( v122 )
        v11 = v122;
      goto LABEL_149;
    }
LABEL_189:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 44) - v116 <= v7 >> 3 )
  {
    sub_356EA90(v129, v7);
    v123 = *(_DWORD *)(a1 + 48);
    if ( v123 )
    {
      v124 = v123 - 1;
      v8 = *(_QWORD *)(a1 + 32);
      v122 = 0;
      v125 = v124 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v116 = *(_DWORD *)(a1 + 40) + 1;
      v126 = 1;
      v11 = (_QWORD *)(v8 + 16LL * v125);
      v127 = *v11;
      if ( *v11 == a2 )
        goto LABEL_149;
      while ( v127 != -4096 )
      {
        if ( v127 == -8192 && !v122 )
          v122 = v11;
        v9 = (unsigned int)(v126 + 1);
        v125 = v124 & (v126 + v125);
        v11 = (_QWORD *)(v8 + 16LL * v125);
        v127 = *v11;
        if ( *v11 == a2 )
          goto LABEL_149;
        ++v126;
      }
      goto LABEL_162;
    }
    goto LABEL_189;
  }
LABEL_149:
  *(_DWORD *)(a1 + 40) = v116;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 44);
  *v11 = a2;
  v15 = v11 + 1;
  *v15 = 0;
LABEL_6:
  *v15 = v6;
  v16 = (unsigned int)v151;
  v17 = (unsigned int)v151 + 1LL;
  if ( v17 > HIDWORD(v151) )
  {
    sub_C8D5F0((__int64)&v150, v152, v17, 8u, v9, v8);
    v16 = (unsigned int)v151;
  }
  *(_QWORD *)&v150[8 * v16] = v6;
  v147 = v149;
  v18 = (_DWORD)v151 == -1;
  v19 = v151 + 1;
  v148 = 0xA00000000LL;
  LODWORD(v151) = v151 + 1;
  if ( !v18 )
  {
    for ( i = 10; ; i = HIDWORD(v148) )
    {
      v24 = 0;
      v22 = *(_QWORD *)&v150[8 * v19 - 8];
      LODWORD(v151) = v19 - 1;
      LODWORD(v148) = 0;
      v25 = *(const void **)(*(_QWORD *)v22 + 64LL);
      v26 = *(unsigned int *)(*(_QWORD *)v22 + 72LL);
      v21 = 0;
      if ( v26 > i )
      {
        v135 = v22;
        sub_C8D5F0((__int64)&v147, v149, *(unsigned int *)(*(_QWORD *)v22 + 72LL), 8u, v22, v8);
        v21 = v148;
        v22 = v135;
        v24 = 8LL * (unsigned int)v148;
      }
      if ( 8 * v26 )
      {
        v134 = v22;
        memcpy(&v147[v24], v25, 8 * v26);
        v21 = v148;
        v22 = v134;
      }
      v23 = v26 + v21;
      LODWORD(v148) = v23;
      *(_DWORD *)(v22 + 40) = v23;
      if ( !v23 )
      {
        *(_QWORD *)(v22 + 48) = 0;
        goto LABEL_14;
      }
      v27 = *(_QWORD *)(a1 + 56);
      v28 = 8LL * v23;
      *(_QWORD *)(a1 + 136) += v28;
      v29 = (v27 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_QWORD *)(a1 + 64) >= v28 + v29 && v27 )
      {
        *(_QWORD *)(a1 + 56) = v28 + v29;
      }
      else
      {
        v143 = v22;
        v29 = sub_9D1E70((__int64)v130, v28, v28, 3);
        v22 = v143;
      }
      v30 = *(_DWORD *)(v22 + 40);
      *(_QWORD *)(v22 + 48) = v29;
      if ( v30 )
        break;
LABEL_14:
      v19 = v151;
      if ( !(_DWORD)v151 )
        goto LABEL_67;
    }
    v31 = 0;
    while ( 1 )
    {
      v39 = *(_DWORD *)(a1 + 48);
      v40 = *(_QWORD *)&v147[8 * v31];
      if ( !v39 )
      {
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_29;
      }
      v8 = *(_QWORD *)(a1 + 32);
      v32 = 1;
      v33 = 0;
      v34 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v35 = (_QWORD *)(v8 + 16LL * v34);
      v36 = *v35;
      if ( v40 != *v35 )
        break;
LABEL_24:
      v37 = v35 + 1;
      v38 = v35[1];
      if ( !v38 )
        goto LABEL_34;
      *(_QWORD *)(*(_QWORD *)(v22 + 48) + 8LL * v31) = v38;
LABEL_26:
      if ( *(_DWORD *)(v22 + 40) == ++v31 )
        goto LABEL_14;
    }
    while ( v36 != -4096 )
    {
      if ( v36 == -8192 && !v33 )
        v33 = v35;
      v34 = (v39 - 1) & (v32 + v34);
      v35 = (_QWORD *)(v8 + 16LL * v34);
      v36 = *v35;
      if ( v40 == *v35 )
        goto LABEL_24;
      ++v32;
    }
    if ( !v33 )
      v33 = v35;
    v60 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    v45 = v60 + 1;
    if ( 4 * v45 < 3 * v39 )
    {
      if ( v39 - *(_DWORD *)(a1 + 44) - v45 <= v39 >> 3 )
      {
        v138 = v22;
        sub_356EA90(v129, v39);
        v61 = *(_DWORD *)(a1 + 48);
        if ( !v61 )
          goto LABEL_189;
        v62 = v61 - 1;
        v63 = *(_QWORD *)(a1 + 32);
        v64 = 0;
        v65 = v62 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v22 = v138;
        v66 = 1;
        v45 = *(_DWORD *)(a1 + 40) + 1;
        v33 = (_QWORD *)(v63 + 16LL * v65);
        v67 = *v33;
        if ( v40 != *v33 )
        {
          while ( v67 != -4096 )
          {
            if ( v64 || v67 != -8192 )
              v33 = v64;
            v65 = v62 & (v66 + v65);
            v67 = *(_QWORD *)(v63 + 16LL * v65);
            if ( v40 == v67 )
            {
              v33 = (_QWORD *)(v63 + 16LL * v65);
              goto LABEL_31;
            }
            ++v66;
            v64 = v33;
            v33 = (_QWORD *)(v63 + 16LL * v65);
          }
          if ( v64 )
            v33 = v64;
        }
      }
      goto LABEL_31;
    }
LABEL_29:
    v136 = v22;
    sub_356EA90(v129, 2 * v39);
    v41 = *(_DWORD *)(a1 + 48);
    if ( !v41 )
      goto LABEL_189;
    v42 = v41 - 1;
    v43 = *(_QWORD *)(a1 + 32);
    v22 = v136;
    v44 = v42 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v45 = *(_DWORD *)(a1 + 40) + 1;
    v33 = (_QWORD *)(v43 + 16LL * v44);
    v46 = *v33;
    if ( v40 != *v33 )
    {
      v111 = 1;
      v112 = 0;
      while ( v46 != -4096 )
      {
        if ( v46 == -8192 && !v112 )
          v112 = v33;
        v44 = v42 & (v111 + v44);
        v33 = (_QWORD *)(v43 + 16LL * v44);
        v46 = *v33;
        if ( v40 == *v33 )
          goto LABEL_31;
        ++v111;
      }
      if ( v112 )
        v33 = v112;
    }
LABEL_31:
    *(_DWORD *)(a1 + 40) = v45;
    if ( *v33 != -4096 )
      --*(_DWORD *)(a1 + 44);
    *v33 = v40;
    v37 = v33 + 1;
    *v37 = 0;
LABEL_34:
    v47 = *(_QWORD *)(a1 + 8);
    v48 = *(_DWORD *)(v47 + 24);
    v49 = *(_QWORD *)(v47 + 8);
    if ( v48 )
    {
      v50 = v48 - 1;
      v51 = v50 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v52 = (__int64 *)(v49 + 16LL * v51);
      v53 = *v52;
      if ( v40 == *v52 )
      {
LABEL_36:
        v131 = v22;
        v132 = v37;
        v137 = *((_DWORD *)v52 + 2);
        v54 = sub_A777F0(0x40u, v130);
        v55 = v132;
        v22 = v131;
        v56 = v54;
        if ( !v54 )
          goto LABEL_40;
        v57 = 0;
        *(_QWORD *)v56 = v40;
        *(_DWORD *)(v56 + 8) = v137;
        if ( v137 )
          v57 = v56;
        goto LABEL_39;
      }
      v70 = 1;
      while ( v53 != -4096 )
      {
        v113 = v70 + 1;
        v51 = v50 & (v70 + v51);
        v52 = (__int64 *)(v49 + 16LL * v51);
        v53 = *v52;
        if ( v40 == *v52 )
          goto LABEL_36;
        v70 = v113;
      }
    }
    v133 = v22;
    v139 = v37;
    v71 = sub_A777F0(0x40u, v130);
    v55 = v139;
    v22 = v133;
    v56 = v71;
    if ( !v71 )
    {
LABEL_40:
      *v55 = v56;
      *(_QWORD *)(*(_QWORD *)(v22 + 48) + 8LL * v31) = v56;
      if ( *(_DWORD *)(v56 + 8) )
      {
        v68 = (unsigned int)v145;
        v69 = (unsigned int)v145 + 1LL;
        if ( v69 > HIDWORD(v145) )
        {
          v142 = v22;
          sub_C8D5F0((__int64)&v144, v146, v69, 8u, v22, v8);
          v68 = (unsigned int)v145;
          v22 = v142;
        }
        *(_QWORD *)&v144[8 * v68] = v56;
        LODWORD(v145) = v145 + 1;
      }
      else
      {
        v58 = (unsigned int)v151;
        v59 = (unsigned int)v151 + 1LL;
        if ( v59 > HIDWORD(v151) )
        {
          v140 = v22;
          sub_C8D5F0((__int64)&v150, v152, v59, 8u, v22, v8);
          v58 = (unsigned int)v151;
          v22 = v140;
        }
        *(_QWORD *)&v150[8 * v58] = v56;
        LODWORD(v151) = v151 + 1;
      }
      goto LABEL_26;
    }
    *(_QWORD *)v71 = v40;
    *(_DWORD *)(v71 + 8) = 0;
    v57 = 0;
LABEL_39:
    *(_QWORD *)(v56 + 16) = v57;
    *(_DWORD *)(v56 + 24) = 0;
    *(_QWORD *)(v56 + 32) = 0;
    *(_DWORD *)(v56 + 40) = 0;
    *(_QWORD *)(v56 + 48) = 0;
    *(_QWORD *)(v56 + 56) = 0;
    goto LABEL_40;
  }
LABEL_67:
  v72 = sub_A777F0(0x40u, v130);
  v75 = v72;
  if ( v72 )
  {
    *(_QWORD *)v72 = 0;
    *(_DWORD *)(v72 + 8) = 0;
    *(_QWORD *)(v72 + 16) = 0;
    *(_DWORD *)(v72 + 24) = 0;
    *(_QWORD *)(v72 + 32) = 0;
    *(_DWORD *)(v72 + 40) = 0;
    *(_QWORD *)(v72 + 48) = 0;
    *(_QWORD *)(v72 + 56) = 0;
  }
  v76 = v145;
  for ( j = (unsigned int)v151; (_DWORD)v145; LODWORD(v151) = v151 + 1 )
  {
    v78 = *(_QWORD *)&v144[8 * v76 - 8];
    v79 = HIDWORD(v151);
    LODWORD(v145) = v76 - 1;
    *(_QWORD *)(v78 + 32) = v75;
    *(_DWORD *)(v78 + 24) = -1;
    if ( j + 1 > v79 )
    {
      sub_C8D5F0((__int64)&v150, v152, j + 1, 8u, v73, (__int64)v74);
      j = (unsigned int)v151;
    }
    *(_QWORD *)&v150[8 * j] = v78;
    v76 = v145;
    j = (unsigned int)(v151 + 1);
  }
  if ( (_DWORD)j )
  {
    v141 = v75;
    v80 = 1;
    while ( 1 )
    {
      v81 = *(_DWORD **)&v150[8 * j - 8];
      if ( v81[6] == -2 )
      {
        v109 = v81[2];
        v81[6] = v80;
        if ( !v109 )
        {
          v110 = *(unsigned int *)(a3 + 8);
          if ( v110 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v110 + 1, 8u, v73, (__int64)v74);
            v110 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v110) = v81;
          ++*(_DWORD *)(a3 + 8);
        }
        ++v80;
        j = (unsigned int)(v151 - 1);
        LODWORD(v151) = v151 - 1;
        goto LABEL_93;
      }
      v81[6] = -2;
      v82 = *(_QWORD *)v81;
      v83 = *(__int64 **)(*(_QWORD *)v81 + 112LL);
      v84 = &v83[*(unsigned int *)(v82 + 120)];
      if ( v84 != v83 )
        break;
LABEL_92:
      j = (unsigned int)v151;
LABEL_93:
      if ( !(_DWORD)j )
      {
        v75 = v141;
        v100 = v80;
        goto LABEL_95;
      }
    }
    while ( 1 )
    {
      v93 = *(_DWORD *)(a1 + 48);
      if ( !v93 )
        break;
      v73 = v93 - 1;
      v85 = *(_QWORD *)(a1 + 32);
      v74 = 0;
      v86 = 1;
      v87 = v73 & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
      v88 = (__int64 *)(v85 + 16LL * v87);
      v89 = *v88;
      if ( *v88 == *v83 )
      {
LABEL_79:
        v90 = v88[1];
        if ( v90 && !*(_DWORD *)(v90 + 24) )
        {
          v91 = (unsigned int)v151;
          v92 = HIDWORD(v151);
          *(_DWORD *)(v90 + 24) = -1;
          if ( v91 + 1 > v92 )
          {
            sub_C8D5F0((__int64)&v150, v152, v91 + 1, 8u, v73, (__int64)v74);
            v91 = (unsigned int)v151;
          }
          *(_QWORD *)&v150[8 * v91] = v90;
          LODWORD(v151) = v151 + 1;
        }
        if ( ++v83 == v84 )
          goto LABEL_92;
      }
      else
      {
        while ( v89 != -4096 )
        {
          if ( v89 == -8192 && !v74 )
            v74 = v88;
          v87 = v73 & (v86 + v87);
          v88 = (__int64 *)(v85 + 16LL * v87);
          v89 = *v88;
          if ( *v83 == *v88 )
            goto LABEL_79;
          ++v86;
        }
        if ( !v74 )
          v74 = v88;
        v102 = *(_DWORD *)(a1 + 40);
        ++*(_QWORD *)(a1 + 24);
        v98 = v102 + 1;
        if ( 4 * (v102 + 1) < 3 * v93 )
        {
          if ( v93 - *(_DWORD *)(a1 + 44) - v98 > v93 >> 3 )
            goto LABEL_89;
          sub_356EA90(v129, v93);
          v103 = *(_DWORD *)(a1 + 48);
          if ( !v103 )
            goto LABEL_189;
          v104 = v103 - 1;
          v105 = *(_QWORD *)(a1 + 32);
          v106 = 0;
          v107 = 1;
          v108 = (v103 - 1) & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
          v98 = *(_DWORD *)(a1 + 40) + 1;
          v74 = (__int64 *)(v105 + 16LL * v108);
          v73 = *v74;
          if ( *v74 == *v83 )
            goto LABEL_89;
          while ( v73 != -4096 )
          {
            if ( v73 == -8192 && !v106 )
              v106 = v74;
            v108 = v104 & (v107 + v108);
            v74 = (__int64 *)(v105 + 16LL * v108);
            v73 = *v74;
            if ( *v83 == *v74 )
              goto LABEL_89;
            ++v107;
          }
          goto LABEL_115;
        }
LABEL_87:
        sub_356EA90(v129, 2 * v93);
        v94 = *(_DWORD *)(a1 + 48);
        if ( !v94 )
          goto LABEL_189;
        v95 = v94 - 1;
        v96 = *(_QWORD *)(a1 + 32);
        v97 = (v94 - 1) & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
        v98 = *(_DWORD *)(a1 + 40) + 1;
        v74 = (__int64 *)(v96 + 16LL * v97);
        v73 = *v74;
        if ( *v83 == *v74 )
          goto LABEL_89;
        v114 = 1;
        v106 = 0;
        while ( v73 != -4096 )
        {
          if ( v73 == -8192 && !v106 )
            v106 = v74;
          v97 = v95 & (v114 + v97);
          v74 = (__int64 *)(v96 + 16LL * v97);
          v73 = *v74;
          if ( *v83 == *v74 )
            goto LABEL_89;
          ++v114;
        }
LABEL_115:
        if ( v106 )
          v74 = v106;
LABEL_89:
        *(_DWORD *)(a1 + 40) = v98;
        if ( *v74 != -4096 )
          --*(_DWORD *)(a1 + 44);
        v99 = *v83++;
        v74[1] = 0;
        *v74 = v99;
        if ( v83 == v84 )
          goto LABEL_92;
      }
    }
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_87;
  }
  v100 = 1;
LABEL_95:
  *(_DWORD *)(v75 + 24) = v100;
  if ( v147 != v149 )
    _libc_free((unsigned __int64)v147);
  if ( v150 != v152 )
    _libc_free((unsigned __int64)v150);
  if ( v144 != v146 )
    _libc_free((unsigned __int64)v144);
  return v75;
}
