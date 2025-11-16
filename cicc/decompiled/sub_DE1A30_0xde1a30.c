// Function: sub_DE1A30
// Address: 0xde1a30
//
__int64 __fastcall sub_DE1A30(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v4; // di
  int v5; // edi
  __int64 v6; // rsi
  int v7; // r9d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // r9
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rsi
  _QWORD *v23; // r9
  _QWORD *v24; // r15
  _QWORD *v25; // r14
  _QWORD *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 v37; // r9
  _QWORD *v38; // rdx
  _QWORD *v39; // r15
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  char v45; // dl
  _QWORD *v46; // rax
  _QWORD *v47; // r9
  _QWORD *v48; // r15
  _QWORD *v49; // r14
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  unsigned __int64 v55; // rax
  _QWORD *v56; // r9
  _QWORD *v57; // r15
  _QWORD *v58; // r14
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rsi
  unsigned int v67; // ecx
  __int64 v68; // r9
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rsi
  unsigned int v73; // ecx
  __int64 v74; // r9
  __int64 v75; // r14
  __int64 v76; // r15
  unsigned int v77; // eax
  unsigned int v78; // r12d
  __int64 v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rax
  _QWORD *v82; // rsi
  __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // rax
  unsigned int v86; // ecx
  _QWORD *v87; // rdx
  _QWORD *v88; // r10
  int v89; // edx
  int v90; // r11d
  __int64 v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rsi
  unsigned int v95; // ecx
  __int64 v96; // r9
  __int64 v97; // r14
  __int64 v98; // rax
  _QWORD *v99; // r9
  _QWORD *v100; // r15
  _QWORD *v101; // r14
  __int64 v102; // rax
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rdx
  __int64 v106; // rcx
  unsigned __int64 v107; // rax
  _QWORD *v108; // r9
  _QWORD *v109; // r15
  _QWORD *v110; // r14
  __int64 v111; // rax
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // rdx
  __int64 *v115; // rax
  int v116; // r10d
  __int64 v117; // rsi
  _QWORD *v118; // rdx
  _QWORD *v119; // r15
  __int64 v120; // r14
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // r8
  unsigned __int64 v128; // rax
  int v129; // eax
  int v130; // r10d
  int v131; // eax
  int v132; // r10d
  int v133; // eax
  int v134; // r10d
  int v135; // eax
  int v136; // r10d
  int v137; // eax
  int v138; // r10d
  __int64 v139; // [rsp+8h] [rbp-98h]
  __int64 v140; // [rsp+8h] [rbp-98h]
  __int64 v141; // [rsp+8h] [rbp-98h]
  __int64 v142; // [rsp+8h] [rbp-98h]
  __int64 v143; // [rsp+8h] [rbp-98h]
  _QWORD *v144; // [rsp+10h] [rbp-90h]
  _QWORD *v145; // [rsp+10h] [rbp-90h]
  _QWORD *v146; // [rsp+10h] [rbp-90h]
  _QWORD *v147; // [rsp+10h] [rbp-90h]
  _QWORD *v148; // [rsp+10h] [rbp-90h]
  _QWORD *v149; // [rsp+10h] [rbp-90h]
  char v150; // [rsp+10h] [rbp-90h]
  char v151; // [rsp+18h] [rbp-88h]
  char v152; // [rsp+18h] [rbp-88h]
  char v153; // [rsp+18h] [rbp-88h]
  char v154; // [rsp+18h] [rbp-88h]
  char v155; // [rsp+18h] [rbp-88h]
  char v156; // [rsp+18h] [rbp-88h]
  _QWORD *v157; // [rsp+18h] [rbp-88h]
  __int64 v158; // [rsp+28h] [rbp-78h] BYREF
  __int64 v159; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v160; // [rsp+40h] [rbp-60h] BYREF
  __int64 v161; // [rsp+48h] [rbp-58h]
  _QWORD v162[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 16);
  v158 = a2;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 24;
    v7 = 3;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 32);
    v6 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v13 )
      goto LABEL_17;
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v2 == *v9 )
    goto LABEL_4;
  v21 = 1;
  while ( v10 != -4096 )
  {
    v116 = v21 + 1;
    v8 = v7 & (v21 + v8);
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v2 == *v9 )
      goto LABEL_4;
    v21 = v116;
  }
  if ( (_BYTE)v5 )
  {
    v20 = 64;
    goto LABEL_18;
  }
  v13 = *(unsigned int *)(a1 + 32);
LABEL_17:
  v20 = 16 * v13;
LABEL_18:
  v9 = (__int64 *)(v6 + v20);
LABEL_4:
  v11 = 64;
  if ( !(_BYTE)v5 )
    v11 = 16LL * *(unsigned int *)(a1 + 32);
  if ( v9 == (__int64 *)(v6 + v11) )
  {
    switch ( *(_WORD *)(v2 + 24) )
    {
      case 0:
      case 1:
      case 8:
      case 0x10:
        goto LABEL_15;
      case 2:
        v91 = sub_DE1A30(a1, *(_QWORD *)(v2 + 32));
        if ( v91 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5200(*(_QWORD *)a1, v91, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_15;
      case 3:
        v70 = *(_QWORD *)(a1 + 88);
        v71 = *(unsigned int *)(v70 + 24);
        v72 = *(_QWORD *)(v70 + 8);
        if ( !(_DWORD)v71 )
          goto LABEL_63;
        v73 = (v71 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v18 = (__int64 *)(v72 + 16LL * v73);
        v74 = *v18;
        if ( v2 == *v18 )
          goto LABEL_62;
        v137 = 1;
        while ( 2 )
        {
          if ( v74 != -4096 )
          {
            v138 = v137 + 1;
            v73 = (v71 - 1) & (v137 + v73);
            v18 = (__int64 *)(v72 + 16LL * v73);
            v74 = *v18;
            if ( v2 != *v18 )
            {
              v137 = v138;
              continue;
            }
LABEL_62:
            if ( v18 != (__int64 *)(v72 + 16 * v71) )
              goto LABEL_14;
          }
          break;
        }
LABEL_63:
        v75 = *(_QWORD *)(v2 + 40);
        v76 = *(_QWORD *)(v2 + 32);
        v77 = sub_BCB060(v75);
        v78 = v77 >> 1;
        if ( (v77 & 0xE) != 0 || v78 <= 7 )
          goto LABEL_104;
        break;
      case 4:
        v64 = *(_QWORD *)(a1 + 88);
        v65 = *(unsigned int *)(v64 + 24);
        v66 = *(_QWORD *)(v64 + 8);
        if ( !(_DWORD)v65 )
          goto LABEL_58;
        v67 = (v65 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v18 = (__int64 *)(v66 + 16LL * v67);
        v68 = *v18;
        if ( v2 == *v18 )
          goto LABEL_57;
        v133 = 1;
        while ( 2 )
        {
          if ( v68 != -4096 )
          {
            v134 = v133 + 1;
            v67 = (v65 - 1) & (v133 + v67);
            v18 = (__int64 *)(v66 + 16LL * v67);
            v68 = *v18;
            if ( v2 != *v18 )
            {
              v133 = v134;
              continue;
            }
LABEL_57:
            if ( v18 != (__int64 *)(v66 + 16 * v65) )
              goto LABEL_14;
          }
          break;
        }
LABEL_58:
        v69 = sub_DE1A30(a1, *(_QWORD *)(v2 + 32));
        if ( v69 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5000(*(_QWORD *)a1, v69, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_15;
      case 5:
        v160 = v162;
        v161 = 0x200000000LL;
        v56 = *(_QWORD **)(v2 + 32);
        v147 = &v56[*(_QWORD *)(v2 + 40)];
        if ( v56 == v147 )
          goto LABEL_15;
        v154 = 0;
        v57 = *(_QWORD **)(v2 + 32);
        do
        {
          v58 = (_QWORD *)*v57;
          v26 = (_QWORD *)*v57;
          v59 = sub_DE1A30(a1, *v57);
          v62 = (unsigned int)v161;
          if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
          {
            v26 = v162;
            v142 = v59;
            sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 8u, v60, v61);
            v62 = (unsigned int)v161;
            v59 = v142;
          }
          v160[v62] = v59;
          v32 = v160;
          LODWORD(v161) = v161 + 1;
          ++v57;
          v154 |= v160[(unsigned int)v161 - 1] != (_QWORD)v58;
        }
        while ( v147 != v57 );
        if ( v154 )
        {
          v26 = &v160;
          v63 = sub_DC7EB0(
                  *(__int64 **)a1,
                  (__int64)&v160,
                  *(_DWORD *)(a1 + 96) & *(unsigned __int16 *)(v2 + 28) & 7u,
                  0);
          v32 = v160;
          v2 = (__int64)v63;
        }
        goto LABEL_31;
      case 6:
        v160 = v162;
        v161 = 0x200000000LL;
        v108 = *(_QWORD **)(v2 + 32);
        v149 = &v108[*(_QWORD *)(v2 + 40)];
        if ( v108 == v149 )
          goto LABEL_15;
        v156 = 0;
        v109 = *(_QWORD **)(v2 + 32);
        do
        {
          v110 = (_QWORD *)*v109;
          v26 = (_QWORD *)*v109;
          v111 = sub_DE1A30(a1, *v109);
          v114 = (unsigned int)v161;
          if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
          {
            v26 = v162;
            v140 = v111;
            sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 8u, v112, v113);
            v114 = (unsigned int)v161;
            v111 = v140;
          }
          v160[v114] = v111;
          v32 = v160;
          LODWORD(v161) = v161 + 1;
          ++v109;
          v156 |= v160[(unsigned int)v161 - 1] != (_QWORD)v110;
        }
        while ( v149 != v109 );
        if ( v156 )
        {
          v26 = &v160;
          v115 = sub_DC8BD0(
                   *(__int64 **)a1,
                   (__int64)&v160,
                   *(_DWORD *)(a1 + 96) & *(unsigned __int16 *)(v2 + 28) & 7u,
                   0);
          v32 = v160;
          v2 = (__int64)v115;
        }
        goto LABEL_31;
      case 7:
        v97 = sub_DE1A30(a1, *(_QWORD *)(v2 + 32));
        v98 = sub_DE1A30(a1, *(_QWORD *)(v2 + 40));
        if ( v97 != *(_QWORD *)(v2 + 32) || v98 != *(_QWORD *)(v2 + 40) )
          v2 = sub_DCB270(*(_QWORD *)a1, v97, v98);
        goto LABEL_15;
      case 9:
        v160 = v162;
        v161 = 0x200000000LL;
        v99 = *(_QWORD **)(v2 + 32);
        v148 = &v99[*(_QWORD *)(v2 + 40)];
        if ( v99 == v148 )
          goto LABEL_15;
        v155 = 0;
        v100 = *(_QWORD **)(v2 + 32);
        do
        {
          v101 = (_QWORD *)*v100;
          v26 = (_QWORD *)*v100;
          v102 = sub_DE1A30(a1, *v100);
          v105 = (unsigned int)v161;
          if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
          {
            v26 = v162;
            v139 = v102;
            sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 8u, v103, v104);
            v105 = (unsigned int)v161;
            v102 = v139;
          }
          v106 = (__int64)v160;
          v160[v105] = v102;
          v32 = v160;
          LODWORD(v161) = v161 + 1;
          ++v100;
          v155 |= v160[(unsigned int)v161 - 1] != (_QWORD)v101;
        }
        while ( v148 != v100 );
        if ( v155 )
        {
          v26 = &v160;
          v107 = sub_DCE040(*(__int64 **)a1, (__int64)&v160, v105, v106, v103);
          v32 = v160;
          v2 = v107;
        }
        goto LABEL_31;
      case 0xA:
        v160 = v162;
        v161 = 0x200000000LL;
        v47 = *(_QWORD **)(v2 + 32);
        v146 = &v47[*(_QWORD *)(v2 + 40)];
        if ( v47 == v146 )
          goto LABEL_15;
        v153 = 0;
        v48 = *(_QWORD **)(v2 + 32);
        do
        {
          v49 = (_QWORD *)*v48;
          v26 = (_QWORD *)*v48;
          v50 = sub_DE1A30(a1, *v48);
          v53 = (unsigned int)v161;
          if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
          {
            v26 = v162;
            v143 = v50;
            sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 8u, v51, v52);
            v53 = (unsigned int)v161;
            v50 = v143;
          }
          v54 = (__int64)v160;
          v160[v53] = v50;
          v32 = v160;
          LODWORD(v161) = v161 + 1;
          ++v48;
          v153 |= v160[(unsigned int)v161 - 1] != (_QWORD)v49;
        }
        while ( v146 != v48 );
        if ( v153 )
        {
          v26 = &v160;
          v55 = sub_DCDF90(*(__int64 **)a1, (__int64)&v160, v53, v54, v51);
          v32 = v160;
          v2 = v55;
        }
        goto LABEL_31;
      case 0xB:
        v33 = *(_QWORD *)(a1 + 88);
        v34 = *(unsigned int *)(v33 + 24);
        v35 = *(_QWORD *)(v33 + 8);
        if ( !(_DWORD)v34 )
          goto LABEL_36;
        v36 = (v34 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v18 = (__int64 *)(v35 + 16LL * v36);
        v37 = *v18;
        if ( v2 == *v18 )
          goto LABEL_35;
        v135 = 1;
        while ( 2 )
        {
          if ( v37 != -4096 )
          {
            v136 = v135 + 1;
            v36 = (v34 - 1) & (v135 + v36);
            v18 = (__int64 *)(v35 + 16LL * v36);
            v37 = *v18;
            if ( v2 != *v18 )
            {
              v135 = v136;
              continue;
            }
LABEL_35:
            if ( v18 != (__int64 *)(v35 + 16 * v34) )
              goto LABEL_14;
          }
          break;
        }
LABEL_36:
        v160 = v162;
        v161 = 0x200000000LL;
        v38 = *(_QWORD **)(v2 + 32);
        v145 = &v38[*(_QWORD *)(v2 + 40)];
        if ( v38 == v145 )
          goto LABEL_15;
        v152 = 0;
        v39 = *(_QWORD **)(v2 + 32);
        do
        {
          v40 = *v39;
          v26 = (_QWORD *)sub_DE1A30(a1, *v39);
          sub_D9B3A0((__int64)&v160, (__int64)v26, v41, v42, v43, v44);
          v32 = v160;
          ++v39;
          v152 |= v160[(unsigned int)v161 - 1] != v40;
        }
        while ( v145 != v39 );
        v45 = 0;
        if ( v152 )
        {
LABEL_40:
          v26 = &v160;
          v46 = sub_DCEE50(*(__int64 **)a1, (__int64)&v160, v45, v31, v28);
          v32 = v160;
          v2 = (__int64)v46;
        }
        goto LABEL_31;
      case 0xC:
        v14 = *(_QWORD *)(a1 + 88);
        v15 = *(unsigned int *)(v14 + 24);
        v16 = *(_QWORD *)(v14 + 8);
        if ( !(_DWORD)v15 )
          goto LABEL_106;
        v17 = (v15 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v2 == *v18 )
          goto LABEL_13;
        v129 = 1;
        while ( 2 )
        {
          if ( v19 != -4096 )
          {
            v130 = v129 + 1;
            v17 = (v15 - 1) & (v129 + v17);
            v18 = (__int64 *)(v16 + 16LL * v17);
            v19 = *v18;
            if ( v2 != *v18 )
            {
              v129 = v130;
              continue;
            }
LABEL_13:
            if ( v18 != (__int64 *)(v16 + 16 * v15) )
              goto LABEL_14;
          }
          break;
        }
LABEL_106:
        v160 = v162;
        v161 = 0x200000000LL;
        v118 = *(_QWORD **)(v2 + 32);
        v157 = &v118[*(_QWORD *)(v2 + 40)];
        if ( v118 != v157 )
        {
          v150 = 0;
          v119 = *(_QWORD **)(v2 + 32);
          do
          {
            v120 = *v119;
            v26 = (_QWORD *)sub_DE1A30(a1, *v119);
            sub_D9B3A0((__int64)&v160, (__int64)v26, v121, v122, v123, v124);
            v32 = v160;
            ++v119;
            v150 |= v160[(unsigned int)v161 - 1] != v120;
          }
          while ( v157 != v119 );
          if ( v150 )
          {
            v26 = &v160;
            v128 = sub_DCE150(*(__int64 **)a1, (__int64)&v160, v125, v126, v127);
            v32 = v160;
            v2 = v128;
          }
LABEL_31:
          if ( v32 != v162 )
            _libc_free(v32, v26);
        }
        goto LABEL_15;
      case 0xD:
        v160 = v162;
        v161 = 0x200000000LL;
        v23 = *(_QWORD **)(v2 + 32);
        v144 = &v23[*(_QWORD *)(v2 + 40)];
        if ( v23 == v144 )
          goto LABEL_15;
        v151 = 0;
        v24 = *(_QWORD **)(v2 + 32);
        do
        {
          v25 = (_QWORD *)*v24;
          v26 = (_QWORD *)*v24;
          v27 = sub_DE1A30(a1, *v24);
          v30 = (unsigned int)v161;
          if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
          {
            v26 = v162;
            v141 = v27;
            sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 8u, v28, v29);
            v30 = (unsigned int)v161;
            v27 = v141;
          }
          v31 = (__int64)v160;
          v160[v30] = v27;
          v32 = v160;
          LODWORD(v161) = v161 + 1;
          ++v24;
          v151 |= v160[(unsigned int)v161 - 1] != (_QWORD)v25;
        }
        while ( v144 != v24 );
        if ( !v151 )
          goto LABEL_31;
        v45 = 1;
        goto LABEL_40;
      case 0xE:
        v22 = sub_DE1A30(a1, *(_QWORD *)(v2 + 32));
        if ( v22 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DD3A70(*(_QWORD *)a1, v22, *(_QWORD *)(v2 + 40));
        goto LABEL_15;
      case 0xF:
        v92 = *(_QWORD *)(a1 + 88);
        v93 = *(unsigned int *)(v92 + 24);
        v94 = *(_QWORD *)(v92 + 8);
        if ( !(_DWORD)v93 )
          goto LABEL_15;
        v95 = (v93 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v18 = (__int64 *)(v94 + 16LL * v95);
        v96 = *v18;
        if ( *v18 == v2 )
          goto LABEL_80;
        v131 = 1;
        while ( 2 )
        {
          if ( v96 != -4096 )
          {
            v132 = v131 + 1;
            v95 = (v93 - 1) & (v131 + v95);
            v18 = (__int64 *)(v94 + 16LL * v95);
            v96 = *v18;
            if ( v2 != *v18 )
            {
              v131 = v132;
              continue;
            }
LABEL_80:
            if ( v18 != (__int64 *)(v94 + 16 * v93) )
LABEL_14:
              v2 = v18[1];
          }
          break;
        }
        goto LABEL_15;
      default:
        BUG();
    }
    do
    {
      v79 = sub_D95540(v76);
      if ( (unsigned int)sub_BCB060(v79) >= v78 )
        break;
      v80 = (_QWORD *)sub_B2BE50(**(_QWORD **)a1);
      v81 = sub_BCCE00(v80, v78);
      v82 = sub_DC2B70(*(_QWORD *)a1, v76, v81, 0);
      v83 = *(_QWORD *)(a1 + 88);
      v84 = *(_QWORD *)(v83 + 8);
      v85 = *(unsigned int *)(v83 + 24);
      if ( (_DWORD)v85 )
      {
        v86 = (v85 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
        v87 = (_QWORD *)(v84 + 16LL * v86);
        v88 = (_QWORD *)*v87;
        if ( v82 == (_QWORD *)*v87 )
        {
LABEL_66:
          if ( v87 != (_QWORD *)(v84 + 16 * v85) )
          {
            v2 = (__int64)sub_DC2B70(*(_QWORD *)a1, v87[1], v75, 0);
            goto LABEL_15;
          }
        }
        else
        {
          v89 = 1;
          while ( v88 != (_QWORD *)-4096LL )
          {
            v90 = v89 + 1;
            v86 = (v85 - 1) & (v89 + v86);
            v87 = (_QWORD *)(v84 + 16LL * v86);
            v88 = (_QWORD *)*v87;
            if ( v82 == (_QWORD *)*v87 )
              goto LABEL_66;
            v89 = v90;
          }
        }
      }
      v78 >>= 1;
      if ( (v78 & 7) != 0 )
        break;
    }
    while ( v78 > 7 );
    v76 = *(_QWORD *)(v2 + 32);
LABEL_104:
    v117 = sub_DE1A30(a1, v76);
    if ( v117 != *(_QWORD *)(v2 + 32) )
      v2 = (__int64)sub_DC2B70(*(_QWORD *)a1, v117, *(_QWORD *)(v2 + 40), 0);
LABEL_15:
    v159 = v2;
    sub_DB11F0((__int64)&v160, a1 + 8, &v158, &v159);
    v9 = (__int64 *)v162[0];
  }
  return v9[1];
}
