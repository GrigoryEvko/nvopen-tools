// Function: sub_2745AB0
// Address: 0x2745ab0
//
void __fastcall sub_2745AB0(_QWORD *a1, const void *a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  size_t v7; // r12
  __int64 v8; // r14
  int v9; // eax
  unsigned int v10; // edx
  __int64 v11; // r12
  __int64 v12; // r9
  unsigned int v13; // edi
  __int64 v14; // rcx
  _BYTE *v15; // r11
  __int64 v16; // r8
  __int64 v17; // rcx
  _BYTE *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r9
  int v21; // edi
  __int64 v22; // r15
  int v23; // edi
  unsigned int v24; // eax
  _BYTE *v25; // rcx
  int v26; // r10d
  _QWORD *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  char *v33; // rcx
  char *v34; // rdx
  __int64 *v35; // r13
  unsigned __int64 v36; // rbx
  __int64 v37; // r14
  unsigned __int64 v38; // rax
  _QWORD *v39; // r13
  __int64 v40; // r14
  int v41; // eax
  _QWORD *v42; // rcx
  int v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 *v46; // r15
  __int64 v47; // rdx
  _QWORD *v48; // r15
  unsigned __int8 *v49; // r15
  const char *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rsi
  unsigned __int8 *v54; // rsi
  __int64 v55; // r12
  __int64 v56; // rax
  __int64 v57; // r15
  __int64 v58; // rbx
  unsigned int v59; // esi
  int v60; // eax
  _QWORD *v61; // rcx
  int v62; // edx
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int64 *v65; // r15
  __int64 v66; // rdx
  _QWORD *v67; // r15
  __int64 v68; // rax
  __int64 v69; // r15
  unsigned int v70; // esi
  __int64 v71; // r9
  unsigned int v72; // edi
  _QWORD *v73; // rdx
  __int64 v74; // r8
  int v75; // ecx
  int v76; // r15d
  __int64 *v77; // r15
  __int64 v78; // rbx
  __int64 *i; // r12
  __int64 v80; // rdx
  __int64 *v81; // r13
  _BYTE *v82; // rdi
  __int64 v83; // r9
  unsigned int v84; // edi
  _QWORD *v85; // rdx
  __int64 v86; // r8
  int v87; // r11d
  int v88; // eax
  int v89; // eax
  int v90; // eax
  __int64 v91; // r8
  int v92; // r11d
  _QWORD *v93; // r10
  unsigned int v94; // edx
  __int64 v95; // rdi
  int v96; // r11d
  int v97; // eax
  int v98; // eax
  int v99; // eax
  __int64 v100; // r8
  int v101; // r11d
  _QWORD *v102; // r10
  unsigned int v103; // edx
  __int64 v104; // rdi
  int v105; // eax
  __int64 v106; // r8
  int v107; // r11d
  unsigned int v108; // edx
  __int64 v109; // rdi
  int v110; // eax
  __int64 v111; // r8
  int v112; // r11d
  unsigned int v113; // edx
  __int64 v114; // rdi
  __int64 *v115; // r12
  __int64 v116; // [rsp+18h] [rbp-148h]
  _QWORD *v117; // [rsp+18h] [rbp-148h]
  _QWORD *v118; // [rsp+18h] [rbp-148h]
  _QWORD *v119; // [rsp+18h] [rbp-148h]
  __int64 *v120; // [rsp+28h] [rbp-138h]
  _QWORD *v122; // [rsp+38h] [rbp-128h]
  _QWORD *v123; // [rsp+38h] [rbp-128h]
  _QWORD *v124; // [rsp+38h] [rbp-128h]
  bool v125; // [rsp+38h] [rbp-128h]
  _QWORD *v126; // [rsp+58h] [rbp-108h]
  __int64 *v127; // [rsp+58h] [rbp-108h]
  __int64 *v128; // [rsp+58h] [rbp-108h]
  __int64 v129; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v130; // [rsp+70h] [rbp-F0h]
  __int64 v131; // [rsp+78h] [rbp-E8h]
  __int64 v132; // [rsp+80h] [rbp-E0h]
  char *v133; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v134; // [rsp+98h] [rbp-C8h] BYREF
  __int64 v135; // [rsp+A0h] [rbp-C0h]
  __int64 v136; // [rsp+A8h] [rbp-B8h]
  __int64 v137; // [rsp+B0h] [rbp-B0h]
  _BYTE *v138; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v139; // [rsp+C8h] [rbp-98h]
  _BYTE v140[32]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v141; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v142; // [rsp+F8h] [rbp-68h]
  _BYTE v143[96]; // [rsp+100h] [rbp-60h] BYREF

  v7 = 8 * a3;
  v8 = (8 * a3) >> 3;
  v138 = v140;
  v139 = 0x400000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x20 )
  {
    sub_C8D5F0((__int64)&v138, v140, (8 * a3) >> 3, 8u, a5, a6);
    v82 = &v138[8 * (unsigned int)v139];
  }
  else
  {
    if ( !v7 )
      goto LABEL_3;
    v82 = v140;
  }
  memcpy(v82, a2, v7);
  LODWORD(v7) = v139;
LABEL_3:
  v9 = v8 + v7;
  v10 = v8 + v7;
  v141 = (__int64 *)v143;
  v142 = 0x600000000LL;
  LODWORD(v139) = v8 + v7;
  v11 = *a1 + 1232LL;
  if ( !a4 )
    v11 = *a1 + 600LL;
  if ( !v9 )
  {
    v115 = (__int64 *)v143;
    goto LABEL_91;
  }
  do
  {
    while ( 1 )
    {
      v16 = a1[1];
      v17 = v10--;
      v18 = *(_BYTE **)&v138[8 * v17 - 8];
      v19 = *(unsigned int *)(v16 + 24);
      LODWORD(v139) = v10;
      if ( (_DWORD)v19 )
      {
        v12 = *(_QWORD *)(v16 + 8);
        v13 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v14 = v12 + ((unsigned __int64)v13 << 6);
        v15 = *(_BYTE **)(v14 + 24);
        if ( v18 == v15 )
        {
LABEL_8:
          if ( v14 != v12 + (v19 << 6) )
            goto LABEL_9;
        }
        else
        {
          v75 = 1;
          while ( v15 != (_BYTE *)-4096LL )
          {
            v76 = v75 + 1;
            v13 = (v19 - 1) & (v75 + v13);
            v14 = v12 + ((unsigned __int64)v13 << 6);
            v15 = *(_BYTE **)(v14 + 24);
            if ( v18 == v15 )
              goto LABEL_8;
            v75 = v76;
          }
        }
      }
      v20 = *(_QWORD *)(v11 + 8);
      v21 = *(_DWORD *)(v11 + 24);
      if ( *v18 > 0x1Cu )
        break;
      if ( v21 )
      {
        v22 = 0;
        goto LABEL_13;
      }
LABEL_9:
      if ( !v10 )
        goto LABEL_27;
    }
    v22 = (__int64)v18;
    if ( !v21 )
      goto LABEL_17;
LABEL_13:
    v23 = v21 - 1;
    v24 = v23 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v25 = *(_BYTE **)(v20 + 16LL * v24);
    if ( v18 == v25 )
      goto LABEL_9;
    v26 = 1;
    while ( v25 != (_BYTE *)-4096LL )
    {
      v24 = v23 & (v26 + v24);
      v25 = *(_BYTE **)(v20 + 16LL * v24);
      if ( v18 == v25 )
        goto LABEL_9;
      ++v26;
    }
    if ( !v22 )
      goto LABEL_9;
LABEL_17:
    v27 = sub_2745840(v16, (__int64)v18);
    v30 = v27[2];
    if ( v30 )
    {
      if ( v30 != -4096 && v30 != -8192 )
      {
        v126 = v27;
        sub_BD60C0(v27);
        v27 = v126;
      }
      v27[2] = 0;
    }
    v31 = (unsigned int)v142;
    v32 = (unsigned int)v142 + 1LL;
    if ( v32 > HIDWORD(v142) )
    {
      sub_C8D5F0((__int64)&v141, v143, v32, 8u, v28, v29);
      v31 = (unsigned int)v142;
    }
    v141[v31] = v22;
    LODWORD(v142) = v142 + 1;
    if ( (*(_BYTE *)(v22 + 7) & 0x40) != 0 )
    {
      v34 = *(char **)(v22 - 8);
      v33 = &v34[32 * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v33 = (char *)v22;
      v34 = (char *)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF));
    }
    sub_2739020((__int64)&v138, &v138[8 * (unsigned int)v139], v34, v33);
    v10 = v139;
  }
  while ( (_DWORD)v139 );
LABEL_27:
  v35 = v141;
  v36 = (unsigned int)v142;
  v115 = &v141[v36];
  if ( v141 != &v141[v36] )
  {
    v37 = a1[2];
    _BitScanReverse64(&v38, (__int64)(v36 * 8) >> 3);
    sub_2738DE0(v141, &v141[v36], 2LL * (int)(63 - (v38 ^ 0x3F)), v37);
    if ( v36 > 16 )
    {
      v77 = v35 + 16;
      sub_2738120(v35, v35 + 16, v37);
      if ( v115 != v35 + 16 )
      {
        v128 = v115;
        do
        {
          v78 = *v77;
          for ( i = v77; ; i[1] = *i )
          {
            v80 = *(i - 1);
            v81 = i--;
            if ( !(unsigned __int8)sub_B19DB0(v37, v78, v80) )
              break;
          }
          *v81 = v78;
          ++v77;
        }
        while ( v128 != v77 );
      }
    }
    else
    {
      sub_2738120(v35, v115, v37);
    }
    v115 = v141;
    v120 = &v141[(unsigned int)v142];
    if ( v141 != v120 )
    {
      v127 = v141;
      v39 = a1;
      v40 = v116;
      while ( 1 )
      {
        v55 = *v127;
        v56 = sub_B47F80((_BYTE *)*v127);
        v131 = v55;
        v57 = v39[1];
        v129 = 2;
        v58 = v56;
        v130 = 0;
        v125 = v55 != -8192 && v55 != -4096 && v55 != 0;
        if ( v125 )
          sub_BD73F0((__int64)&v129);
        v132 = v57;
        v59 = *(_DWORD *)(v57 + 24);
        if ( !v59 )
          break;
        v64 = v131;
        v83 = *(_QWORD *)(v57 + 8);
        v84 = (v59 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
        v85 = (_QWORD *)(v83 + ((unsigned __int64)v84 << 6));
        v86 = v85[3];
        if ( v131 == v86 )
        {
LABEL_111:
          v67 = v85 + 5;
          goto LABEL_73;
        }
        v87 = 1;
        v61 = 0;
        while ( v86 != -4096 )
        {
          if ( !v61 && v86 == -8192 )
            v61 = v85;
          v84 = (v59 - 1) & (v87 + v84);
          v85 = (_QWORD *)(v83 + ((unsigned __int64)v84 << 6));
          v86 = v85[3];
          if ( v131 == v86 )
            goto LABEL_111;
          ++v87;
        }
        v88 = *(_DWORD *)(v57 + 16);
        if ( !v61 )
          v61 = v85;
        ++*(_QWORD *)v57;
        v62 = v88 + 1;
        if ( 4 * (v88 + 1) >= 3 * v59 )
          goto LABEL_59;
        if ( v59 - *(_DWORD *)(v57 + 20) - v62 <= v59 >> 3 )
        {
          sub_CF32C0(v57, v59);
          v89 = *(_DWORD *)(v57 + 24);
          v61 = 0;
          if ( v89 )
          {
            v90 = v89 - 1;
            v91 = *(_QWORD *)(v57 + 8);
            v92 = 1;
            v93 = 0;
            v94 = v90 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
            v61 = (_QWORD *)(v91 + ((unsigned __int64)v94 << 6));
            v95 = v61[3];
            if ( v131 != v95 )
            {
              while ( v95 != -4096 )
              {
                if ( v95 == -8192 && !v93 )
                  v93 = v61;
                v94 = v90 & (v92 + v94);
                v61 = (_QWORD *)(v91 + ((unsigned __int64)v94 << 6));
                v95 = v61[3];
                if ( v131 == v95 )
                  goto LABEL_60;
                ++v92;
              }
LABEL_150:
              if ( v93 )
                v61 = v93;
            }
          }
LABEL_60:
          v62 = *(_DWORD *)(v57 + 16) + 1;
        }
        *(_DWORD *)(v57 + 16) = v62;
        v134 = 2;
        v135 = 0;
        v136 = -4096;
        v137 = 0;
        if ( v61[3] != -4096 )
        {
          --*(_DWORD *)(v57 + 20);
          v133 = (char *)&unk_49DB368;
          if ( v136 != 0 && v136 != -4096 && v136 != -8192 )
          {
            v117 = v61;
            sub_BD60C0(&v134);
            v61 = v117;
          }
        }
        v63 = v61[3];
        v64 = v131;
        if ( v63 != v131 )
        {
          v65 = v61 + 1;
          if ( v63 != -4096 && v63 != 0 && v63 != -8192 )
          {
            v118 = v61;
            sub_BD60C0(v61 + 1);
            v64 = v131;
            v61 = v118;
          }
          v61[3] = v64;
          if ( v64 == 0 || v64 == -4096 || v64 == -8192 )
          {
            v64 = v131;
          }
          else
          {
            v119 = v61;
            sub_BD6050(v65, v129 & 0xFFFFFFFFFFFFFFF8LL);
            v64 = v131;
            v61 = v119;
          }
        }
        v66 = v132;
        v61[5] = 6;
        v67 = v61 + 5;
        v61[6] = 0;
        v61[4] = v66;
        v61[7] = 0;
LABEL_73:
        if ( v64 != 0 && v64 != -4096 && v64 != -8192 )
          sub_BD60C0(&v129);
        v68 = v67[2];
        if ( v58 != v68 )
        {
          if ( v68 != -4096 && v68 != 0 && v68 != -8192 )
            sub_BD60C0(v67);
          v67[2] = v58;
          if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
            sub_BD73F0((__int64)v67);
        }
        v131 = v55;
        v129 = 2;
        v69 = v39[1];
        v130 = 0;
        if ( v125 )
          sub_BD73F0((__int64)&v129);
        v132 = v69;
        v70 = *(_DWORD *)(v69 + 24);
        if ( !v70 )
        {
          ++*(_QWORD *)v69;
          goto LABEL_33;
        }
        v45 = v131;
        v71 = *(_QWORD *)(v69 + 8);
        v72 = (v70 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
        v73 = (_QWORD *)(v71 + ((unsigned __int64)v72 << 6));
        v74 = v73[3];
        if ( v131 != v74 )
        {
          v96 = 1;
          v42 = 0;
          while ( v74 != -4096 )
          {
            if ( !v42 && v74 == -8192 )
              v42 = v73;
            v72 = (v70 - 1) & (v96 + v72);
            v73 = (_QWORD *)(v71 + ((unsigned __int64)v72 << 6));
            v74 = v73[3];
            if ( v131 == v74 )
              goto LABEL_87;
            ++v96;
          }
          v97 = *(_DWORD *)(v69 + 16);
          if ( !v42 )
            v42 = v73;
          ++*(_QWORD *)v69;
          v43 = v97 + 1;
          if ( 4 * (v97 + 1) >= 3 * v70 )
          {
LABEL_33:
            sub_CF32C0(v69, 2 * v70);
            v41 = *(_DWORD *)(v69 + 24);
            v42 = 0;
            if ( v41 )
            {
              v105 = v41 - 1;
              v106 = *(_QWORD *)(v69 + 8);
              v107 = 1;
              v102 = 0;
              v108 = v105 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
              v42 = (_QWORD *)(v106 + ((unsigned __int64)v108 << 6));
              v109 = v42[3];
              if ( v131 != v109 )
              {
                while ( v109 != -4096 )
                {
                  if ( v109 == -8192 && !v102 )
                    v102 = v42;
                  v108 = v105 & (v107 + v108);
                  v42 = (_QWORD *)(v106 + ((unsigned __int64)v108 << 6));
                  v109 = v42[3];
                  if ( v131 == v109 )
                    goto LABEL_34;
                  ++v107;
                }
LABEL_145:
                if ( v102 )
                  v42 = v102;
              }
            }
LABEL_34:
            v43 = *(_DWORD *)(v69 + 16) + 1;
          }
          else if ( v70 - *(_DWORD *)(v69 + 20) - v43 <= v70 >> 3 )
          {
            sub_CF32C0(v69, v70);
            v98 = *(_DWORD *)(v69 + 24);
            v42 = 0;
            if ( v98 )
            {
              v99 = v98 - 1;
              v100 = *(_QWORD *)(v69 + 8);
              v101 = 1;
              v102 = 0;
              v103 = v99 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
              v42 = (_QWORD *)(v100 + ((unsigned __int64)v103 << 6));
              v104 = v42[3];
              if ( v104 != v131 )
              {
                while ( v104 != -4096 )
                {
                  if ( !v102 && v104 == -8192 )
                    v102 = v42;
                  v103 = v99 & (v101 + v103);
                  v42 = (_QWORD *)(v100 + ((unsigned __int64)v103 << 6));
                  v104 = v42[3];
                  if ( v131 == v104 )
                    goto LABEL_34;
                  ++v101;
                }
                goto LABEL_145;
              }
            }
            goto LABEL_34;
          }
          *(_DWORD *)(v69 + 16) = v43;
          v134 = 2;
          v135 = 0;
          v136 = -4096;
          v137 = 0;
          if ( v42[3] != -4096 )
          {
            --*(_DWORD *)(v69 + 20);
            v133 = (char *)&unk_49DB368;
            if ( v136 != 0 && v136 != -4096 && v136 != -8192 )
            {
              v122 = v42;
              sub_BD60C0(&v134);
              v42 = v122;
            }
          }
          v44 = v42[3];
          v45 = v131;
          if ( v44 != v131 )
          {
            v46 = v42 + 1;
            if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
            {
              v123 = v42;
              sub_BD60C0(v42 + 1);
              v45 = v131;
              v42 = v123;
            }
            v42[3] = v45;
            if ( v45 == 0 || v45 == -4096 || v45 == -8192 )
            {
              v45 = v131;
            }
            else
            {
              v124 = v42;
              sub_BD6050(v46, v129 & 0xFFFFFFFFFFFFFFF8LL);
              v45 = v131;
              v42 = v124;
            }
          }
          v47 = v132;
          v42[5] = 6;
          v48 = v42 + 5;
          v42[6] = 0;
          v42[4] = v47;
          v42[7] = 0;
          goto LABEL_47;
        }
LABEL_87:
        v48 = v73 + 5;
LABEL_47:
        if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
          sub_BD60C0(&v129);
        v49 = (unsigned __int8 *)v48[2];
        v50 = sub_BD5D20(v55);
        LOWORD(v137) = 261;
        v133 = (char *)v50;
        v134 = v51;
        sub_BD6B50(v49, (const char **)&v133);
        v52 = v39[3];
        LOWORD(v40) = *(_WORD *)(v52 + 64);
        sub_B44220((_QWORD *)v58, *(_QWORD *)(v52 + 56), v40);
        sub_B9ADA0(v58, 0, 0);
        v133 = 0;
        if ( (char **)(v58 + 48) != &v133 )
        {
          v53 = *(_QWORD *)(v58 + 48);
          if ( v53 )
          {
            sub_B91220(v58 + 48, v53);
            v54 = (unsigned __int8 *)v133;
            *(_QWORD *)(v58 + 48) = v133;
            if ( v54 )
              sub_B976B0((__int64)&v133, v54, v58 + 48);
          }
        }
        if ( v120 == ++v127 )
        {
          v115 = v141;
          goto LABEL_91;
        }
      }
      ++*(_QWORD *)v57;
LABEL_59:
      sub_CF32C0(v57, 2 * v59);
      v60 = *(_DWORD *)(v57 + 24);
      v61 = 0;
      if ( v60 )
      {
        v110 = v60 - 1;
        v111 = *(_QWORD *)(v57 + 8);
        v112 = 1;
        v93 = 0;
        v113 = v110 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
        v61 = (_QWORD *)(v111 + ((unsigned __int64)v113 << 6));
        v114 = v61[3];
        if ( v131 != v114 )
        {
          while ( v114 != -4096 )
          {
            if ( v114 == -8192 && !v93 )
              v93 = v61;
            v113 = v110 & (v112 + v113);
            v61 = (_QWORD *)(v111 + ((unsigned __int64)v113 << 6));
            v114 = v61[3];
            if ( v131 == v114 )
              goto LABEL_60;
            ++v112;
          }
          goto LABEL_150;
        }
      }
      goto LABEL_60;
    }
  }
LABEL_91:
  if ( v115 != (__int64 *)v143 )
    _libc_free((unsigned __int64)v115);
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
}
