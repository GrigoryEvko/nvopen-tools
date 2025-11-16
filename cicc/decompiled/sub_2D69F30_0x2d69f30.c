// Function: sub_2D69F30
// Address: 0x2d69f30
//
__int64 __fastcall sub_2D69F30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // r8
  const void *v9; // r14
  int v10; // eax
  size_t v11; // r15
  __int64 v12; // r13
  unsigned int i; // eax
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // r9
  const void *v22; // r8
  size_t v23; // r10
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 *v29; // r9
  __int64 v30; // rcx
  unsigned __int64 v31; // rsi
  int v32; // eax
  unsigned __int64 *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r14
  __int64 v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // r12
  __int64 v40; // rcx
  int v41; // eax
  int v42; // eax
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rbx
  _QWORD *v58; // r12
  __int64 v59; // rax
  __int64 v61; // rbx
  __int64 v62; // r12
  char *v63; // rax
  char *v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r15
  __int64 v67; // rcx
  __int64 v68; // rax
  unsigned __int64 v69; // rdi
  int v70; // edx
  __int64 v71; // rax
  __int64 v72; // r13
  unsigned __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  unsigned __int64 v77; // rdi
  __int64 *v78; // r12
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rax
  unsigned int v82; // esi
  __int64 v83; // rax
  __int64 v84; // r15
  __int64 v85; // rdx
  __int64 v86; // rbx
  __int64 v87; // rcx
  __int64 v88; // r12
  __int64 v89; // rdx
  int v90; // eax
  unsigned int v91; // esi
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rsi
  __int64 v95; // r13
  int v96; // eax
  __int64 v97; // r13
  int v98; // r14d
  unsigned int v99; // ebx
  unsigned int v100; // r15d
  unsigned int v101; // r14d
  int v102; // eax
  __int64 v103; // r12
  __int64 v104; // rbx
  __int64 v105; // rdx
  __int64 v106; // rsi
  __int64 v107; // rdx
  __int64 *v108; // rax
  char *v109; // r14
  __int64 v110; // rax
  __int64 j; // rbx
  __int64 *v112; // r15
  __int64 v113; // rax
  __int64 v114; // r9
  __int64 v115; // rdx
  __int64 v116; // r8
  __int64 v117; // rsi
  __int64 v118; // rdi
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  unsigned __int64 v122; // r14
  __int64 *k; // rax
  unsigned __int64 v124; // rax
  unsigned __int64 v125; // rdx
  __int64 v126; // r8
  __int64 v127; // r9
  unsigned __int8 v128; // bl
  __int64 v129; // r15
  __int64 v130; // rax
  bool v131; // cf
  unsigned __int64 v132; // rax
  __int64 v133; // [rsp-8h] [rbp-468h]
  __int64 v135; // [rsp+28h] [rbp-438h]
  _BYTE *v136; // [rsp+38h] [rbp-428h]
  unsigned __int8 v137; // [rsp+48h] [rbp-418h]
  __int64 v138; // [rsp+48h] [rbp-418h]
  __int64 v139; // [rsp+50h] [rbp-410h]
  size_t v140; // [rsp+58h] [rbp-408h]
  __int64 v141; // [rsp+58h] [rbp-408h]
  __int64 v142; // [rsp+58h] [rbp-408h]
  const void *v143; // [rsp+60h] [rbp-400h]
  __int64 v144; // [rsp+60h] [rbp-400h]
  int v145; // [rsp+60h] [rbp-400h]
  __int64 v146; // [rsp+60h] [rbp-400h]
  __int64 v147; // [rsp+68h] [rbp-3F8h]
  __int64 *v148; // [rsp+68h] [rbp-3F8h]
  __int64 v149; // [rsp+68h] [rbp-3F8h]
  __int64 v150; // [rsp+78h] [rbp-3E8h] BYREF
  _QWORD v151[4]; // [rsp+80h] [rbp-3E0h] BYREF
  unsigned __int64 v152; // [rsp+A0h] [rbp-3C0h] BYREF
  unsigned __int64 v153; // [rsp+A8h] [rbp-3B8h]
  __int64 *v154; // [rsp+B0h] [rbp-3B0h]
  __int64 v155; // [rsp+B8h] [rbp-3A8h]
  _BYTE *v156; // [rsp+D0h] [rbp-390h] BYREF
  __int64 v157; // [rsp+D8h] [rbp-388h]
  _BYTE dest[128]; // [rsp+E0h] [rbp-380h] BYREF
  __int64 v159; // [rsp+160h] [rbp-300h] BYREF
  char *v160; // [rsp+168h] [rbp-2F8h]
  __int64 v161; // [rsp+170h] [rbp-2F0h]
  int v162; // [rsp+178h] [rbp-2E8h]
  char v163; // [rsp+17Ch] [rbp-2E4h]
  char v164; // [rsp+180h] [rbp-2E0h] BYREF
  __int64 v165; // [rsp+200h] [rbp-260h] BYREF
  char *v166; // [rsp+208h] [rbp-258h]
  __int64 v167; // [rsp+210h] [rbp-250h]
  int v168; // [rsp+218h] [rbp-248h]
  char v169; // [rsp+21Ch] [rbp-244h]
  char v170; // [rsp+220h] [rbp-240h] BYREF
  __int64 v171; // [rsp+2A0h] [rbp-1C0h] BYREF
  __int64 v172; // [rsp+2A8h] [rbp-1B8h]
  _BYTE v173[432]; // [rsp+2B0h] [rbp-1B0h] BYREF

  v160 = &v164;
  v6 = *(_QWORD *)(a1 + 56);
  v7 = dest;
  v8 = *(_QWORD *)(v6 + 40);
  v9 = *(const void **)(v6 + 32);
  v159 = 0;
  v157 = 0x1000000000LL;
  v10 = 0;
  v163 = 1;
  v11 = v8 - (_QWORD)v9;
  v156 = dest;
  v161 = 16;
  v12 = (v8 - (__int64)v9) >> 3;
  v162 = 0;
  if ( (unsigned __int64)(v8 - (_QWORD)v9) > 0x80 )
  {
    v149 = v8;
    sub_C8D5F0((__int64)&v156, dest, (v8 - (__int64)v9) >> 3, 8u, v8, a6);
    v10 = v157;
    v8 = v149;
    v7 = &v156[8 * (unsigned int)v157];
  }
  if ( v9 != (const void *)v8 )
  {
    memmove(v7, v9, v11);
    v10 = v157;
  }
  LODWORD(v157) = v10 + v12;
  for ( i = v10 + v12; i; i = v157 )
  {
    v18 = (unsigned __int64)v156;
    v19 = i;
    v14 = i - 1;
    v20 = *(_QWORD *)&v156[8 * v19 - 8];
    LODWORD(v157) = v14;
    v21 = *(_QWORD *)(v20 + 16);
    v22 = *(const void **)(v20 + 8);
    v23 = v21 - (_QWORD)v22;
    v24 = (v21 - (__int64)v22) >> 3;
    if ( v14 + v24 > (unsigned __int64)HIDWORD(v157) )
    {
      v140 = *(_QWORD *)(v20 + 16) - (_QWORD)v22;
      v143 = *(const void **)(v20 + 8);
      v147 = *(_QWORD *)(v20 + 16);
      sub_C8D5F0((__int64)&v156, dest, v14 + v24, 8u, (__int64)v22, v21);
      v18 = (unsigned __int64)v156;
      v14 = (unsigned int)v157;
      v23 = v140;
      v22 = v143;
      v21 = v147;
    }
    if ( (const void *)v21 != v22 )
    {
      memmove((void *)(v18 + 8 * v14), v22, v23);
      LODWORD(v14) = v157;
    }
    LODWORD(v157) = v24 + v14;
    v15 = (__int64 *)sub_D4B130(v20);
    if ( v15 )
      sub_D695C0((__int64)&v171, (__int64)&v159, v15, v16, v8, v17);
  }
  v25 = a2 + 72;
  v137 = 0;
  v171 = (__int64)v173;
  v172 = 0x1000000000LL;
  v26 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != *(_QWORD *)(v26 + 8) )
  {
    v27 = *(_QWORD *)(v26 + 8);
    do
    {
      v35 = v27 - 24;
      if ( !v27 )
        v35 = 0;
      v36 = v35;
      if ( !byte_5016A08 )
        v137 |= sub_F39260(v35, *(_QWORD *)(a1 + 48), 0);
      v166 = 0;
      v165 = 6;
      v167 = v36;
      if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
        sub_BD73F0((__int64)&v165);
      v28 = (unsigned int)v172;
      v29 = &v165;
      v30 = v171;
      v31 = (unsigned int)v172 + 1LL;
      v32 = v172;
      if ( v31 > HIDWORD(v172) )
      {
        if ( v171 > (unsigned __int64)&v165
          || (unsigned __int64)&v165 >= v171 + 24 * (unsigned __int64)(unsigned int)v172 )
        {
          sub_F39130((__int64)&v171, v31, (unsigned int)v172, v171, v8, (__int64)&v165);
          v28 = (unsigned int)v172;
          v30 = v171;
          v29 = &v165;
          v32 = v172;
        }
        else
        {
          v109 = (char *)&v165 - v171;
          sub_F39130((__int64)&v171, v31, (unsigned int)v172, v171, v8, (__int64)&v165);
          v30 = v171;
          v28 = (unsigned int)v172;
          v29 = (__int64 *)&v109[v171];
          v32 = v172;
        }
      }
      v33 = (unsigned __int64 *)(v30 + 24 * v28);
      if ( v33 )
      {
        *v33 = 6;
        v34 = v29[2];
        v33[1] = 0;
        v33[2] = v34;
        if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
          sub_BD6050(v33, *v29 & 0xFFFFFFFFFFFFFFF8LL);
        v32 = v172;
      }
      LODWORD(v172) = v32 + 1;
      if ( v167 != -4096 && v167 != 0 && v167 != -8192 )
        sub_BD60C0(&v165);
      v27 = *(_QWORD *)(v27 + 8);
    }
    while ( v27 != v25 );
    v139 = v171;
    v136 = (_BYTE *)(v171 + 24LL * (unsigned int)v172);
    if ( (_BYTE *)v171 != v136 )
    {
      while ( 1 )
      {
        v61 = *(_QWORD *)(v139 + 16);
        v148 = (__int64 *)v61;
        if ( !v61 )
          goto LABEL_64;
        v62 = sub_2D64A00(v61);
        if ( !v62 )
          goto LABEL_64;
        if ( v163 )
        {
          v63 = v160;
          v64 = &v160[8 * HIDWORD(v161)];
          if ( v160 != v64 )
          {
            while ( v61 != *(_QWORD *)v63 )
            {
              v63 += 8;
              if ( v64 == v63 )
                goto LABEL_89;
            }
            if ( !byte_5017B88 )
            {
LABEL_87:
              if ( !sub_AA54C0(v61) )
                goto LABEL_64;
              v65 = sub_AA54C0(v61);
              if ( !sub_AA56F0(v65) )
                goto LABEL_64;
            }
          }
        }
        else
        {
          v108 = sub_C8CA60((__int64)&v159, v61);
          if ( byte_5017B88 != 1 && v108 )
            goto LABEL_87;
        }
LABEL_89:
        v66 = *(_QWORD *)(v61 + 16);
        if ( v66 )
        {
          while ( 1 )
          {
            v67 = *(_QWORD *)(v66 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v67 - 30) <= 0xAu )
              break;
            v66 = *(_QWORD *)(v66 + 8);
            if ( !v66 )
              goto LABEL_98;
          }
LABEL_93:
          v68 = *(_QWORD *)(v67 + 40);
          v69 = *(_QWORD *)(v68 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v69 == v68 + 48 )
            goto LABEL_137;
          if ( !v69 )
            goto LABEL_216;
          v70 = *(unsigned __int8 *)(v69 - 24);
          if ( (unsigned int)(v70 - 30) > 0xA )
LABEL_137:
            BUG();
          if ( (_BYTE)v70 != 40 )
            goto LABEL_97;
          v97 = v69 - 24;
          v145 = sub_B46E30(v69 - 24);
          v98 = v145 >> 2;
          if ( v145 >> 2 > 0 )
          {
            v142 = v66;
            v99 = 0;
            while ( v62 != sub_B46EC0(v97, v99) )
            {
              v100 = v99 + 1;
              if ( v62 == sub_B46EC0(v97, v99 + 1)
                || (v100 = v99 + 2, v62 == sub_B46EC0(v97, v99 + 2))
                || (v100 = v99 + 3, v62 == sub_B46EC0(v97, v99 + 3)) )
              {
                v101 = v100;
                v66 = v142;
                v99 = v101;
                goto LABEL_135;
              }
              v99 += 4;
              if ( !--v98 )
              {
                v66 = v142;
                v102 = v145 - v99;
                goto LABEL_146;
              }
            }
            v66 = v142;
            goto LABEL_135;
          }
          v102 = v145;
          v99 = 0;
LABEL_146:
          if ( v102 != 2 )
          {
            if ( v102 != 3 )
            {
              if ( v102 != 1 || v62 != sub_B46EC0(v97, v99) )
                goto LABEL_97;
LABEL_135:
              if ( v99 == v145 )
                goto LABEL_97;
              goto LABEL_64;
            }
            if ( v62 == sub_B46EC0(v97, v99) )
              goto LABEL_135;
            ++v99;
          }
          if ( v62 == sub_B46EC0(v97, v99) )
            goto LABEL_135;
          if ( v62 == sub_B46EC0(v97, ++v99) )
            goto LABEL_135;
LABEL_97:
          while ( 1 )
          {
            v66 = *(_QWORD *)(v66 + 8);
            if ( !v66 )
              break;
            v67 = *(_QWORD *)(v66 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v67 - 30) <= 0xAu )
              goto LABEL_93;
          }
        }
LABEL_98:
        v71 = sub_AA5510((__int64)v148);
        v72 = v71;
        if ( !v71 || (unsigned __int8)(*(_BYTE *)sub_986580(v71) - 32) > 1u )
          goto LABEL_103;
        v73 = sub_986580((__int64)v148);
        v74 = sub_AA5030((__int64)v148, 1);
        v76 = v74 - 24;
        if ( v74 )
          v74 -= 24;
        if ( v73 != v74 )
          goto LABEL_103;
        v110 = *(_QWORD *)(v62 + 56);
        if ( !v110 )
          goto LABEL_216;
        if ( *(_BYTE *)(v110 - 24) != 84 )
          goto LABEL_103;
        v165 = 0;
        v166 = &v170;
        v167 = 16;
        v168 = 0;
        v169 = 1;
        j = *(_QWORD *)(v62 + 16);
        if ( j )
        {
          while ( 1 )
          {
            v76 = *(_QWORD *)(j + 24);
            if ( (unsigned __int8)(*(_BYTE *)v76 - 30) <= 0xAu )
              break;
            j = *(_QWORD *)(j + 8);
            if ( !j )
              goto LABEL_191;
          }
LABEL_172:
          v112 = *(__int64 **)(v76 + 40);
          if ( v148 == v112 )
            goto LABEL_188;
          v113 = sub_AA5930(v62);
          v116 = v115;
          v117 = v113;
          if ( v113 == v115 )
            goto LABEL_187;
          while ( 1 )
          {
            v118 = *(_QWORD *)(v117 - 8);
            v76 = *(_DWORD *)(v117 + 4) & 0x7FFFFFF;
            if ( (*(_DWORD *)(v117 + 4) & 0x7FFFFFF) != 0 )
            {
              v119 = 0;
              v75 = v118 + 32LL * *(unsigned int *)(v117 + 72);
              do
              {
                if ( v148 == *(__int64 **)(v75 + 8 * v119) )
                {
                  v114 = *(_QWORD *)(v118 + 32 * v119);
                  goto LABEL_179;
                }
                ++v119;
              }
              while ( (_DWORD)v76 != (_DWORD)v119 );
              v114 = *(_QWORD *)(v118 + 0x1FFFFFFFE0LL);
LABEL_179:
              v120 = 0;
              do
              {
                if ( v112 == *(__int64 **)(v75 + 8 * v120) )
                {
                  if ( *(_QWORD *)(v118 + 32 * v120) != v114 )
                    goto LABEL_201;
                  goto LABEL_183;
                }
                ++v120;
              }
              while ( (_DWORD)v76 != (_DWORD)v120 );
              if ( *(_QWORD *)(v118 + 0x1FFFFFFFE0LL) != v114 )
                break;
            }
LABEL_183:
            v121 = *(_QWORD *)(v117 + 32);
            if ( !v121 )
              goto LABEL_216;
            v117 = 0;
            if ( *(_BYTE *)(v121 - 24) == 84 )
              v117 = v121 - 24;
            if ( v116 == v117 )
            {
LABEL_187:
              sub_D695C0((__int64)&v152, (__int64)&v165, v112, v75, v116, v114);
              goto LABEL_188;
            }
          }
LABEL_201:
          if ( v116 == v117 )
            goto LABEL_187;
LABEL_188:
          for ( j = *(_QWORD *)(j + 8); j; j = *(_QWORD *)(j + 8) )
          {
            v76 = *(_QWORD *)(j + 24);
            if ( (unsigned __int8)(*(_BYTE *)v76 - 30) <= 0xAu )
              goto LABEL_172;
          }
        }
LABEL_191:
        if ( (unsigned __int8)sub_B19060((__int64)&v165, v72, v76, v75) )
          goto LABEL_195;
        v122 = sub_FDD860(*(__int64 **)(a1 + 64), v72);
        v150 = sub_FDD860(*(__int64 **)(a1 + 64), (__int64)v148);
        v151[1] = sub_254BB00((__int64)&v165);
        v151[0] = v166;
        sub_254BBF0((__int64)v151);
        v151[2] = &v165;
        v151[3] = v165;
        v152 = sub_254BB00((__int64)&v165);
        v153 = v152;
        sub_254BBF0((__int64)&v152);
        v154 = &v165;
        v155 = v165;
        for ( k = (__int64 *)v151[0]; v151[0] != v152; k = (__int64 *)v151[0] )
        {
          v129 = *k;
          if ( v72 == sub_AA5510(*k) && v62 == sub_2D64A00(v129) )
          {
            v130 = sub_FDD860(*(__int64 **)(a1 + 64), v129);
            v131 = __CFADD__(v150, v130);
            v132 = v150 + v130;
            if ( v131 )
              v150 = -1;
            else
              v150 = v132;
          }
          v151[0] += 8LL;
          sub_254BBF0((__int64)v151);
        }
        v124 = sub_1098D90((unsigned __int64 *)&v150, qword_5017808);
        v153 = v125;
        v152 = v124;
        if ( !(_BYTE)v125 || v152 >= v122 )
        {
LABEL_195:
          if ( !v169 )
            _libc_free((unsigned __int64)v166);
LABEL_103:
          v77 = v148[6] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (__int64 *)v77 == v148 + 6 )
            goto LABEL_217;
          if ( !v77 )
LABEL_216:
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v77 - 24) - 30 > 0xA )
LABEL_217:
            BUG();
          v135 = *(_QWORD *)(v77 - 56);
          v78 = (__int64 *)sub_AA54C0(v135);
          v137 = v78 != 0 && v135 != (_QWORD)v78;
          if ( v137 )
          {
            sub_F39690(v135, 0, 0, 0, 0, 0, 0);
            v128 = *(_BYTE *)(a1 + 832);
            if ( v128 )
            {
              sub_D695C0((__int64)&v165, a1 + 840, v78, v133, v126, v127);
              sub_25DDDB0(a1 + 840, v135);
              v137 = v128;
            }
          }
          else
          {
            v79 = sub_AA5930(v135);
            v138 = v80;
            v48 = v79;
            while ( v138 != v48 )
            {
              if ( (*(_DWORD *)(v48 + 4) & 0x7FFFFFF) != 0 )
              {
                v81 = 0;
                while ( 1 )
                {
                  v82 = v81;
                  if ( v148 == *(__int64 **)(*(_QWORD *)(v48 - 8) + 32LL * *(unsigned int *)(v48 + 72) + 8 * v81) )
                    break;
                  if ( (*(_DWORD *)(v48 + 4) & 0x7FFFFFF) == (_DWORD)++v81 )
                    goto LABEL_141;
                }
              }
              else
              {
LABEL_141:
                v82 = -1;
              }
              v83 = sub_B48BF0(v48, v82, 0);
              v84 = v83;
              if ( *(_BYTE *)v83 == 84 && v148 == *(__int64 **)(v83 + 40) )
              {
                if ( (*(_DWORD *)(v83 + 4) & 0x7FFFFFF) != 0 )
                {
                  v103 = 0;
                  v104 = 8LL * (*(_DWORD *)(v83 + 4) & 0x7FFFFFF);
                  do
                  {
                    v105 = *(_QWORD *)(v84 - 8);
                    v106 = *(_QWORD *)(v105 + 4 * v103);
                    v107 = *(_QWORD *)(v105 + 32LL * *(unsigned int *)(v84 + 72) + v103);
                    v103 += 8;
                    sub_F0A850(v48, v106, v107);
                  }
                  while ( v104 != v103 );
                }
              }
              else
              {
                v85 = v148[7];
                if ( !v85 )
                  goto LABEL_216;
                if ( *(_BYTE *)(v85 - 24) == 84 )
                {
                  if ( (*(_DWORD *)(v85 - 20) & 0x7FFFFFF) != 0 )
                  {
                    v86 = 0;
                    v87 = v48;
                    v88 = v148[7];
                    v89 = 8LL * (*(_DWORD *)(v85 - 20) & 0x7FFFFFF);
                    do
                    {
                      v95 = *(_QWORD *)(*(_QWORD *)(v88 - 32) + 32LL * *(unsigned int *)(v88 + 48) + v86);
                      v96 = *(_DWORD *)(v87 + 4) & 0x7FFFFFF;
                      if ( v96 == *(_DWORD *)(v87 + 72) )
                      {
                        v141 = v89;
                        v144 = v87;
                        sub_B48D90(v87);
                        v87 = v144;
                        v89 = v141;
                        v96 = *(_DWORD *)(v144 + 4) & 0x7FFFFFF;
                      }
                      v90 = (v96 + 1) & 0x7FFFFFF;
                      v91 = v90 | *(_DWORD *)(v87 + 4) & 0xF8000000;
                      v92 = *(_QWORD *)(v87 - 8) + 32LL * (unsigned int)(v90 - 1);
                      *(_DWORD *)(v87 + 4) = v91;
                      if ( *(_QWORD *)v92 )
                      {
                        v93 = *(_QWORD *)(v92 + 8);
                        **(_QWORD **)(v92 + 16) = v93;
                        if ( v93 )
                          *(_QWORD *)(v93 + 16) = *(_QWORD *)(v92 + 16);
                      }
                      *(_QWORD *)v92 = v84;
                      v94 = *(_QWORD *)(v84 + 16);
                      *(_QWORD *)(v92 + 8) = v94;
                      if ( v94 )
                        *(_QWORD *)(v94 + 16) = v92 + 8;
                      *(_QWORD *)(v92 + 16) = v84 + 16;
                      v86 += 8;
                      *(_QWORD *)(v84 + 16) = v92;
                      *(_QWORD *)(*(_QWORD *)(v87 - 8)
                                + 32LL * *(unsigned int *)(v87 + 72)
                                + 8LL * ((*(_DWORD *)(v87 + 4) & 0x7FFFFFFu) - 1)) = v95;
                    }
                    while ( v89 != v86 );
                    v48 = v87;
                  }
                }
                else
                {
                  v37 = v148[2];
                  if ( v37 )
                  {
                    while ( 1 )
                    {
                      v38 = *(_QWORD *)(v37 + 24);
                      if ( (unsigned __int8)(*(_BYTE *)v38 - 30) <= 0xAu )
                        break;
                      v37 = *(_QWORD *)(v37 + 8);
                      if ( !v37 )
                        goto LABEL_46;
                    }
                    v39 = v83 + 16;
LABEL_37:
                    v40 = *(_QWORD *)(v38 + 40);
                    v41 = *(_DWORD *)(v48 + 4) & 0x7FFFFFF;
                    if ( v41 == *(_DWORD *)(v48 + 72) )
                    {
                      v146 = *(_QWORD *)(v38 + 40);
                      sub_B48D90(v48);
                      v40 = v146;
                      v41 = *(_DWORD *)(v48 + 4) & 0x7FFFFFF;
                    }
                    v42 = (v41 + 1) & 0x7FFFFFF;
                    v43 = v42 | *(_DWORD *)(v48 + 4) & 0xF8000000;
                    v44 = *(_QWORD *)(v48 - 8) + 32LL * (unsigned int)(v42 - 1);
                    *(_DWORD *)(v48 + 4) = v43;
                    if ( *(_QWORD *)v44 )
                    {
                      v45 = *(_QWORD *)(v44 + 8);
                      **(_QWORD **)(v44 + 16) = v45;
                      if ( v45 )
                        *(_QWORD *)(v45 + 16) = *(_QWORD *)(v44 + 16);
                    }
                    *(_QWORD *)v44 = v84;
                    v46 = *(_QWORD *)(v84 + 16);
                    *(_QWORD *)(v44 + 8) = v46;
                    if ( v46 )
                      *(_QWORD *)(v46 + 16) = v44 + 8;
                    *(_QWORD *)(v44 + 16) = v39;
                    *(_QWORD *)(v84 + 16) = v44;
                    *(_QWORD *)(*(_QWORD *)(v48 - 8)
                              + 32LL * *(unsigned int *)(v48 + 72)
                              + 8LL * ((*(_DWORD *)(v48 + 4) & 0x7FFFFFFu) - 1)) = v40;
                    while ( 1 )
                    {
                      v37 = *(_QWORD *)(v37 + 8);
                      if ( !v37 )
                        break;
                      v38 = *(_QWORD *)(v37 + 24);
                      if ( (unsigned __int8)(*(_BYTE *)v38 - 30) <= 0xAu )
                        goto LABEL_37;
                    }
                  }
                }
              }
LABEL_46:
              v47 = *(_QWORD *)(v48 + 32);
              if ( !v47 )
                goto LABEL_216;
              v48 = 0;
              if ( *(_BYTE *)(v47 - 24) == 84 )
                v48 = v47 - 24;
            }
            if ( (*(_BYTE *)(v77 - 17) & 0x20) != 0 )
            {
              v49 = v77 - 24;
              if ( sub_B91C10(v77 - 24, 18) )
              {
                v50 = v148[2];
                if ( v50 )
                {
                  while ( 1 )
                  {
                    v51 = *(_QWORD *)(v50 + 24);
                    if ( (unsigned __int8)(*(_BYTE *)v51 - 30) <= 0xAu )
                      break;
                    v50 = *(_QWORD *)(v50 + 8);
                    if ( !v50 )
                      goto LABEL_63;
                  }
LABEL_56:
                  v52 = *(_QWORD *)(v51 + 40);
                  v53 = *(_QWORD *)(v52 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v53 == v52 + 48 )
                  {
                    v56 = 0;
                  }
                  else
                  {
                    if ( !v53 )
                      goto LABEL_216;
                    v54 = *(unsigned __int8 *)(v53 - 24);
                    v55 = v53 - 24;
                    if ( (unsigned int)(v54 - 30) >= 0xB )
                      v55 = 0;
                    v56 = v55;
                  }
                  LODWORD(v165) = 18;
                  sub_B47C00(v56, v49, (int *)&v165, 1);
                  while ( 1 )
                  {
                    v50 = *(_QWORD *)(v50 + 8);
                    if ( !v50 )
                      break;
                    v51 = *(_QWORD *)(v50 + 24);
                    if ( (unsigned __int8)(*(_BYTE *)v51 - 30) <= 0xAu )
                      goto LABEL_56;
                  }
                }
              }
            }
LABEL_63:
            sub_BD84D0((__int64)v148, v135);
            sub_AA5450(v148);
            v137 = 1;
          }
          goto LABEL_64;
        }
        if ( !v169 )
          _libc_free((unsigned __int64)v166);
LABEL_64:
        v139 += 24;
        if ( (_BYTE *)v139 == v136 )
        {
          v57 = v171;
          v58 = (_QWORD *)(v171 + 24LL * (unsigned int)v172);
          v136 = v58;
          if ( (_QWORD *)v171 != v58 )
          {
            do
            {
              v59 = *(v58 - 1);
              v58 -= 3;
              if ( v59 != 0 && v59 != -4096 && v59 != -8192 )
                sub_BD60C0(v58);
            }
            while ( (_QWORD *)v57 != v58 );
            v136 = (_BYTE *)v171;
          }
          break;
        }
      }
    }
    if ( v136 != v173 )
      _libc_free((unsigned __int64)v136);
  }
  if ( v156 != dest )
    _libc_free((unsigned __int64)v156);
  if ( !v163 )
    _libc_free((unsigned __int64)v160);
  return v137;
}
