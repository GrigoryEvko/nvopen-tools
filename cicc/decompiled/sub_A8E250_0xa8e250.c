// Function: sub_A8E250
// Address: 0xa8e250
//
__int64 __fastcall sub_A8E250(__int64 a1, _QWORD *a2, char a3)
{
  void *v6; // rax
  size_t v7; // rdx
  char *v8; // r13
  unsigned int v9; // r12d
  unsigned __int64 v11; // r14
  char *v12; // r8
  int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // r13
  void *v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  unsigned int v21; // r12d
  void *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  size_t v28; // rdx
  unsigned int v29; // r12d
  unsigned int v30; // r13d
  __int64 v31; // rax
  size_t v32; // rdx
  _QWORD *v33; // r8
  int v34; // edx
  size_t v35; // r12
  _WORD *v36; // rdi
  _WORD *v37; // r14
  size_t v38; // r12
  char *v39; // rdx
  __int64 v40; // rcx
  const void *v41; // r14
  unsigned int v42; // r12d
  void *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rdi
  size_t v47; // r14
  char *v48; // r8
  bool v49; // al
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  void *v53; // r13
  size_t v54; // r14
  void *v55; // rax
  size_t v56; // rdx
  __int64 v57; // r14
  __int64 v58; // rax
  __int64 v59; // rsi
  void *v60; // rdx
  _BYTE *v61; // r9
  size_t v62; // r8
  _BYTE *v63; // rax
  _QWORD *v64; // rdx
  size_t v65; // r9
  __int64 v66; // rax
  unsigned int v67; // r15d
  __int64 v68; // rax
  __int64 v69; // r13
  void *v70; // rax
  __int64 v71; // rdx
  _QWORD *v72; // rdi
  _WORD *v73; // r13
  const void *v74; // rdi
  __int64 v75; // rdi
  unsigned int v76; // r12d
  __int64 v77; // rsi
  size_t v78; // r13
  void *v79; // r12
  const void *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rsi
  __int64 v83; // rsi
  size_t v84; // rdx
  const void *v85; // rdi
  __int64 v86; // rdi
  size_t v87; // rdx
  char *v88; // rsi
  int v89; // eax
  unsigned __int64 v90; // rax
  unsigned int v91; // r13d
  __int64 v92; // rax
  __int64 v93; // rcx
  __int64 v94; // rdi
  void *v95; // rdx
  __int64 v96; // rax
  _QWORD *v97; // rsi
  __int64 v98; // r13
  __int64 v99; // rax
  __int64 v100; // r13
  unsigned __int64 v101; // rdx
  void *v102; // rdi
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rax
  void *v106; // rax
  __int64 v107; // rax
  _QWORD *v108; // rdi
  __int64 v109; // rdi
  __int64 v110; // rax
  __int64 v111; // r13
  __int64 v112; // rax
  __int64 v113; // rdx
  void *v114; // rax
  void *v115; // r12
  size_t v116; // r13
  unsigned int v117; // eax
  __int64 v118; // rdi
  const void *v119; // rdi
  __int64 v120; // rdi
  char *v121; // rdx
  size_t v122; // rcx
  __int64 v123; // r13
  __int64 v124; // rdi
  void *v125; // r13
  size_t v126; // r14
  void *v127; // r14
  size_t v128; // r13
  bool v129; // al
  bool v130; // al
  char v131; // al
  size_t v132; // [rsp+8h] [rbp-E8h]
  size_t v133; // [rsp+8h] [rbp-E8h]
  size_t v134; // [rsp+8h] [rbp-E8h]
  unsigned int src; // [rsp+10h] [rbp-E0h]
  _BYTE *srca; // [rsp+10h] [rbp-E0h]
  void *v137; // [rsp+18h] [rbp-D8h]
  void *s1; // [rsp+20h] [rbp-D0h] BYREF
  size_t v139; // [rsp+28h] [rbp-C8h]
  _QWORD *v140; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE *v141; // [rsp+38h] [rbp-B8h]
  _QWORD v142[2]; // [rsp+40h] [rbp-B0h] BYREF
  void *v143; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v144; // [rsp+58h] [rbp-98h]
  int v145; // [rsp+60h] [rbp-90h]
  int v146; // [rsp+64h] [rbp-8Ch]
  __int16 v147; // [rsp+70h] [rbp-80h]
  void *s2; // [rsp+80h] [rbp-70h] BYREF
  size_t n; // [rsp+88h] [rbp-68h]
  _QWORD v150[12]; // [rsp+90h] [rbp-60h] BYREF

  v6 = (void *)sub_BD5D20(a1);
  v139 = v7;
  s1 = v6;
  if ( v7 <= 4 )
    return 0;
  v8 = (char *)s1;
  if ( *(_DWORD *)s1 != 1836477548 )
    return 0;
  if ( *((_BYTE *)s1 + 4) != 46 )
    return 0;
  v11 = v7 - 5;
  v12 = (char *)s1 + 5;
  v13 = 0;
  s1 = (char *)s1 + 5;
  v139 = v7 - 5;
  if ( v7 == 5 )
    return 0;
  switch ( v8[5] )
  {
    case 'a':
      if ( v11 <= 3 )
        goto LABEL_9;
      if ( *(_DWORD *)(v8 + 5) == 778924641 )
      {
        v13 = 1;
        s1 = v8 + 9;
        v139 = v7 - 9;
        goto LABEL_177;
      }
      if ( v11 <= 7 )
      {
        if ( v7 != 12 )
          goto LABEL_9;
        goto LABEL_162;
      }
      if ( *(_QWORD *)(v8 + 5) == 0x2E34366863726161LL )
      {
        s1 = v8 + 13;
        v139 = v7 - 13;
LABEL_177:
        if ( !(unsigned __int8)sub_A7D7F0(v13, a1, (__int64)s1, v139, a2) )
          goto LABEL_9;
        return 1;
      }
LABEL_162:
      if ( *(_DWORD *)(v8 + 5) != 1734634849 || *(_WORD *)(v8 + 9) != 28259 || v8[11] != 46 )
        goto LABEL_9;
      s1 = v8 + 12;
      v139 = v7 - 12;
      if ( v7 == 20 && *(_QWORD *)(v8 + 12) == 0x7469626E67696C61LL )
      {
        v109 = *(_QWORD *)(a1 + 40);
        v9 = 1;
        s2 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
        *a2 = sub_B6E160(v109, 181, &s2, 1);
        return v9;
      }
      v9 = sub_95CB50((const void **)&s1, "atomic.", 7u);
      if ( (_BYTE)v9 )
      {
        v53 = s1;
        v54 = v139;
        if ( !sub_A7BBF0(s1, v139, "inc", 3u) && !sub_A7BBF0(v53, v54, "dec", 3u) )
          goto LABEL_9;
        goto LABEL_170;
      }
      if ( (unsigned __int8)sub_95CB50((const void **)&s1, "ds.", 3u)
        || (unsigned __int8)sub_95CB50((const void **)&s1, "global.atomic.", 0xEu)
        || (unsigned __int8)sub_95CB50((const void **)&s1, "flat.atomic.", 0xCu) )
      {
        v115 = s1;
        v116 = v139;
        if ( sub_A7BBF0(s1, v139, "fadd", 4u)
          || sub_A7BBF0(v115, v116, "fmin", 4u) && !sub_A7BBF0(v115, v116, "fmin.num", 8u)
          || sub_A7BBF0(v115, v116, "fmax", 4u) && !sub_A7BBF0(v115, v116, "fmax.num", 8u) )
        {
          goto LABEL_39;
        }
      }
      else
      {
        v116 = v139;
        v115 = s1;
      }
      LOBYTE(v117) = sub_A7BBF0(v115, v116, "ldexp.", 6u);
      v9 = v117;
      if ( !(_BYTE)v117 )
        goto LABEL_9;
      s2 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        sub_B2C6D0(a1);
      v118 = *(_QWORD *)(a1 + 40);
      n = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 48LL);
      *a2 = sub_B6E160(v118, 209, &s2, 2);
      return v9;
    case 'c':
      v50 = *(_QWORD *)(a1 + 104);
      if ( v50 != 1 )
      {
        if ( v50 == 2 && v7 == 13 )
        {
          if ( *(_QWORD *)(v8 + 5) != 0x646E652E6F726F63LL )
            goto LABEL_9;
          v9 = 1;
          sub_A7BA00(a1);
          *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 43, 0, 0);
          return v9;
        }
        goto LABEL_143;
      }
      if ( v11 <= 4 )
        goto LABEL_143;
      v21 = 65;
      if ( memcmp(v8 + 5, "ctlz.", 5u) )
      {
        if ( memcmp(v8 + 5, "cttz.", 5u) )
        {
          v12 = v8 + 5;
LABEL_143:
          if ( v11 != 16 || *(_QWORD *)(v8 + 5) ^ 0x6E79732E61647563LL | *((_QWORD *)v12 + 1) ^ 0x7364616572687463LL )
            goto LABEL_9;
          goto LABEL_39;
        }
        v21 = 67;
      }
      sub_A7BA00(a1);
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        sub_B2C6D0(a1);
      v22 = *(void **)(*(_QWORD *)(a1 + 96) + 8LL);
      goto LABEL_26;
    case 'd':
      if ( v11 <= 3 || *(_DWORD *)(v8 + 5) != 778527332 )
        goto LABEL_9;
      v47 = v7 - 9;
      v48 = v8 + 9;
      s1 = v8 + 9;
      v139 = v7 - 9;
      if ( !a3 )
        goto LABEL_135;
      v9 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 872LL);
      if ( !(_BYTE)v9 )
        goto LABEL_135;
      if ( v7 == 13 )
      {
        if ( *(_DWORD *)(v8 + 9) == 1919181921 )
          goto LABEL_170;
LABEL_133:
        v132 = (size_t)v48;
        if ( !sub_9691B0(s1, v139, "declare", 7) )
        {
          v49 = sub_9691B0(s1, v139, "label", 5);
          v48 = (char *)v132;
          if ( !v49 )
          {
LABEL_135:
            if ( v47 == 4 )
            {
              if ( *(_DWORD *)(v8 + 9) == 1919181921 )
              {
LABEL_140:
                v9 = 1;
                sub_A7BA00(a1);
                *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 71, 0, 0);
                return v9;
              }
            }
            else if ( v47 == 5 && *(_DWORD *)(v8 + 9) == 1970037110 && v48[4] == 101 && *(_QWORD *)(a1 + 104) == 4 )
            {
              goto LABEL_140;
            }
LABEL_9:
            v14 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 16LL);
            v15 = *v14;
            if ( *(_BYTE *)(*v14 + 8) != 15
              || (v16 = (void *)(*(_DWORD *)(v15 + 8) >> 8), (*(_DWORD *)(v15 + 8) & 0x400) != 0)
              && (v16 = (void *)(BYTE1(*(_DWORD *)(v15 + 8)) & 2), (*(_DWORD *)(v15 + 8) & 0x200) == 0)
              || (v17 = *(unsigned int *)(a1 + 36), !(_DWORD)v17) )
            {
LABEL_174:
              v55 = (void *)sub_B6E300(a1);
              n = v56;
              v9 = (unsigned __int8)v56;
              s2 = v55;
              if ( (_BYTE)v56 )
                *a2 = s2;
              return v9;
            }
            if ( v139 > 3 )
            {
              v16 = s1;
              if ( *(_DWORD *)((char *)s1 + v139 - 4) == 1684827950 )
              {
                v19 = sub_BD5D20(a1);
                if ( v18 > 3 && *(_DWORD *)(v19 + v18 - 4) == 1684827950 )
                  v18 -= 4LL;
                *a2 = sub_BA8CB0(*(_QWORD *)(a1 + 40), v19, v18);
                return 1;
              }
            }
            s2 = v150;
            n = 0x400000000LL;
            sub_B6DAB0(v17, &s2, v16);
            if ( *(_DWORD *)s2 != 13 )
            {
              if ( s2 != v150 )
                _libc_free(s2, &s2);
              goto LABEL_174;
            }
            v57 = *(_QWORD *)(a1 + 24);
            v58 = sub_BD0B90(*(_QWORD *)v15, *(_QWORD *)(v15 + 16), *(unsigned int *)(v15 + 12), 0);
            v59 = *(_QWORD *)(v57 + 16) + 8LL;
            v137 = (void *)sub_BCF480(
                             v58,
                             v59,
                             (8LL * *(unsigned int *)(v57 + 12) - 8) >> 3,
                             *(_DWORD *)(v57 + 8) >> 8 != 0);
            v61 = (_BYTE *)sub_BD5D20(a1);
            v62 = (size_t)v60;
            if ( !v61 )
            {
              LOBYTE(v142[0]) = 0;
              v140 = v142;
              v141 = 0;
LABEL_188:
              sub_A7BA00(a1);
              v65 = *(_QWORD *)(a1 + 40);
              v143 = &v140;
              v147 = 260;
              v66 = *(_QWORD *)(a1 + 8);
              v133 = v65;
              v67 = *(_BYTE *)(a1 + 32) & 0xF;
              src = *(_DWORD *)(v66 + 8) >> 8;
              v68 = sub_BD2DA0(136);
              v69 = v68;
              if ( v68 )
              {
                v59 = (__int64)v137;
                sub_B2C3B0(v68, v137, v67, src, &v143, v133);
              }
              *a2 = v69;
              v70 = (void *)sub_B6E300(v69);
              v144 = v71;
              v143 = v70;
              if ( (_BYTE)v71 )
                *a2 = v143;
              if ( v140 != v142 )
              {
                v59 = v142[0] + 1LL;
                j_j___libc_free_0(v140, v142[0] + 1LL);
              }
              v72 = s2;
              if ( s2 != v150 )
LABEL_195:
                _libc_free(v72, v59);
              return 1;
            }
            v143 = v60;
            v63 = v60;
            v140 = v142;
            if ( (unsigned __int64)v60 > 0xF )
            {
              v134 = (size_t)v60;
              srca = v61;
              v107 = sub_22409D0(&v140, &v143, 0);
              v61 = srca;
              v62 = v134;
              v140 = (_QWORD *)v107;
              v108 = (_QWORD *)v107;
              v142[0] = v143;
            }
            else
            {
              if ( v60 == (void *)1 )
              {
                LOBYTE(v142[0]) = *v61;
                v64 = v142;
LABEL_187:
                v141 = v63;
                v63[(_QWORD)v64] = 0;
                goto LABEL_188;
              }
              if ( !v60 )
              {
                v64 = v142;
                goto LABEL_187;
              }
              v108 = v142;
            }
            v59 = (__int64)v61;
            memcpy(v108, v61, v62);
            v63 = v143;
            v64 = v140;
            goto LABEL_187;
          }
        }
LABEL_170:
        *a2 = 0;
        return v9;
      }
      if ( v7 == 14 )
      {
        v87 = 5;
        v88 = "value";
      }
      else
      {
        v87 = 6;
        v88 = "assign";
        if ( v47 != 6 )
          goto LABEL_133;
      }
      v89 = memcmp(v8 + 9, v88, v87);
      v48 = v8 + 9;
      if ( !v89 )
        goto LABEL_170;
      goto LABEL_133;
    case 'e':
      if ( v11 <= 0x13 )
        goto LABEL_9;
      if ( *(_QWORD *)(v8 + 5) ^ 0x656D697265707865LL | *(_QWORD *)(v8 + 13) ^ 0x6365762E6C61746ELL
        || *(_DWORD *)(v8 + 21) != 779251572 )
      {
        if ( v11 > 0x17
          && !(*(_QWORD *)(v8 + 5) ^ 0x656D697265707865LL | *(_QWORD *)(v8 + 13) ^ 0x6574732E6C61746ELL)
          && *(_QWORD *)(v8 + 21) == 0x2E726F7463657670LL )
        {
          v9 = 1;
          v139 = v7 - 29;
          s1 = v8 + 29;
          sub_A7BA00(a1);
          v46 = *(_QWORD *)(a1 + 40);
          s2 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
          *a2 = sub_B6E160(v46, 345, &s2, 1);
          return v9;
        }
        goto LABEL_9;
      }
      v90 = v7 - 25;
      s1 = v8 + 25;
      v139 = v7 - 25;
      if ( v7 - 25 <= 0x13 )
      {
        if ( v90 <= 7 )
        {
          if ( v7 != 32 )
          {
LABEL_303:
            if ( v90 <= 6 )
              goto LABEL_9;
LABEL_304:
            if ( *(_DWORD *)(v8 + 25) != 1969513842 || *(_WORD *)(v8 + 29) != 25955 || v8[31] != 46 )
              goto LABEL_9;
            s1 = v8 + 32;
            v139 = v7 - 32;
            s2 = v150;
            n = 0x200000000LL;
            if ( !byte_4F80BE0 && (unsigned int)sub_2207590(&byte_4F80BE0) )
            {
              sub_C88F40(&unk_4F80BF0, "^([a-z]+)\\.[a-z][0-9]+", 22, 0);
              __cxa_atexit(sub_C88FF0, &unk_4F80BF0, &qword_4A427C0);
              sub_2207640(&byte_4F80BE0);
            }
            if ( !(unsigned __int8)sub_C89090(&unk_4F80BF0, s1, v139, &s2, 0) )
              goto LABEL_561;
            v102 = (void *)*((_QWORD *)s2 + 2);
            v103 = *((_QWORD *)s2 + 3);
            v146 = 0;
            v143 = v102;
            v144 = v103;
            if ( v103 == 3 && !memcmp(v102, "add", 3u) )
            {
              v145 = 387;
              LOBYTE(v146) = 1;
            }
            LODWORD(v140) = 395;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "mul", 3);
            LODWORD(v140) = 388;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "and", 3);
            LODWORD(v140) = 396;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "or", 2);
            if ( !(_BYTE)v146 && v144 == 3 && !memcmp(v143, "xor", 3u) )
            {
              v145 = 401;
              LOBYTE(v146) = 1;
            }
            else if ( !(_BYTE)v146 && v144 == 4 && *(_DWORD *)v143 == 2019650931 )
            {
              v145 = 397;
              LOBYTE(v146) = 1;
            }
            LODWORD(v140) = 398;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "smin", 4);
            if ( !(_BYTE)v146 && v144 == 4 && *(_DWORD *)v143 == 2019650933 )
            {
              v145 = 399;
              LOBYTE(v146) = 1;
            }
            LODWORD(v140) = 400;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "umin", 4);
            LODWORD(v140) = 390;
            sub_A8A0E0((__int64)&v143, (int *)&v140, "fmax", 4);
            if ( !(_BYTE)v146 && v144 == 4 && *(_DWORD *)v143 == 1852403046 )
            {
              v145 = 392;
              LOBYTE(v146) = 1;
            }
            if ( (_BYTE)v146 && (v91 = v145) != 0 )
            {
              sub_A7BA00(a1);
              v104 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
              v105 = 0;
            }
            else
            {
LABEL_561:
              if ( !byte_4F80BC8 && (unsigned int)sub_2207590(&byte_4F80BC8) )
              {
                sub_C88F40(&unk_4F80BD0, "^v2\\.([a-z]+)\\.[fi][0-9]+", 25, 0);
                __cxa_atexit(sub_C88FF0, &unk_4F80BD0, &qword_4A427C0);
                sub_2207640(&byte_4F80BC8);
              }
              v97 = s1;
              LODWORD(n) = 0;
              if ( !(unsigned __int8)sub_C89090(&unk_4F80BD0, s1, v139, &s2, 0) )
                goto LABEL_311;
              v113 = *((_QWORD *)s2 + 3);
              v114 = (void *)*((_QWORD *)s2 + 2);
              v146 = 0;
              LODWORD(v140) = 389;
              v144 = v113;
              v143 = v114;
              sub_A8A0E0((__int64)&v143, (int *)&v140, "fadd", 4);
              v97 = &v140;
              LODWORD(v140) = 394;
              sub_A8A0E0((__int64)&v143, (int *)&v140, "fmul", 4);
              if ( !(_BYTE)v146 || (v91 = v145) == 0 )
              {
LABEL_311:
                if ( s2 != v150 )
                  _libc_free(s2, v97);
                goto LABEL_9;
              }
              sub_A7BA00(a1);
              v104 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
              v105 = 1;
            }
            v106 = *(void **)(v104 + 8 * v105 + 8);
            v94 = *(_QWORD *)(a1 + 40);
            v95 = &v143;
            v93 = 1;
            v143 = v106;
LABEL_299:
            v59 = v91;
            v96 = sub_B6E160(v94, v91, v95, v93);
            v72 = s2;
            *a2 = v96;
            if ( v72 != v150 )
              goto LABEL_195;
            return 1;
          }
LABEL_290:
          if ( *(_DWORD *)(v8 + 25) == 1702063721 && *(_WORD *)(v8 + 29) == 29810 && v8[31] == 46 )
          {
            v91 = 382;
            goto LABEL_295;
          }
          if ( *(_DWORD *)(v8 + 25) == 1768714355 && *(_WORD *)(v8 + 29) == 25955 && v8[31] == 46 )
          {
            v91 = 403;
            goto LABEL_295;
          }
          if ( v90 > 7 )
          {
            if ( *(_QWORD *)(v8 + 25) == 0x2E65737265766572LL )
            {
              v91 = 402;
LABEL_295:
              v92 = *(_QWORD *)(a1 + 24);
              s2 = v150;
              n = 0x200000000LL;
              goto LABEL_296;
            }
            if ( v90 > 0xB && *(_QWORD *)(v8 + 25) == 0x61656C7265746E69LL && *(_DWORD *)(v8 + 33) == 775054710 )
            {
              v110 = *(_QWORD *)(a1 + 24);
              s2 = v150;
              v91 = 383;
              n = 0x200000000LL;
              sub_94F8E0((__int64)&s2, **(_QWORD **)(v110 + 16));
LABEL_298:
              sub_A7BA00(a1);
              v93 = (unsigned int)n;
              v94 = *(_QWORD *)(a1 + 40);
              v95 = s2;
              goto LABEL_299;
            }
            if ( v90 > 0xD
              && *(_QWORD *)(v8 + 25) == 0x6C7265746E696564LL
              && *(_DWORD *)(v8 + 33) == 1702256997
              && *(_WORD *)(v8 + 37) == 11826 )
            {
              v92 = *(_QWORD *)(a1 + 24);
              v91 = 377;
              s2 = v150;
              n = 0x200000000LL;
LABEL_296:
              *(_QWORD *)s2 = *(_QWORD *)(*(_QWORD *)(v92 + 16) + 8LL);
              LODWORD(n) = n + 1;
              if ( v91 == 382 )
                sub_94F8E0((__int64)&s2, *(_QWORD *)(*(_QWORD *)(v92 + 16) + 16LL));
              goto LABEL_298;
            }
          }
          goto LABEL_303;
        }
      }
      else if ( !(*(_QWORD *)(v8 + 25) ^ 0x2E74636172747865LL | *(_QWORD *)(v8 + 33) ^ 0x7463612E7473616CLL)
             && *(_DWORD *)(v8 + 41) == 778401385 )
      {
        goto LABEL_304;
      }
      if ( *(_QWORD *)(v8 + 25) == 0x2E74636172747865LL )
      {
        v98 = *(_QWORD *)(a1 + 24);
        s2 = v150;
        n = 0x200000000LL;
        sub_94F8E0((__int64)&s2, **(_QWORD **)(v98 + 16));
        v99 = (unsigned int)n;
        v100 = *(_QWORD *)(*(_QWORD *)(v98 + 16) + 8LL);
        v101 = (unsigned int)n + 1LL;
        if ( v101 > HIDWORD(n) )
        {
          sub_C8D5F0(&s2, v150, v101, 8);
          v99 = (unsigned int)n;
        }
        *((_QWORD *)s2 + v99) = v100;
        v91 = 381;
        LODWORD(n) = n + 1;
        goto LABEL_298;
      }
      goto LABEL_290;
    case 'f':
      if ( v11 <= 9 || *(_QWORD *)(v8 + 5) != 0x6E756F722E746C66LL || *(_WORD *)(v8 + 13) != 29540 )
        goto LABEL_9;
      v9 = 1;
      sub_A7BA00(a1);
      *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 189, 0, 0);
      return v9;
    case 'i':
      if ( v11 <= 0x16
        || *(_QWORD *)(v8 + 5) ^ 0x6E61697261766E69LL | *(_QWORD *)(v8 + 13) ^ 0x2E70756F72672E74LL
        || *(_DWORD *)(v8 + 21) != 1920098658
        || *(_WORD *)(v8 + 25) != 25961
        || v8[27] != 114 )
      {
        goto LABEL_9;
      }
      v9 = 1;
      s2 = *(void **)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL);
      sub_A7BA00(a1);
      *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 208, &s2, 1);
      return v9;
    case 'm':
      if ( v11 <= 6 )
        goto LABEL_9;
      if ( *(_DWORD *)(v8 + 5) == 1668113773 && *(_WORD *)(v8 + 9) == 31088 && (v76 = 238, v8[11] == 46)
        || v7 != 12 && (v76 = 241, *(_QWORD *)(v8 + 5) == 0x2E65766F6D6D656DLL) )
      {
        if ( *(_QWORD *)(a1 + 104) == 5 )
        {
          sub_A7BA00(a1);
          v77 = v76;
          v9 = 1;
          *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), v77, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL, 3);
          return v9;
        }
      }
      if ( *(_DWORD *)(v8 + 5) == 1936549229
        && *(_WORD *)(v8 + 9) == 29797
        && v8[11] == 46
        && *(_QWORD *)(a1 + 104) == 5 )
      {
        v9 = 1;
        sub_A7BA00(a1);
        v51 = *(_QWORD *)(a1 + 40);
        v52 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
        s2 = *(void **)(v52 + 8);
        n = *(_QWORD *)(v52 + 24);
        *a2 = sub_B6E160(v51, 243, &s2, 2);
        return v9;
      }
      goto LABEL_9;
    case 'n':
      if ( v11 <= 4 || *(_DWORD *)(v8 + 5) != 1836480110 || v8[9] != 46 )
        goto LABEL_9;
      v31 = *(_QWORD *)(a1 + 104);
      v32 = v7 - 10;
      s1 = v8 + 10;
      v139 = v32;
      if ( v31 != 1 )
        goto LABEL_80;
      s2 = v8 + 10;
      n = v32;
      HIDWORD(v150[0]) = 0;
      LODWORD(v143) = 14;
      if ( v32 == 6 && !memcmp(v8 + 10, "brev32", 6u) )
      {
        LODWORD(v150[0]) = 14;
        BYTE4(v150[0]) = 1;
      }
      else
      {
        sub_A8A0E0((__int64)&s2, (int *)&v143, "brev64", 6);
      }
      if ( BYTE4(v150[0]) )
        goto LABEL_324;
      if ( n == 5 && !memcmp(s2, "clz.i", 5u) )
      {
        LODWORD(v150[0]) = 65;
        BYTE4(v150[0]) = 1;
      }
      else
      {
        if ( n != 6 || memcmp(s2, "popc.i", 6u) )
        {
          LODWORD(v143) = 1;
          if ( n == 6 && !memcmp(s2, "abs.i8", 6u) )
          {
            LODWORD(v150[0]) = 1;
            BYTE4(v150[0]) = 1;
            goto LABEL_244;
          }
LABEL_243:
          if ( !(unsigned __int8)sub_A8A0E0((__int64)&s2, (int *)&v143, "abs.i16", 7)
            && !(unsigned __int8)sub_A8A0E0((__int64)&s2, (int *)&v143, "abs.i32", 7) )
          {
            sub_A8A0E0((__int64)&s2, (int *)&v143, "abs.i64", 7);
          }
LABEL_244:
          v9 = BYTE4(v150[0]);
          if ( BYTE4(v150[0]) && LODWORD(v150[0]) )
          {
            v120 = *(_QWORD *)(a1 + 40);
            s2 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
            *a2 = sub_B6E160(v120, LODWORD(v150[0]), &s2, 1);
            return v9;
          }
          if ( v139 == 23 )
          {
            v82 = 8932;
            if ( memcmp(s1, "isspacep.cluster.shared", 0x17u) )
            {
LABEL_249:
              v31 = *(_QWORD *)(a1 + 104);
LABEL_80:
              if ( v31 != 2 )
                goto LABEL_81;
              if ( v139 > 7 && *(_QWORD *)s1 == 0x2E7461732E646461LL )
              {
                s1 = (char *)s1 + 8;
                v139 -= 8LL;
                v83 = 311;
                if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "s.i", 3u) )
                {
                  if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "u.i", 3u) )
                    goto LABEL_255;
                  v83 = 359;
                }
              }
              else if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "sub.sat.", 8u)
                     || (v83 = 338, !(unsigned __int8)sub_95CB50((const void **)&s1, "s.i", 3u))
                     && (v83 = 371, !(unsigned __int8)sub_95CB50((const void **)&s1, "u.i", 3u)) )
              {
LABEL_255:
                if ( v139 == 16 && !memcmp(s1, "cluster.set.rank", 0x10u) )
                {
                  v9 = 1;
                  *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 9005, 0, 0);
                  return v9;
                }
LABEL_81:
                v33 = *(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL);
                v34 = *(unsigned __int8 *)(*v33 + 8LL);
                if ( (unsigned int)(v34 - 17) <= 1 )
                  LOBYTE(v34) = *(_BYTE *)(**(_QWORD **)(*v33 + 16LL) + 8LL);
                if ( (_BYTE)v34 != 1 && (unsigned int)sub_A7CD60((__int64)s1, v139) )
                  goto LABEL_39;
                v35 = v139;
                v36 = s1;
                if ( v139 > 6
                  && (*(_DWORD *)s1 == 779054182 && *((_WORD *)s1 + 2) == 11891 && *((_BYTE *)s1 + 6) == 105
                   || *(_DWORD *)s1 == 779054182 && *((_WORD *)s1 + 2) == 11893 && *((_BYTE *)s1 + 6) == 105) )
                {
                  s1 = (char *)s1 + 7;
                  v139 -= 7LL;
                  if ( !(unsigned __int8)sub_BCAC40(*v33, 32) )
                    goto LABEL_39;
                  v36 = s1;
                  v35 = v139;
                }
                if ( v35 <= 3 )
                {
                  if ( v35 == 3 && !memcmp(v36, "h2f", 3u) )
                    goto LABEL_39;
                }
                else
                {
                  if ( *(_DWORD *)v36 == 779313761 )
                  {
                    v38 = v35 - 4;
                    v37 = v36 + 2;
                    s1 = v36 + 2;
                    v139 = v38;
                    if ( v38 == 1 )
                    {
                      if ( *((_BYTE *)v36 + 4) != 105 )
                        goto LABEL_203;
                    }
                    else
                    {
                      if ( v38 != 2 )
                        goto LABEL_100;
                      if ( v36[2] != 27756 )
                        goto LABEL_203;
                    }
                    goto LABEL_39;
                  }
                  if ( v35 == 6 )
                  {
                    if ( *(_DWORD *)v36 == 779775075 && v36[2] == 27756 )
                      goto LABEL_39;
                  }
                  else if ( v35 == 7 && !memcmp(v36, "popc.ll", 7u) )
                  {
                    goto LABEL_39;
                  }
                }
                if ( (unsigned __int8)sub_95CB50((const void **)&s1, "max.", 4u)
                  || (unsigned __int8)sub_95CB50((const void **)&s1, "min.", 4u) )
                {
                  v37 = s1;
                  v38 = v139;
                  if ( sub_9691B0(s1, v139, "s", 1)
                    || sub_9691B0(v37, v38, "i", 1)
                    || sub_9691B0(v37, v38, "ll", 2)
                    || sub_9691B0(v37, v38, "us", 2)
                    || sub_9691B0(v37, v38, "ui", 2) )
                  {
                    goto LABEL_39;
                  }
                  v39 = "ull";
                  v40 = 3;
                }
                else
                {
                  if ( (unsigned __int8)sub_95CB50((const void **)&s1, "atomic.load.add.", 0x10u) )
                  {
                    v37 = s1;
                    v38 = v139;
                    if ( !sub_A7BBF0(s1, v139, "f32.p", 5u) )
                    {
                      v121 = "f64.p";
                      v122 = 5;
                      goto LABEL_460;
                    }
LABEL_39:
                    *a2 = 0;
                    return 1;
                  }
                  v125 = s1;
                  v126 = v139;
                  if ( sub_9691B0(s1, v139, "shfl.sync.i32", 13)
                    || sub_9691B0(v125, v126, "shfl.i32", 8)
                    || sub_9691B0(v125, v126, "read.cluster.info.i1", 20)
                    || sub_9691B0(v125, v126, "read.cluster.info.i32", 21)
                    || sub_9691B0(v125, v126, "cluster.barrier", 15)
                    || sub_9691B0(v125, v126, "cluster.barrier.aligned", 23)
                    || sub_9691B0(v125, v126, "idp2a", 5)
                    || sub_9691B0(v125, v126, "idp4a", 5)
                    || sub_A7BBF0(v125, v126, "shf.i", 5u)
                    || sub_A7BBF0(v125, v126, "rotate.i", 8u) )
                  {
                    goto LABEL_39;
                  }
                  if ( (unsigned __int8)sub_95CB50((const void **)&s1, "shf.", 4u)
                    && ((unsigned __int8)sub_95CB50((const void **)&s1, "r.", 2u)
                     || (unsigned __int8)sub_95CB50((const void **)&s1, "l.", 2u)) )
                  {
                    v37 = s1;
                    v38 = v139;
                    v130 = sub_9691B0(s1, v139, "clamp", 5);
                    v39 = "wrap";
                    if ( v130 )
                      goto LABEL_39;
                  }
                  else
                  {
                    v127 = s1;
                    v128 = v139;
                    if ( sub_9691B0(s1, v139, "bar.sync.all", 12) || sub_9691B0(v127, v128, "bar.sync.all.cnt", 16) )
                      goto LABEL_39;
                    if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "bitcast.", 8u) )
                    {
                      if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "rotate.", 7u) )
                      {
                        if ( (unsigned __int8)sub_95CB50((const void **)&s1, "ptr.gen.to.", 0xBu) )
                        {
                          v37 = s1;
                          v38 = v139;
                          if ( sub_A7BBF0(s1, v139, "local", 5u)
                            || sub_A7BBF0(v37, v38, "shared", 6u)
                            || sub_A7BBF0(v37, v38, "global", 6u) )
                          {
                            goto LABEL_39;
                          }
                          v121 = "constant";
                          v122 = 8;
                        }
                        else
                        {
                          if ( (unsigned __int8)sub_95CB50((const void **)&s1, "ptr.", 4u) )
                          {
                            if ( ((unsigned __int8)sub_95CB50((const void **)&s1, "local", 5u)
                               || (unsigned __int8)sub_95CB50((const void **)&s1, "shared", 6u)
                               || (unsigned __int8)sub_95CB50((const void **)&s1, "global", 6u)
                               || (unsigned __int8)sub_95CB50((const void **)&s1, "constant", 8u))
                              && sub_A7BBF0(s1, v139, ".to.gen", 7u) )
                            {
                              goto LABEL_39;
                            }
                            v37 = s1;
                            v38 = v139;
                            goto LABEL_100;
                          }
                          v131 = sub_95CB50((const void **)&s1, "ldg.global.", 0xBu);
                          v37 = s1;
                          v38 = v139;
                          if ( !v131 )
                            goto LABEL_100;
                          if ( sub_A7BBF0(s1, v139, "i.", 2u) || sub_A7BBF0(v37, v38, "f.", 2u) )
                            goto LABEL_39;
                          v121 = "p.";
                          v122 = 2;
                        }
LABEL_460:
                        if ( sub_A7BBF0(v37, v38, v121, v122) )
                          goto LABEL_39;
LABEL_100:
                        if ( v38 == 11 )
                        {
                          if ( !memcmp(v37, "permute.i32", 0xBu) )
                          {
                            v9 = 1;
                            *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 9248, 0, 0);
                            return v9;
                          }
                          goto LABEL_203;
                        }
                        if ( v38 <= 0xE || memcmp(v37, "tcgen05.commit.", 0xFu) )
                          goto LABEL_372;
                        if ( v38 - 15 <= 6 )
                        {
                          if ( v38 - 15 <= 2 )
                          {
LABEL_371:
                            v38 -= 15LL;
                            v37 = (_WORD *)((char *)v37 + 15);
LABEL_372:
                            if ( v38 > 0x18 && !memcmp(v37, "cp.async.bulk.tensor.g2s.", 0x19u) )
                            {
                              v73 = (_WORD *)((char *)v37 + 25);
                              if ( v38 - 25 > 4 )
                              {
                                if ( !memcmp((char *)v37 + 25, "tile.", 5u) )
                                {
                                  v38 -= 30LL;
                                  v73 = v37 + 15;
                                  if ( v38 == 2 )
                                  {
                                    v42 = 9222;
                                    if ( v37[15] != 25649 )
                                    {
                                      switch ( v37[15] )
                                      {
                                        case 0x6432:
                                          v42 = 9223;
                                          break;
                                        case 0x6433:
                                          v42 = 9224;
                                          break;
                                        case 0x6434:
                                          v42 = 9225;
                                          break;
                                        case 0x6435:
                                          v42 = 9226;
                                          break;
                                        default:
                                          goto LABEL_9;
                                      }
                                    }
                                    goto LABEL_383;
                                  }
LABEL_204:
                                  if ( v38 <= 0xD || memcmp(v73, "cp.async.bulk.", 0xEu) )
                                    goto LABEL_9;
                                  v74 = v73 + 7;
                                  if ( v38 == 27 )
                                  {
                                    if ( !memcmp(v74, "gmem.to.dsmem", 0xDu) )
                                    {
                                      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                                        sub_B2C6D0(a1);
                                      v75 = *(_QWORD *)(a1 + 40);
                                      v9 = 1;
                                      s2 = *(void **)(*(_QWORD *)(a1 + 96) + 208LL);
                                      *a2 = sub_B6E160(v75, 8316, &s2, 1);
                                      return v9;
                                    }
                                    goto LABEL_9;
                                  }
                                  if ( v38 == 38 )
                                  {
                                    if ( !memcmp(v74, "global.to.shared.cluster", 0x18u) )
                                    {
                                      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                                        sub_B2C6D0(a1);
                                      v124 = *(_QWORD *)(a1 + 40);
                                      v9 = 1;
                                      s2 = *(void **)(*(_QWORD *)(a1 + 96) + 168LL);
                                      *a2 = sub_B6E160(v124, 8315, &s2, 1);
                                      return v9;
                                    }
                                    goto LABEL_9;
                                  }
                                  if ( v38 - 14 <= 0x13 || memcmp(v74, "tensor.gmem.to.smem.", 0x14u) )
                                    goto LABEL_9;
                                  if ( v38 - 34 <= 8 )
                                  {
                                    if ( v38 != 36 )
                                      goto LABEL_9;
                                    switch ( v73[17] )
                                    {
                                      case 0x6431:
                                        v42 = 8324;
                                        break;
                                      case 0x6432:
                                        v42 = 8325;
                                        break;
                                      case 0x6433:
                                        v42 = 8326;
                                        break;
                                      case 0x6434:
                                        v42 = 8327;
                                        break;
                                      case 0x6435:
                                        v42 = 8328;
                                        break;
                                      default:
                                        goto LABEL_9;
                                    }
                                  }
                                  else
                                  {
                                    if ( memcmp(v73 + 17, "im2col.w.", 9u) || v38 != 45 )
                                      goto LABEL_9;
                                    v42 = 8329;
                                    if ( *(_WORD *)((char *)v73 + 43) != 25651 )
                                    {
                                      if ( *(_WORD *)((char *)v73 + 43) == 25652 )
                                      {
                                        v42 = 8330;
                                      }
                                      else
                                      {
                                        if ( *(_WORD *)((char *)v73 + 43) != 25653 )
                                          goto LABEL_9;
                                        v42 = 8331;
                                      }
                                    }
                                  }
                                  v123 = *(_QWORD *)(a1 + 104);
                                  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                                    sub_B2C6D0(a1);
                                  v112 = (unsigned int)(v123 - 2);
                                  goto LABEL_386;
                                }
                                if ( v38 - 25 > 6 && !memcmp((char *)v37 + 25, "im2col.", 7u) )
                                {
                                  v38 -= 32LL;
                                  v73 = v37 + 16;
                                  if ( v38 == 2 )
                                  {
                                    switch ( v37[16] )
                                    {
                                      case 0x6433:
                                        v42 = 9213;
                                        break;
                                      case 0x6434:
                                        v42 = 9214;
                                        break;
                                      case 0x6435:
                                        v42 = 9215;
                                        break;
                                      default:
                                        goto LABEL_9;
                                    }
LABEL_383:
                                    v111 = *(_QWORD *)(a1 + 104);
                                    if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                                      sub_B2C6D0(a1);
                                    v112 = (unsigned int)(v111 - 4);
LABEL_386:
                                    v43 = *(void **)(*(_QWORD *)(a1 + 96) + 40 * v112 + 8);
                                    goto LABEL_113;
                                  }
                                  goto LABEL_204;
                                }
                              }
                              v38 -= 25LL;
                              goto LABEL_204;
                            }
LABEL_203:
                            v73 = v37;
                            goto LABEL_204;
                          }
                        }
                        else if ( !memcmp((char *)v37 + 15, "arrive.", 7u) )
                        {
                          v119 = v37 + 11;
                          if ( v38 == 38 )
                          {
                            v42 = 10091;
                            if ( memcmp(v119, "multicast.shared", 0x10u) )
                              goto LABEL_9;
                          }
                          else
                          {
                            if ( v38 != 31 || memcmp(v119, "multicast", 9u) )
                              goto LABEL_9;
                            v42 = 10090;
                          }
                          if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                            sub_B2C6D0(a1);
                          v43 = *(void **)(*(_QWORD *)(a1 + 96) + 88LL);
                          goto LABEL_113;
                        }
                        if ( !memcmp((char *)v37 + 15, "mc.", 3u) )
                        {
                          v41 = v37 + 9;
                          if ( v38 == 21 )
                          {
                            v42 = 10095;
                            if ( memcmp(v41, "cg1", 3u) )
                            {
                              if ( memcmp(v41, "cg2", 3u) )
                                goto LABEL_9;
                              v42 = 10096;
                            }
                          }
                          else
                          {
                            if ( v38 != 28 )
                              goto LABEL_9;
                            if ( !memcmp(v41, "shared.cg1", 0xAu) )
                            {
                              v42 = 10097;
                            }
                            else
                            {
                              if ( memcmp(v41, "shared.cg2", 0xAu) )
                                goto LABEL_9;
                              v42 = 10098;
                            }
                          }
                          if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
                            sub_B2C6D0(a1);
                          v43 = *(void **)(*(_QWORD *)(a1 + 96) + 48LL);
LABEL_113:
                          v44 = *(_QWORD *)(a1 + 40);
                          v45 = v42;
                          s2 = v43;
                          v9 = 1;
                          *a2 = sub_B6E160(v44, v45, &s2, 1);
                          return v9;
                        }
                        goto LABEL_371;
                      }
                      v37 = s1;
                      v38 = v139;
                      if ( sub_9691B0(s1, v139, "b32", 3) || sub_9691B0(v37, v38, "b64", 3) )
                        goto LABEL_39;
                      v39 = "right.b64";
                      v40 = 9;
                      goto LABEL_99;
                    }
                    v37 = s1;
                    v38 = v139;
                    if ( sub_9691B0(s1, v139, "f2i", 3) )
                      goto LABEL_39;
                    if ( sub_9691B0(v37, v38, "i2f", 3) )
                      goto LABEL_39;
                    v129 = sub_9691B0(v37, v38, "ll2d", 4);
                    v39 = "d2ll";
                    if ( v129 )
                      goto LABEL_39;
                  }
                  v40 = 4;
                }
LABEL_99:
                if ( !sub_9691B0(v37, v38, v39, v40) )
                  goto LABEL_100;
                goto LABEL_39;
              }
              v86 = *(_QWORD *)(a1 + 40);
              v9 = 1;
              s2 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
              *a2 = sub_B6E160(v86, v83, &s2, 1);
              return v9;
            }
          }
          else
          {
            if ( v139 != 16 )
              goto LABEL_249;
            v82 = 8825;
            if ( memcmp(s1, "cluster.get.rank", 0x10u) )
              goto LABEL_249;
          }
          v9 = 1;
          *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), v82, 0, 0);
          return v9;
        }
        LODWORD(v150[0]) = 66;
        BYTE4(v150[0]) = 1;
      }
LABEL_324:
      LODWORD(v143) = 1;
      goto LABEL_243;
    case 'o':
      if ( v11 <= 0xA || *(_QWORD *)(v8 + 5) != 0x69737463656A626FLL || *(_WORD *)(v8 + 13) != 25978 || v8[15] != 46 )
        goto LABEL_9;
      v143 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        sub_B2C6D0(a1);
      v144 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 104) - 2LL) <= 1 )
        goto LABEL_56;
      sub_B6E0E0(&s2, 282, &v143, 2, *(_QWORD *)(a1 + 40), 0);
      v78 = n;
      v79 = s2;
      v80 = (const void *)sub_BD5D20(a1);
      if ( v78 == v81 && (!v78 || !memcmp(v80, v79, v78)) )
      {
        if ( s2 != v150 )
          j_j___libc_free_0(s2, v150[0] + 1LL);
        goto LABEL_9;
      }
      if ( s2 != v150 )
        j_j___libc_free_0(s2, v150[0] + 1LL);
LABEL_56:
      v9 = 1;
      sub_A7BA00(a1);
      *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), 282, &v143, 2);
      return v9;
    case 'p':
      if ( v11 <= 0xE
        || *(_QWORD *)(v8 + 5) != 0x6F6E6E612E727470LL
        || *(_DWORD *)(v8 + 13) != 1769234804
        || *(_WORD *)(v8 + 17) != 28271
        || v8[19] != 46
        || *(_QWORD *)(a1 + 104) != 4 )
      {
        goto LABEL_9;
      }
      sub_A7BA00(a1);
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
      {
        sub_B2C6D0(a1);
        v26 = *(_QWORD *)(a1 + 96);
        s2 = *(void **)(v26 + 8);
        if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a1);
          v26 = *(_QWORD *)(a1 + 96);
        }
      }
      else
      {
        v26 = *(_QWORD *)(a1 + 96);
        s2 = *(void **)(v26 + 8);
      }
      v27 = *(_QWORD *)(a1 + 40);
      v9 = 1;
      n = *(_QWORD *)(v26 + 48);
      *a2 = sub_B6E160(v27, 292, &s2, 2);
      return v9;
    case 'r':
      if ( v11 <= 5 || *(_DWORD *)(v8 + 5) != 1668508018 || *(_WORD *)(v8 + 9) != 11894 )
        goto LABEL_9;
      v28 = v7 - 11;
      s1 = v8 + 11;
      v139 = v28;
      if ( v28 == 8 )
      {
        v29 = 11874;
        if ( *(_QWORD *)(v8 + 11) != 0x6973643233736561LL )
        {
          if ( *(_QWORD *)(v8 + 11) != 0x6973653233736561LL )
            goto LABEL_63;
          v29 = 11876;
        }
      }
      else
      {
        if ( v28 != 9 )
        {
          if ( v28 <= 4 )
            goto LABEL_9;
LABEL_63:
          if ( *(_DWORD *)(v8 + 11) == 1798598003 && (v29 = 12031, v8[15] == 115)
            || *(_DWORD *)(v8 + 11) == 1697934707 && (v29 = 12030, v8[15] == 100) )
          {
            if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 24LL), 32)
              && !(unsigned __int8)sub_BCAC40(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL), 64) )
            {
              goto LABEL_9;
            }
            goto LABEL_225;
          }
          if ( v28 > 9 )
          {
            if ( *(_QWORD *)(v8 + 11) == 0x6973363532616873LL && *(_WORD *)(v8 + 19) == 12391 )
            {
              v30 = 12014;
              goto LABEL_74;
            }
            if ( *(_QWORD *)(v8 + 11) == 0x6973363532616873LL && *(_WORD *)(v8 + 19) == 12647 )
            {
              v30 = 12015;
LABEL_74:
              v9 = sub_BCAC40(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL), 64);
              if ( !(_BYTE)v9 )
                goto LABEL_9;
              sub_A7BA00(a1);
              *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), v30, 0, 0);
              return v9;
            }
            if ( *(_QWORD *)(v8 + 11) == 0x7573363532616873LL && *(_WORD *)(v8 + 19) == 12397 )
            {
              v30 = 12016;
              goto LABEL_74;
            }
            if ( *(_QWORD *)(v8 + 11) == 0x7573363532616873LL && *(_WORD *)(v8 + 19) == 12653 )
            {
              v30 = 12017;
              goto LABEL_74;
            }
          }
          if ( *(_DWORD *)(v8 + 11) == 1882418547 && v8[15] == 48 )
          {
            v30 = 12028;
          }
          else
          {
            if ( *(_DWORD *)(v8 + 11) != 1882418547 || v8[15] != 49 )
              goto LABEL_9;
            v30 = 12029;
          }
          goto LABEL_74;
        }
        if ( *(_QWORD *)(v8 + 11) != 0x6D73643233736561LL || (v29 = 11875, v8[19] != 105) )
        {
          if ( *(_QWORD *)(v8 + 11) != 0x6D73653233736561LL || v8[19] != 105 )
            goto LABEL_63;
          v29 = 11877;
        }
      }
      if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 24LL), 32) )
        goto LABEL_9;
LABEL_225:
      sub_A7BA00(a1);
      *a2 = sub_B6E160(*(_QWORD *)(a1 + 40), v29, 0, 0);
      return 1;
    case 's':
      if ( v7 == 24
        && !(*(_QWORD *)(v8 + 5) ^ 0x6F72706B63617473LL | *(_QWORD *)(v8 + 13) ^ 0x6863726F74636574LL)
        && *(_WORD *)(v8 + 21) == 25445
        && v8[23] == 107 )
      {
        goto LABEL_39;
      }
      goto LABEL_9;
    case 'v':
      if ( v7 != 19
        || *(_QWORD *)(v8 + 5) != 0x6F6E6E612E726176LL
        || *(_DWORD *)(v8 + 13) != 1769234804
        || *(_WORD *)(v8 + 17) != 28271
        || *(_QWORD *)(a1 + 104) != 4 )
      {
        goto LABEL_9;
      }
      sub_A7BA00(a1);
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
      {
        sub_B2C6D0(a1);
        v24 = *(_QWORD *)(a1 + 96);
        s2 = *(void **)(v24 + 8);
        if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a1);
          v24 = *(_QWORD *)(a1 + 96);
        }
      }
      else
      {
        v24 = *(_QWORD *)(a1 + 96);
        s2 = *(void **)(v24 + 8);
      }
      v25 = *(_QWORD *)(a1 + 40);
      v9 = 1;
      n = *(_QWORD *)(v24 + 48);
      *a2 = sub_B6E160(v25, 376, &s2, 2);
      return v9;
    case 'w':
      if ( v11 <= 4 )
        goto LABEL_9;
      if ( *(_DWORD *)(v8 + 5) != 1836278135 )
        goto LABEL_9;
      if ( v8[9] != 46 )
        goto LABEL_9;
      v20 = v7 - 10;
      s1 = v8 + 10;
      v139 = v7 - 10;
      if ( v7 - 10 <= 3 )
        goto LABEL_9;
      v21 = 14219;
      if ( *(_DWORD *)(v8 + 10) == 778136934
        || (v21 = 14222, *(_DWORD *)(v8 + 10) == 779316582)
        || v20 > 0xA
        && *(_QWORD *)(v8 + 10) == 0x656C6573656E616CLL
        && *((_WORD *)v8 + 9) == 29795
        && (v21 = 14218, v8[20] == 46) )
      {
        sub_A7BA00(a1);
        v22 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
LABEL_26:
        v23 = *(_QWORD *)(a1 + 40);
        s2 = v22;
        *a2 = sub_B6E160(v23, v21, &s2, 1);
        return 1;
      }
      if ( v20 <= 0xF || *(_QWORD *)(v8 + 10) ^ 0x317838692E746F64LL | *(_QWORD *)(v8 + 18) ^ 0x2E36317837692E36LL )
        goto LABEL_9;
      v84 = v7 - 26;
      v85 = v8 + 26;
      s1 = v8 + 26;
      v139 = v84;
      if ( v84 == 6 )
      {
        v29 = 14217;
        if ( memcmp(v85, "signed", 6u) )
          goto LABEL_9;
        goto LABEL_225;
      }
      if ( v84 == 10 && !memcmp(v85, "add.signed", 0xAu) )
      {
        v29 = 14216;
        goto LABEL_225;
      }
      goto LABEL_9;
    case 'x':
      if ( !(unsigned __int8)sub_A8A170((_QWORD *)a1, (__int64)s1, v139, a2) )
        goto LABEL_9;
      return 1;
    default:
      goto LABEL_9;
  }
}
