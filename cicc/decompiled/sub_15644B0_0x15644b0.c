// Function: sub_15644B0
// Address: 0x15644b0
//
__int64 __fastcall sub_15644B0(__int64 a1, void **a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // r13
  __int64 result; // rax
  __int64 v8; // r15
  unsigned __int64 v9; // rsi
  __int64 v10; // rdi
  char v11; // r9
  __int64 v12; // rax
  const char *v13; // rdx
  __int64 v14; // rdi
  bool v15; // zf
  const char *v16; // rdx
  unsigned __int64 v17; // r14
  _WORD *v18; // rcx
  bool v19; // r10
  char v20; // r8
  bool v21; // r9
  bool v22; // al
  bool v23; // si
  bool v24; // dl
  int v25; // eax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned int v33; // r15d
  const void *v34; // rax
  void *v35; // rdx
  const char *v36; // rdx
  size_t v37; // r15
  const void *v38; // rax
  __int64 v39; // rdx
  const char *v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // rax
  void *v43; // r15
  const void *v44; // rax
  void *v45; // rdx
  const char *v46; // rdx
  size_t v47; // r15
  const void *v48; // rax
  __int64 v49; // rdx
  const char *v50; // rdx
  __int64 v51; // rax
  void *v52; // r15
  const void *v53; // rax
  void *v54; // rdx
  const char *v55; // rdx
  size_t v56; // r15
  const void *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rax
  const void *v61; // rax
  void *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  const void *v66; // r9
  signed __int64 v67; // r8
  void *v68; // rax
  __int64 v69; // r15
  char *v70; // rdi
  int v71; // edx
  _QWORD *v72; // rsi
  __int64 v73; // rax
  void *v74; // r8
  char v75; // r15
  unsigned int v76; // r15d
  __int64 v77; // rax
  __int64 v78; // r12
  _QWORD *v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // r15
  const char *v83; // rdx
  __int64 v84; // r13
  __int64 v85; // rax
  char v86; // r14
  unsigned int v87; // r14d
  __int64 v88; // rax
  __int64 v89; // r12
  __int64 v90; // rax
  int v91; // eax
  int v92; // eax
  int v93; // eax
  int v94; // eax
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // r15
  __int64 v98; // rsi
  __int64 v99; // rdi
  __int64 v100; // r13
  __int64 v101; // rax
  const char *v102; // rdx
  int v103; // r14d
  int v104; // eax
  __int64 v105; // rsi
  char v106; // al
  const char *v107; // rdx
  const char *v108; // rdx
  char v109; // [rsp+8h] [rbp-E8h]
  signed __int64 v110; // [rsp+8h] [rbp-E8h]
  const void *s2; // [rsp+10h] [rbp-E0h]
  const void *s2a; // [rsp+10h] [rbp-E0h]
  void *s2b; // [rsp+10h] [rbp-E0h]
  bool s2c; // [rsp+10h] [rbp-E0h]
  void *s2d; // [rsp+10h] [rbp-E0h]
  _WORD *s1; // [rsp+18h] [rbp-D8h]
  char s1j; // [rsp+18h] [rbp-D8h]
  char s1k; // [rsp+18h] [rbp-D8h]
  void *s1a; // [rsp+18h] [rbp-D8h]
  void *s1b; // [rsp+18h] [rbp-D8h]
  const void *s1c; // [rsp+18h] [rbp-D8h]
  void *s1d; // [rsp+18h] [rbp-D8h]
  const void *s1e; // [rsp+18h] [rbp-D8h]
  void *s1f; // [rsp+18h] [rbp-D8h]
  const void *s1g; // [rsp+18h] [rbp-D8h]
  void *s1h; // [rsp+18h] [rbp-D8h]
  void *s1i; // [rsp+18h] [rbp-D8h]
  _WORD *s1l; // [rsp+18h] [rbp-D8h]
  void *s1m; // [rsp+18h] [rbp-D8h]
  __int64 v130; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int64 v131; // [rsp+28h] [rbp-C8h]
  _BYTE v132[16]; // [rsp+30h] [rbp-C0h] BYREF
  const char *v133; // [rsp+40h] [rbp-B0h] BYREF
  __int64 *v134; // [rsp+48h] [rbp-A8h]
  __int16 v135; // [rsp+50h] [rbp-A0h]
  const char **v136; // [rsp+60h] [rbp-90h] BYREF
  const char *v137; // [rsp+68h] [rbp-88h]
  __int16 v138; // [rsp+70h] [rbp-80h]
  void *v139; // [rsp+80h] [rbp-70h] BYREF
  size_t n; // [rsp+88h] [rbp-68h]
  _QWORD dest[2]; // [rsp+90h] [rbp-60h] BYREF
  int v142; // [rsp+A0h] [rbp-50h]

  v4 = sub_1649960(a1);
  v131 = v5;
  v6 = v5;
  v130 = v4;
  result = 0;
  if ( v5 > 8 )
  {
    v8 = v130;
    if ( *(_DWORD *)v130 != 1836477548 || *(_BYTE *)(v130 + 4) != 46 )
      return 0;
    v9 = v5 - 5;
    v10 = v130 + 5;
    v130 += 5;
    v131 = v5 - 5;
    switch ( *(_BYTE *)(v8 + 5) )
    {
      case 'a':
        if ( v9 <= 7 )
          goto LABEL_52;
        if ( *(_QWORD *)(v8 + 5) == 0x746962722E6D7261LL )
          goto LABEL_364;
        if ( v9 <= 0xB )
          goto LABEL_52;
        if ( *(_QWORD *)(v8 + 5) == 0x2E34366863726161LL && *(_DWORD *)(v8 + 13) == 1953063538 )
        {
LABEL_364:
          if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
            sub_15E08E0(a1);
          v32 = *(_QWORD *)(a1 + 40);
          v139 = **(void ***)(a1 + 88);
          *a2 = (void *)sub_15E26F0(v32, 5, &v139, 1);
          return 1;
        }
        if ( v5 != 17 )
        {
          if ( *(_QWORD *)(v8 + 5) == 0x6E6F656E2E6D7261LL
            && *(_DWORD *)(v8 + 13) == 1818457646
            && *(_BYTE *)(v8 + 17) == 122 )
          {
            if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
              sub_15E08E0(a1);
            v133 = **(const char ***)(a1 + 88);
            v80 = sub_15E0530(a1);
            v134 = (__int64 *)sub_1643320(v80);
            v81 = sub_1644EA0(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL), &v133, 2, 0);
            v82 = *(_QWORD *)(a1 + 40);
            v83 = 0;
            v84 = v81;
            v85 = v131;
            if ( v131 > 0xD )
            {
              v83 = (const char *)(v131 - 14);
              v85 = 14;
            }
            v137 = v83;
            v136 = (const char **)(v130 + v85);
            LOWORD(dest[0]) = 1283;
            v86 = *(_BYTE *)(a1 + 32);
            v139 = "llvm.ctlz.";
            n = (size_t)&v136;
            v87 = v86 & 0xF;
            v88 = sub_1648B60(120);
            v89 = v88;
            if ( v88 )
              sub_15E2490(v88, v84, v87, &v139, v82);
            *a2 = (void *)v89;
            return 1;
          }
          if ( *(_QWORD *)(v8 + 5) == 0x6E6F656E2E6D7261LL
            && *(_DWORD *)(v8 + 13) == 1852012078
            && *(_BYTE *)(v8 + 17) == 116 )
          {
            if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
              sub_15E08E0(a1);
            v29 = *(_QWORD *)(a1 + 40);
            v139 = **(void ***)(a1 + 88);
            *a2 = (void *)sub_15E26F0(v29, 32, &v139, 1);
            return 1;
          }
        }
LABEL_52:
        sub_16C9340(v132, "^arm\\.neon\\.vld([1234]|[234]lane)\\.v[a-z0-9]*$", 46, 0);
        if ( (unsigned __int8)sub_16C9490(v132, v130, v131, 0) )
        {
          v63 = *(_QWORD *)(a1 + 24);
          v64 = *(_QWORD *)(v63 + 16);
          v65 = *(unsigned int *)(v63 + 12);
          v139 = dest;
          n = 0x400000000LL;
          v65 *= 8;
          v66 = (const void *)(v64 + 8);
          v67 = v65 - 8;
          v68 = (void *)(v64 + v65);
          v69 = v67 >> 3;
          if ( (unsigned __int64)v67 > 0x20 )
          {
            v110 = v67;
            s2d = (void *)(v64 + 8);
            s1m = v68;
            sub_16CD150(&v139, dest, v67 >> 3, 8);
            v72 = v139;
            v71 = n;
            v68 = s1m;
            v66 = s2d;
            v67 = v110;
            v70 = (char *)v139 + 8 * (unsigned int)n;
          }
          else
          {
            v70 = (char *)dest;
            v71 = 0;
            v72 = dest;
          }
          if ( v68 != v66 )
          {
            memcpy(v70, v66, v67);
            v72 = v139;
            v71 = n;
          }
          LODWORD(n) = v69 + v71;
          v73 = sub_1644EA0(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL), v72, (unsigned int)(v69 + v71), 0);
          v74 = *(void **)(a1 + 40);
          v75 = *(_BYTE *)(a1 + 32);
          s2b = (void *)v73;
          v133 = "llvm.";
          v76 = v75 & 0xF;
          v134 = &v130;
          v136 = &v133;
          s1i = v74;
          v135 = 1283;
          v137 = ".p0i8";
          v138 = 770;
          v77 = sub_1648B60(120);
          v78 = v77;
          if ( v77 )
            sub_15E2490(v77, s2b, v76, &v136, s1i);
          v79 = v139;
          *a2 = (void *)v78;
          if ( v79 != dest )
            _libc_free((unsigned __int64)v79);
          goto LABEL_438;
        }
        sub_16C9340(&v136, "^arm\\.neon\\.vst([1234]|[234]lane)\\.v[a-z0-9]*$", 46, 0);
        if ( (unsigned __int8)sub_16C9490(&v136, v130, v131, 0) )
        {
          v95 = *(_QWORD *)(a1 + 24);
          v96 = *(_QWORD *)(v95 + 16);
          v97 = (8LL * *(unsigned int *)(v95 + 12) - 8) >> 3;
          v139 = *(void **)(v96 + 8);
          n = *(_QWORD *)(v96 + 16);
          if ( sub_16D20C0(&v130, "lane", 4, 0) == -1 )
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), dword_4293C40[v97 - 3], &v139, 2);
          else
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), dword_4293C30[v97 - 5], &v139, 2);
          goto LABEL_535;
        }
        if ( v131 == 22 )
        {
          if ( !(*(_QWORD *)v130 ^ 0x2E34366863726161LL | *(_QWORD *)(v130 + 8) ^ 0x702E646165726874LL)
            && *(_DWORD *)(v130 + 16) == 1953393007
            && *(_WORD *)(v130 + 20) == 29285 )
          {
            goto LABEL_534;
          }
        }
        else if ( v131 == 18
               && !(*(_QWORD *)v130 ^ 0x657268742E6D7261LL | *(_QWORD *)(v130 + 8) ^ 0x746E696F702E6461LL)
               && *(_WORD *)(v130 + 16) == 29285 )
        {
LABEL_534:
          *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 204, 0, 0);
LABEL_535:
          sub_16C93F0(&v136);
LABEL_438:
          sub_16C93F0(v132);
          return 1;
        }
        sub_16C93F0(&v136);
        sub_16C93F0(v132);
        goto LABEL_11;
      case 'c':
        if ( v5 == 9 )
          goto LABEL_11;
        if ( *(_DWORD *)(v8 + 5) == 2053928035 && *(_BYTE *)(v8 + 9) == 46 )
        {
          if ( *(_QWORD *)(a1 + 96) != 1 )
            goto LABEL_11;
          sub_1564080(a1);
          if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
            sub_15E08E0(a1);
          v41 = *(_QWORD *)(a1 + 40);
          v139 = **(void ***)(a1 + 88);
          *a2 = (void *)sub_15E26F0(v41, 31, &v139, 1);
          return 1;
        }
        if ( *(_DWORD *)(v8 + 5) != 2054452323 || *(_BYTE *)(v8 + 9) != 46 || *(_QWORD *)(a1 + 96) != 1 )
          goto LABEL_11;
        sub_1564080(a1);
        if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
          sub_15E08E0(a1);
        v14 = *(_QWORD *)(a1 + 40);
        v139 = **(void ***)(a1 + 88);
        *a2 = (void *)sub_15E26F0(v14, 33, &v139, 1);
        return 1;
      case 'd':
        if ( v5 != 14
          || *(_QWORD *)(v8 + 5) != 0x756C61762E676264LL
          || *(_BYTE *)(v8 + 13) != 101
          || *(_QWORD *)(a1 + 96) != 4 )
        {
          goto LABEL_11;
        }
        sub_1564080(a1);
        *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 38, 0, 0);
        return 1;
      case 'i':
      case 'l':
        if ( v9 <= 0xD )
          goto LABEL_333;
        if ( *(_QWORD *)(v8 + 5) == 0x656D69746566696CLL
          && *(_DWORD *)(v8 + 13) == 1635021614
          && *(_WORD *)(v8 + 17) == 29810 )
        {
          v33 = 117;
        }
        else
        {
          if ( v5 == 19
            || *(_QWORD *)(v8 + 5) != 0x6E61697261766E69LL
            || *(_DWORD *)(v8 + 13) != 1953705588
            || *(_WORD *)(v8 + 17) != 29281
            || *(_BYTE *)(v8 + 19) != 116 )
          {
            if ( *(_QWORD *)(v8 + 5) == 0x656D69746566696CLL && *(_DWORD *)(v8 + 13) == 1684956462 )
            {
LABEL_453:
              v33 = 116;
              v59 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
              v60 = 16;
              goto LABEL_425;
            }
LABEL_18:
            if ( *(_QWORD *)v10 != 0x6E61697261766E69LL
              || *(_DWORD *)(v10 + 8) != 1852124788
              || *(_BYTE *)(v10 + 12) != 100 )
            {
              goto LABEL_19;
            }
            v33 = 113;
            v59 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
            v60 = 24;
LABEL_425:
            v133 = *(const char **)(v59 + v60);
            sub_15E1070(&v139, v33, &v133, 1);
            s1h = (void *)n;
            s2a = v139;
            v61 = (const void *)sub_1649960(a1);
            if ( s1h == v62 && (!s1h || !memcmp(v61, s2a, (size_t)s1h)) )
            {
              if ( v139 != dest )
                j_j___libc_free_0(v139, dest[0] + 1LL);
LABEL_19:
              if ( v131 > 0x16
                && !(*(_QWORD *)v130 ^ 0x6E61697261766E69LL | *(_QWORD *)(v130 + 8) ^ 0x2E70756F72672E74LL)
                && *(_DWORD *)(v130 + 16) == 1920098658
                && *(_WORD *)(v130 + 20) == 25961
                && *(_BYTE *)(v130 + 22) == 114 )
              {
                v133 = *(const char **)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL);
                v12 = sub_1649960(a1);
                LOWORD(dest[0]) = 773;
                v136 = (const char **)v12;
                v139 = &v136;
                v137 = v13;
                n = (size_t)".old";
                sub_164B780(a1, &v139);
                *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 115, &v133, 1);
                return 1;
              }
              goto LABEL_11;
            }
LABEL_379:
            if ( v139 != dest )
              j_j___libc_free_0(v139, dest[0] + 1LL);
            v136 = (const char **)sub_1649960(a1);
            v139 = &v136;
            v137 = v36;
            LOWORD(dest[0]) = 773;
            n = (size_t)".old";
            sub_164B780(a1, &v139);
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), v33, &v133, 1);
            return 1;
          }
          v33 = 114;
        }
        v133 = *(const char **)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 16LL);
        sub_15E1070(&v139, v33, &v133, 1);
        s1b = (void *)n;
        s2 = v139;
        v34 = (const void *)sub_1649960(a1);
        if ( s1b != v35 || s1b && memcmp(v34, s2, (size_t)s1b) )
          goto LABEL_379;
        if ( v139 != dest )
          j_j___libc_free_0(v139, dest[0] + 1LL);
        v9 = v131;
LABEL_333:
        if ( v9 <= 0xB )
        {
LABEL_11:
          sub_15E33D0(&v139, a1);
          result = (unsigned __int8)n;
          if ( (_BYTE)n )
            *a2 = v139;
          return result;
        }
        v10 = v130;
        if ( *(_QWORD *)v130 == 0x656D69746566696CLL && *(_DWORD *)(v130 + 8) == 1684956462 )
          goto LABEL_453;
        if ( v9 == 12 )
          goto LABEL_19;
        goto LABEL_18;
      case 'm':
        if ( v9 <= 0xB )
          goto LABEL_354;
        if ( *(_QWORD *)(v8 + 5) != 0x6C2E64656B73616DLL || *(_DWORD *)(v8 + 13) != 778330479 )
          goto LABEL_319;
        v133 = **(const char ***)(*(_QWORD *)(a1 + 24) + 16LL);
        if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
          sub_15E08E0(a1);
        v134 = **(__int64 ***)(a1 + 88);
        sub_15E1070(&v139, 129, &v133, 2);
        v37 = n;
        s1c = v139;
        v38 = (const void *)sub_1649960(a1);
        if ( v37 != v39 || v37 && memcmp(v38, s1c, v37) )
        {
          if ( v139 != dest )
            j_j___libc_free_0(v139, dest[0] + 1LL);
          v136 = (const char **)sub_1649960(a1);
          v139 = &v136;
          v137 = v40;
          LOWORD(dest[0]) = 773;
          n = (size_t)".old";
          sub_164B780(a1, &v139);
          *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 129, &v133, 2);
          return 1;
        }
        if ( v139 != dest )
          j_j___libc_free_0(v139, dest[0] + 1LL);
        v9 = v131;
LABEL_319:
        if ( v9 <= 0xC )
          goto LABEL_354;
        if ( *(_QWORD *)v130 == 0x732E64656B73616DLL
          && *(_DWORD *)(v130 + 8) == 1701998452
          && *(_BYTE *)(v130 + 12) == 46 )
        {
          v42 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
          v133 = *(const char **)(v42 + 8);
          v134 = *(__int64 **)(v42 + 16);
          sub_15E1070(&v139, 131, &v133, 2);
          v43 = v139;
          s1d = (void *)n;
          v44 = (const void *)sub_1649960(a1);
          if ( s1d != v45 || s1d && memcmp(v44, v43, (size_t)s1d) )
          {
            if ( v139 != dest )
              j_j___libc_free_0(v139, dest[0] + 1LL);
            v136 = (const char **)sub_1649960(a1);
            LOWORD(dest[0]) = 773;
            v139 = &v136;
            v137 = v46;
            n = (size_t)".old";
            sub_164B780(a1, &v139);
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 131, &v133, 2);
            return 1;
          }
          if ( v139 != dest )
            j_j___libc_free_0(v139, dest[0] + 1LL);
          v9 = v131;
        }
        if ( v9 <= 0xD )
          goto LABEL_354;
        v28 = v130;
        if ( *(_QWORD *)v130 == 0x672E64656B73616DLL
          && *(_DWORD *)(v130 + 8) == 1701344353
          && *(_WORD *)(v130 + 12) == 11890 )
        {
          v133 = **(const char ***)(*(_QWORD *)(a1 + 24) + 16LL);
          if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
            sub_15E08E0(a1);
          v134 = **(__int64 ***)(a1 + 88);
          sub_15E1070(&v139, 128, &v133, 2);
          v47 = n;
          s1e = v139;
          v48 = (const void *)sub_1649960(a1);
          if ( v47 != v49 || v47 && memcmp(v48, s1e, v47) )
          {
            if ( v139 != dest )
              j_j___libc_free_0(v139, dest[0] + 1LL);
            v136 = (const char **)sub_1649960(a1);
            LOWORD(dest[0]) = 773;
            v139 = &v136;
            v137 = v50;
            n = (size_t)".old";
            sub_164B780(a1, &v139);
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 128, &v133, 2);
            return 1;
          }
          if ( v139 != dest )
            j_j___libc_free_0(v139, dest[0] + 1LL);
          v9 = v131;
          if ( v131 <= 0xE )
            goto LABEL_354;
          v28 = v130;
        }
        else if ( v9 == 14 )
        {
          if ( *(_DWORD *)v130 == 1668113773
            && *(_WORD *)(v130 + 4) == 31088
            && *(_BYTE *)(v130 + 6) == 46
            && *(_QWORD *)(a1 + 96) == 5 )
          {
LABEL_442:
            sub_1564080(a1);
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 133, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL, 3);
            return 1;
          }
          goto LABEL_325;
        }
        if ( *(_QWORD *)v28 != 0x732E64656B73616DLL
          || *(_DWORD *)(v28 + 8) != 1953784163
          || *(_WORD *)(v28 + 12) != 29285
          || *(_BYTE *)(v28 + 14) != 46 )
        {
          v9 = v131;
          goto LABEL_356;
        }
        v51 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
        v133 = *(const char **)(v51 + 8);
        v134 = *(__int64 **)(v51 + 16);
        sub_15E1070(&v139, 130, &v133, 2);
        v52 = v139;
        s1f = (void *)n;
        v53 = (const void *)sub_1649960(a1);
        if ( s1f != v54 || s1f && memcmp(v53, v52, (size_t)s1f) )
        {
          if ( v139 != dest )
            j_j___libc_free_0(v139, dest[0] + 1LL);
          v136 = (const char **)sub_1649960(a1);
          v139 = &v136;
          v137 = v55;
          LOWORD(dest[0]) = 773;
          n = (size_t)".old";
          sub_164B780(a1, &v139);
          *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 130, &v133, 2);
          return 1;
        }
        if ( v139 != dest )
          j_j___libc_free_0(v139, dest[0] + 1LL);
        v9 = v131;
LABEL_354:
        if ( v9 <= 6 )
          goto LABEL_11;
        v28 = v130;
LABEL_356:
        if ( *(_DWORD *)v28 == 1668113773
          && *(_WORD *)(v28 + 4) == 31088
          && *(_BYTE *)(v28 + 6) == 46
          && *(_QWORD *)(a1 + 96) == 5 )
        {
          goto LABEL_442;
        }
        if ( v9 <= 7 )
        {
LABEL_358:
          if ( *(_DWORD *)v28 == 1936549229
            && *(_WORD *)(v28 + 4) == 29797
            && *(_BYTE *)(v28 + 6) == 46
            && *(_QWORD *)(a1 + 96) == 5 )
          {
            sub_1564080(a1);
            v30 = *(_QWORD *)(a1 + 40);
            v31 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
            v139 = *(void **)(v31 + 8);
            n = *(_QWORD *)(v31 + 24);
            *a2 = (void *)sub_15E26F0(v30, 137, &v139, 2);
            return 1;
          }
          goto LABEL_11;
        }
LABEL_325:
        if ( *(_QWORD *)v28 != 0x2E65766F6D6D656DLL )
          goto LABEL_358;
        if ( *(_QWORD *)(a1 + 96) != 5 )
          goto LABEL_11;
        sub_1564080(a1);
        *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 135, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL, 3);
        return 1;
      case 'n':
        if ( v5 == 9 || *(_DWORD *)(v8 + 5) != 1836480110 || *(_BYTE *)(v8 + 9) != 46 )
          goto LABEL_11;
        v17 = v5 - 10;
        v18 = (_WORD *)(v8 + 10);
        v130 = v8 + 10;
        v131 = v5 - 10;
        v19 = v5 == 15;
        if ( v5 == 16 )
        {
          if ( *(_DWORD *)(v8 + 10) == 1986359906 && *(_WORD *)(v8 + 14) == 12851
            || *(_DWORD *)(v8 + 10) == 1986359906 && *(_WORD *)(v8 + 14) == 13366 )
          {
            v98 = 5;
            v20 = 1;
          }
          else
          {
            if ( *(_DWORD *)(v8 + 10) != 1668312944 || *(_WORD *)(v8 + 14) != 26926 )
            {
              v20 = 1;
              goto LABEL_63;
            }
            v20 = 1;
            v98 = 32;
          }
        }
        else
        {
          v20 = 0;
          if ( v5 != 15 )
            goto LABEL_62;
          if ( *(_DWORD *)(v8 + 10) != 779775075 || *(_BYTE *)(v8 + 14) != 105 )
          {
            v20 = 0;
            goto LABEL_63;
          }
          v98 = 31;
          v20 = 0;
        }
        if ( *(_QWORD *)(a1 + 96) == 1 )
        {
          v99 = *(_QWORD *)(a1 + 40);
          v139 = **(void ***)(*(_QWORD *)(a1 + 24) + 16LL);
          *a2 = (void *)sub_15E26F0(v99, v98, &v139, 1);
          return 1;
        }
LABEL_62:
        if ( v5 == 33 )
        {
          if ( !(*(_QWORD *)(v8 + 10) ^ 0x7065636170737369LL | *(_QWORD *)(v8 + 18) ^ 0x72657473756C632ELL)
            && *(_DWORD *)(v8 + 26) == 1634235182
            && *(_WORD *)(v8 + 30) == 25970
            && *(_BYTE *)(v8 + 32) == 100 )
          {
            *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 4053, 0, 0);
            return 1;
          }
          v21 = 0;
          goto LABEL_64;
        }
LABEL_63:
        v21 = v17 == 7;
        if ( v5 == 15 )
        {
          if ( *(_DWORD *)(v8 + 10) != 779313761 || (v91 = 0, *(_BYTE *)(v8 + 14) != 105) )
            v91 = 1;
          v22 = v91 == 0;
          v23 = !v22;
          v24 = v22;
          goto LABEL_67;
        }
LABEL_64:
        if ( v20 )
        {
          if ( *(_DWORD *)(v8 + 10) == 779313761 && *(_WORD *)(v8 + 14) == 27756 )
            goto LABEL_76;
          v109 = v20;
          s2c = v21;
          v92 = memcmp((const void *)(v8 + 10), "clz.ll", 6u);
          v18 = (_WORD *)(v8 + 10);
          v21 = s2c;
          v20 = v109;
          v19 = v6 == 15;
          if ( !v92 )
            goto LABEL_76;
          if ( v6 == 13 )
          {
            v24 = 0;
LABEL_552:
            if ( *(_WORD *)(v8 + 10) == 12904 )
            {
              if ( *((_BYTE *)v18 + 2) == 102 )
                v24 = 1;
              v22 = v24;
            }
            else
            {
              v22 = v24;
            }
LABEL_69:
            if ( v22 )
              goto LABEL_76;
            if ( v20 )
            {
              if ( *(_DWORD *)(v8 + 10) == 779641197 && v18[2] == 27756 )
                goto LABEL_76;
              s1 = v18;
              v25 = memcmp(v18, "max.ui", 6u);
              v18 = s1;
              if ( !v25 )
                goto LABEL_76;
              if ( v6 != 15 )
              {
                if ( *(_DWORD *)(v8 + 10) == 778987885 && s1[2] == 27756 )
                  goto LABEL_76;
                v26 = memcmp(s1, "min.ui", 6u);
                v18 = s1;
                if ( !v26 )
                  goto LABEL_76;
LABEL_10:
                v11 = 0;
                if ( v17 <= 0xF )
                  goto LABEL_11;
                goto LABEL_78;
              }
              goto LABEL_8;
            }
            goto LABEL_527;
          }
          v23 = v109;
          v22 = 0;
          v24 = 0;
LABEL_517:
          if ( v23 && v19 )
          {
            if ( *(_DWORD *)(v8 + 10) == 779641197 && *((_BYTE *)v18 + 4) == 105 || v22 )
              goto LABEL_76;
LABEL_8:
            if ( *(_DWORD *)(v8 + 10) != 778987885 || *((_BYTE *)v18 + 4) != 105 )
              goto LABEL_10;
LABEL_76:
            if ( v17 <= 0xF )
              goto LABEL_42;
            v11 = 1;
LABEL_78:
            if ( !(*(_QWORD *)(v8 + 10) ^ 0x722E657461746F72LL | *((_QWORD *)v18 + 1) ^ 0x3436622E74686769LL) )
            {
              s1j = v11;
              v27 = *(_QWORD *)(a1 + 40) + 240LL;
              v130 = v8 + 26;
              v136 = (const char **)v27;
              v131 = v6 - 26;
              v138 = 260;
              sub_16E1010(&v139);
              v11 = s1j;
              if ( (unsigned int)(v142 - 34) >= 2 )
                v11 = 1;
              if ( v139 != dest )
              {
                s1k = v11;
                j_j___libc_free_0(v139, dest[0] + 1LL);
                v11 = s1k;
              }
            }
            if ( v11 )
            {
LABEL_42:
              *a2 = 0;
              return 1;
            }
            goto LABEL_11;
          }
          goto LABEL_69;
        }
        if ( v21 )
        {
          if ( *(_DWORD *)(v8 + 10) != 1668312944
            || *(_WORD *)(v8 + 14) != 27694
            || (v94 = 0, *(_BYTE *)(v8 + 16) != 108) )
          {
            v94 = 1;
          }
          v24 = v94 == 0;
          if ( !v94 )
            goto LABEL_76;
LABEL_527:
          if ( v21 )
          {
            if ( (*(_DWORD *)(v8 + 10) != 779641197 || v18[2] != 27765 || *((_BYTE *)v18 + 6) != 108) && !v24 )
            {
              s1l = v18;
              v93 = memcmp(v18, "min.ull", 7u);
              v18 = s1l;
              if ( v93 )
                goto LABEL_11;
            }
            goto LABEL_76;
          }
          if ( v6 != 15 )
            goto LABEL_10;
          goto LABEL_8;
        }
        v22 = 0;
        v23 = 1;
        v24 = 0;
LABEL_67:
        if ( v6 == 13 )
        {
          if ( !v23 )
            goto LABEL_69;
          goto LABEL_552;
        }
        goto LABEL_517;
      case 'o':
        if ( v9 <= 0xA
          || *(_QWORD *)(v8 + 5) != 0x69737463656A626FLL
          || *(_WORD *)(v8 + 13) != 25978
          || *(_BYTE *)(v8 + 15) != 46 )
        {
          goto LABEL_11;
        }
        v133 = **(const char ***)(*(_QWORD *)(a1 + 24) + 16LL);
        if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
          sub_15E08E0(a1);
        v15 = *(_QWORD *)(a1 + 96) == 2;
        v134 = **(__int64 ***)(a1 + 88);
        if ( v15 )
          goto LABEL_50;
        sub_15E1070(&v139, 144, &v133, 2);
        v56 = n;
        s1g = v139;
        v57 = (const void *)sub_1649960(a1);
        if ( v56 == v58 && (!v56 || !memcmp(v57, s1g, v56)) )
        {
          if ( v139 != dest )
            j_j___libc_free_0(v139, dest[0] + 1LL);
          goto LABEL_11;
        }
        if ( v139 != dest )
          j_j___libc_free_0(v139, dest[0] + 1LL);
LABEL_50:
        v136 = (const char **)sub_1649960(a1);
        LOWORD(dest[0]) = 773;
        v139 = &v136;
        v137 = v16;
        n = (size_t)".old";
        sub_164B780(a1, &v139);
        *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 144, &v133, 2);
        return 1;
      case 's':
        if ( v5 == 24
          && !(*(_QWORD *)(v8 + 5) ^ 0x6F72706B63617473LL | *(_QWORD *)(v8 + 13) ^ 0x6863726F74636574LL)
          && *(_WORD *)(v8 + 21) == 25445
          && *(_BYTE *)(v8 + 23) == 107 )
        {
          goto LABEL_42;
        }
        goto LABEL_11;
      case 'x':
        if ( *(_DWORD *)(v8 + 5) != 775305336 )
          goto LABEL_11;
        s1a = (void *)(v5 - 9);
        if ( v5 == 25 )
        {
          v90 = *(_QWORD *)(v8 + 9) ^ 0x61702E3365737373LL;
          if ( !(v90 | *(_QWORD *)(v8 + 17) ^ 0x3832312E622E7362LL)
            || !(v90 | *(_QWORD *)(v8 + 17) ^ 0x3832312E772E7362LL)
            || !(v90 | *(_QWORD *)(v8 + 17) ^ 0x3832312E642E7362LL) )
          {
            goto LABEL_42;
          }
        }
        else if ( (unsigned __int64)s1a <= 0xC )
        {
          if ( (unsigned __int64)s1a <= 0xA )
          {
LABEL_89:
            if ( (unsigned __int64)s1a <= 9 )
            {
LABEL_90:
              if ( s1a == (void *)12 )
              {
                if ( !memcmp((const void *)(v8 + 9), "sse2.sqrt.sd", 0xCu) )
                  goto LABEL_42;
LABEL_92:
                if ( !memcmp((const void *)(v8 + 9), "avx.sqrt.p", 0xAu) )
                  goto LABEL_42;
                goto LABEL_93;
              }
              if ( (unsigned __int64)s1a <= 9 )
                goto LABEL_831;
              goto LABEL_865;
            }
            if ( !memcmp((const void *)(v8 + 9), "avx2.pabs.", 0xAu) )
              goto LABEL_42;
LABEL_598:
            if ( s1a == (void *)11 )
            {
              if ( !memcmp((const void *)(v8 + 9), "sse.sqrt.ss", 0xBu) )
                goto LABEL_42;
              goto LABEL_92;
            }
            goto LABEL_90;
          }
LABEL_466:
          if ( *(_QWORD *)(v8 + 9) == 0x616D66762E616D66LL && *(_WORD *)(v8 + 17) == 25700 && *(_BYTE *)(v8 + 19) == 46
            || *(_QWORD *)(v8 + 9) == 0x736D66762E616D66LL && *(_WORD *)(v8 + 17) == 25205 && *(_BYTE *)(v8 + 19) == 46 )
          {
            goto LABEL_42;
          }
          if ( (unsigned __int64)s1a <= 0xD )
          {
            if ( (unsigned __int64)s1a <= 0xB )
              goto LABEL_602;
          }
          else if ( !memcmp((const void *)(v8 + 9), "fma.vfmaddsub.", 0xEu)
                 || !memcmp((const void *)(v8 + 9), "fma.vfmsubadd.", 0xEu) )
          {
            goto LABEL_42;
          }
          if ( !memcmp((const void *)(v8 + 9), "fma.vfnmadd.", 0xCu)
            || !memcmp((const void *)(v8 + 9), "fma.vfnmsub.", 0xCu) )
          {
            goto LABEL_42;
          }
          if ( (unsigned __int64)s1a > 0x12 )
          {
            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vfmadd.", 0x13u)
              || s1a != (void *)19
              && (!memcmp((const void *)(v8 + 9), "avx512.mask.vfnmadd.", 0x14u)
               || !memcmp((const void *)(v8 + 9), "avx512.mask.vfnmsub.", 0x14u)
               || !memcmp((const void *)(v8 + 9), "avx512.mask3.vfmadd.", 0x14u)
               || !memcmp((const void *)(v8 + 9), "avx512.maskz.vfmadd.", 0x14u)
               || !memcmp((const void *)(v8 + 9), "avx512.mask3.vfmsub.", 0x14u)
               || s1a != (void *)20
               && (!memcmp((const void *)(v8 + 9), "avx512.mask3.vfnmsub.", 0x15u)
                || s1a != (void *)21
                && (!memcmp((const void *)(v8 + 9), "avx512.mask.vfmaddsub.", 0x16u)
                 || s1a != (void *)22
                 && (!memcmp((const void *)(v8 + 9), "avx512.maskz.vfmaddsub.", 0x17u)
                  || !memcmp((const void *)(v8 + 9), "avx512.mask3.vfmaddsub.", 0x17u)
                  || !memcmp((const void *)(v8 + 9), "avx512.mask3.vfmsubadd.", 0x17u))))) )
            {
              goto LABEL_42;
            }
            goto LABEL_489;
          }
          if ( s1a == (void *)18 )
          {
LABEL_489:
            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.i", 0x12u)
              || !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.f", 0x12u) )
            {
              goto LABEL_42;
            }
            goto LABEL_491;
          }
LABEL_602:
          if ( (unsigned __int64)s1a <= 0xC )
            goto LABEL_89;
LABEL_491:
          if ( !memcmp((const void *)(v8 + 9), "avx512.kunpck", 0xDu)
            || !memcmp((const void *)(v8 + 9), "avx2.pabs.", 0xAu) )
          {
            goto LABEL_42;
          }
          if ( (unsigned __int64)s1a > 0x10 )
          {
            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pabs.", 0x11u)
              || !memcmp((const void *)(v8 + 9), "avx512.broadcastm", 0x11u) )
            {
              goto LABEL_42;
            }
            if ( s1a != (void *)17 )
            {
              if ( !memcmp((const void *)(v8 + 9), "avx512.mask.sqrt.p", 0x12u) )
                goto LABEL_42;
              goto LABEL_92;
            }
LABEL_865:
            if ( !memcmp((const void *)(v8 + 9), "avx.sqrt.p", 0xAu) )
              goto LABEL_42;
            if ( (unsigned __int64)s1a <= 0xA )
            {
              if ( !memcmp((const void *)(v8 + 9), "sse.sqrt.p", 0xAu)
                || !memcmp((const void *)(v8 + 9), "sse.add.ss", 0xAu)
                || !memcmp((const void *)(v8 + 9), "sse.sub.ss", 0xAu)
                || !memcmp((const void *)(v8 + 9), "sse.mul.ss", 0xAu)
                || !memcmp((const void *)(v8 + 9), "sse.div.ss", 0xAu) )
              {
                goto LABEL_42;
              }
              goto LABEL_107;
            }
LABEL_93:
            if ( !memcmp((const void *)(v8 + 9), "sse2.sqrt.p", 0xBu)
              || !memcmp((const void *)(v8 + 9), "sse.sqrt.p", 0xAu) )
            {
              goto LABEL_42;
            }
            if ( (unsigned __int64)s1a > 0x15 )
            {
              if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pbroadcast", 0x16u) )
                goto LABEL_42;
LABEL_97:
              if ( !memcmp((const void *)(v8 + 9), "sse2.pcmpeq.", 0xCu)
                || !memcmp((const void *)(v8 + 9), "sse2.pcmpgt.", 0xCu)
                || !memcmp((const void *)(v8 + 9), "avx2.pcmpeq.", 0xCu)
                || !memcmp((const void *)(v8 + 9), "avx2.pcmpgt.", 0xCu) )
              {
                goto LABEL_42;
              }
              if ( (unsigned __int64)s1a > 0x12 )
              {
                if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pcmpeq.", 0x13u)
                  || !memcmp((const void *)(v8 + 9), "avx512.mask.pcmpgt.", 0x13u)
                  || !memcmp((const void *)(v8 + 9), "avx.vperm2f128.", 0xFu) )
                {
                  goto LABEL_42;
                }
LABEL_105:
                if ( (unsigned __int64)s1a <= 0x13 )
                {
                  if ( (unsigned __int64)s1a <= 8 )
                    goto LABEL_803;
                }
                else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pshuf.b.", 0x14u) )
                {
                  goto LABEL_42;
                }
LABEL_107:
                if ( !memcmp((const void *)(v8 + 9), "avx2.pmax", 9u)
                  || !memcmp((const void *)(v8 + 9), "avx2.pmin", 9u) )
                {
                  goto LABEL_42;
                }
                if ( (unsigned __int64)s1a > 0xF )
                {
                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmax", 0x10u)
                    || !memcmp((const void *)(v8 + 9), "avx512.mask.pmin", 0x10u) )
                  {
                    goto LABEL_42;
                  }
                  goto LABEL_112;
                }
                if ( (unsigned __int64)s1a > 0xE )
                {
LABEL_112:
                  if ( !memcmp((const void *)(v8 + 9), "avx2.vbroadcast", 0xFu)
                    || !memcmp((const void *)(v8 + 9), "avx2.pbroadcast", 0xFu) )
                  {
                    goto LABEL_42;
                  }
                  goto LABEL_114;
                }
LABEL_803:
                if ( (unsigned __int64)s1a <= 0xB )
                {
                  if ( (unsigned __int64)s1a <= 9 )
                    goto LABEL_695;
LABEL_115:
                  if ( !memcmp((const void *)(v8 + 9), "sse2.pshuf", 0xAu) )
                    goto LABEL_42;
                  if ( (unsigned __int64)s1a > 0x10 )
                  {
                    if ( !memcmp((const void *)(v8 + 9), "avx512.pbroadcast", 0x11u) )
                      goto LABEL_42;
                    if ( (unsigned __int64)s1a > 0x16 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.broadcast.s", 0x17u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.movddup", 0x13u) )
                      {
                        goto LABEL_42;
                      }
                      goto LABEL_121;
                    }
                    if ( (unsigned __int64)s1a > 0x12 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.movddup", 0x13u) )
                        goto LABEL_42;
                      if ( s1a == (void *)19 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.p", 0x12u) )
                          goto LABEL_42;
LABEL_131:
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.punpckl", 0x13u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.punpckh", 0x13u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.unpckl.", 0x13u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.unpckh.", 0x13u) )
                        {
                          goto LABEL_42;
                        }
                        goto LABEL_135;
                      }
LABEL_121:
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.movshdup", 0x14u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.movsldup", 0x14u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.pshuf.d.", 0x14u) )
                      {
                        goto LABEL_42;
                      }
                      if ( (unsigned __int64)s1a <= 0x14 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.p", 0x12u) )
                          goto LABEL_42;
                      }
                      else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pshufl.w.", 0x15u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.pshufh.w.", 0x15u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.p", 0x12u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.vpermil.p", 0x15u) )
                      {
                        goto LABEL_42;
                      }
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.perm.df.", 0x14u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.perm.di.", 0x14u) )
                      {
                        goto LABEL_42;
                      }
                      goto LABEL_131;
                    }
                  }
                  if ( s1a == (void *)18 )
                  {
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.shuf.p", 0x12u) )
                      goto LABEL_42;
                  }
                  else if ( (unsigned __int64)s1a <= 0x10 )
                  {
                    goto LABEL_776;
                  }
LABEL_135:
                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pand.", 0x11u) )
                    goto LABEL_42;
                  if ( (unsigned __int64)s1a > 0x11 )
                  {
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pandn.", 0x12u)
                      || !memcmp((const void *)(v8 + 9), "avx512.mask.por.", 0x10u) )
                    {
                      goto LABEL_42;
                    }
                    goto LABEL_139;
                  }
LABEL_776:
                  if ( (unsigned __int64)s1a <= 0xF )
                  {
                    if ( s1a == (void *)15 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.or.", 0xFu) )
                        goto LABEL_42;
                      goto LABEL_200;
                    }
                    goto LABEL_766;
                  }
                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.por.", 0x10u) )
                    goto LABEL_42;
                  if ( s1a == (void *)16 )
                  {
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.and.", 0x10u) )
                      goto LABEL_42;
LABEL_142:
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.or.", 0xFu)
                      || !memcmp((const void *)(v8 + 9), "avx512.mask.xor.", 0x10u) )
                    {
                      goto LABEL_42;
                    }
                    if ( (unsigned __int64)s1a > 0x10 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.padd.", 0x11u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.psub.", 0x11u) )
                      {
                        goto LABEL_42;
                      }
                      if ( s1a == (void *)17 )
                        goto LABEL_173;
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmull.", 0x12u) )
                        goto LABEL_42;
                      if ( (unsigned __int64)s1a > 0x14 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.cvtdq2pd.", 0x15u) )
                          goto LABEL_42;
                        if ( s1a != (void *)21 )
                        {
                          if ( !memcmp((const void *)(v8 + 9), "avx512.mask.cvtudq2pd.", 0x16u) )
                            goto LABEL_42;
                          if ( s1a == (void *)25 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.cvtudq2ps.128", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtudq2ps.256", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtuqq2pd.128", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtuqq2pd.256", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvttpd2dq.256", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvttps2dq.128", 0x19u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvttps2dq.256", 0x19u) )
                            {
                              goto LABEL_42;
                            }
                          }
                          else if ( s1a == (void *)24 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.cvtqq2pd.128", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtqq2pd.256", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtdq2ps.128", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtdq2ps.256", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtpd2dq.256", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtpd2ps.256", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtps2pd.128", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cvtps2pd.256", 0x18u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.permvar.", 0x14u) )
                            {
                              goto LABEL_42;
                            }
LABEL_159:
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmul.dq.", 0x14u) )
                              goto LABEL_42;
                            if ( (unsigned __int64)s1a <= 0x14 )
                            {
                              if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmulh.w.", 0x14u) )
                                goto LABEL_42;
                            }
                            else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmulu.dq.", 0x15u)
                                   || (unsigned __int64)s1a > 0x16
                                   && !memcmp((const void *)(v8 + 9), "avx512.mask.pmul.hr.sw.", 0x17u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.pmulh.w.", 0x14u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.pmulhu.w.", 0x15u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.pmaddw.d.", 0x15u)
                                   || (unsigned __int64)s1a > 0x16
                                   && !memcmp((const void *)(v8 + 9), "avx512.mask.pmaddubs.w.", 0x17u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.packsswb.", 0x15u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.packssdw.", 0x15u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.packuswb.", 0x15u)
                                   || !memcmp((const void *)(v8 + 9), "avx512.mask.packusdw.", 0x15u) )
                            {
                              goto LABEL_42;
                            }
LABEL_173:
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.cmp.b", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cmp.d", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cmp.q", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cmp.w", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.cmp.p", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.ucmp.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.cvtb2mask.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.cvtw2mask.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.cvtd2mask.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.cvtq2mask.", 0x11u) )
                            {
                              goto LABEL_42;
                            }
                            if ( (unsigned __int64)s1a <= 0x16 )
                            {
                              if ( (unsigned __int64)s1a <= 0x11 )
                              {
LABEL_194:
                                if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pslli", 0x11u)
                                  || !memcmp((const void *)(v8 + 9), "avx512.mask.psrai", 0x11u)
                                  || !memcmp((const void *)(v8 + 9), "avx512.mask.psrli", 0x11u)
                                  || !memcmp((const void *)(v8 + 9), "avx512.mask.psllv", 0x11u)
                                  || !memcmp((const void *)(v8 + 9), "avx512.mask.psrav", 0x11u)
                                  || !memcmp((const void *)(v8 + 9), "avx512.mask.psrlv", 0x11u) )
                                {
                                  goto LABEL_42;
                                }
                                goto LABEL_200;
                              }
                            }
                            else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpermilvar.", 0x17u) )
                            {
                              goto LABEL_42;
                            }
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.psll.d", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psll.q", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psll.w", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psra.d", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psra.q", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psra.w", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psrl.d", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psrl.q", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.psrl.w", 0x12u) )
                            {
                              goto LABEL_42;
                            }
                            goto LABEL_194;
                          }
                        }
                        goto LABEL_155;
                      }
                    }
LABEL_766:
                    if ( s1a == (void *)16 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.cvtusi2sd", 0x10u) )
                        goto LABEL_42;
                      goto LABEL_200;
                    }
                    if ( (unsigned __int64)s1a <= 0x13 )
                    {
                      if ( s1a == (void *)13 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "sse2.pmulu.dq", 0xDu)
                          || !memcmp((const void *)(v8 + 9), "avx2.pmulu.dq", 0xDu) )
                        {
                          goto LABEL_42;
                        }
                        goto LABEL_200;
                      }
                      if ( s1a == (void *)12 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "sse41.pmuldq", 0xCu)
                          || !memcmp((const void *)(v8 + 9), "avx2.pmul.dq", 0xCu) )
                        {
                          goto LABEL_42;
                        }
                        goto LABEL_200;
                      }
                      goto LABEL_156;
                    }
LABEL_155:
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.permvar.", 0x14u) )
                      goto LABEL_42;
LABEL_156:
                    if ( s1a == (void *)19 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.pmulu.dq.512", 0x13u) )
                        goto LABEL_42;
                      goto LABEL_173;
                    }
                    if ( s1a == (void *)18 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.pmul.dq.512", 0x12u) )
                        goto LABEL_42;
                      goto LABEL_173;
                    }
                    if ( (unsigned __int64)s1a > 0x13 )
                      goto LABEL_159;
                    if ( (unsigned __int64)s1a <= 0xB )
                    {
                      if ( s1a != (void *)11 )
                        goto LABEL_254;
                      goto LABEL_202;
                    }
LABEL_200:
                    if ( !memcmp((const void *)(v8 + 9), "sse41.pmovsx", 0xCu)
                      || !memcmp((const void *)(v8 + 9), "sse41.pmovzx", 0xCu) )
                    {
                      goto LABEL_42;
                    }
LABEL_202:
                    if ( !memcmp((const void *)(v8 + 9), "avx2.pmovsx", 0xBu)
                      || !memcmp((const void *)(v8 + 9), "avx2.pmovzx", 0xBu) )
                    {
                      goto LABEL_42;
                    }
                    if ( (unsigned __int64)s1a <= 0x11 )
                      goto LABEL_717;
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pmovsx", 0x12u)
                      || !memcmp((const void *)(v8 + 9), "avx512.mask.pmovzx", 0x12u)
                      || !memcmp((const void *)(v8 + 9), "avx512.mask.lzcnt.", 0x12u) )
                    {
                      goto LABEL_42;
                    }
                    if ( (unsigned __int64)s1a > 0x14 )
                    {
                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pternlog.", 0x15u)
                        || s1a != (void *)21 && !memcmp((const void *)(v8 + 9), "avx512.maskz.pternlog.", 0x16u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.vpmadd52", 0x14u)
                        || !memcmp((const void *)(v8 + 9), "avx512.maskz.vpmadd52", 0x15u) )
                      {
                        goto LABEL_42;
                      }
                      if ( (unsigned __int64)s1a <= 0x16 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpbusd.", 0x15u) )
                          goto LABEL_42;
                        if ( s1a != (void *)22 )
                        {
                          if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpwssd.", 0x15u) )
                            goto LABEL_42;
LABEL_229:
                          if ( !memcmp((const void *)(v8 + 9), "avx512.mask.dbpsadbw.", 0x15u) )
                            goto LABEL_42;
                          goto LABEL_230;
                        }
                      }
                      else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpermi2var.", 0x17u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.vpermt2var.", 0x17u)
                             || s1a != (void *)23 && !memcmp((const void *)(v8 + 9), "avx512.maskz.vpermt2var.", 0x18u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpbusd.", 0x15u) )
                      {
                        goto LABEL_42;
                      }
                      if ( !memcmp((const void *)(v8 + 9), "avx512.maskz.vpdpbusd.", 0x16u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpbusds.", 0x16u) )
                      {
                        goto LABEL_42;
                      }
                      if ( (unsigned __int64)s1a <= 0x16 )
                      {
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpwssd.", 0x15u) )
                          goto LABEL_42;
                      }
                      else if ( !memcmp((const void *)(v8 + 9), "avx512.maskz.vpdpbusds.", 0x17u)
                             || !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpwssd.", 0x15u) )
                      {
                        goto LABEL_42;
                      }
                      if ( !memcmp((const void *)(v8 + 9), "avx512.maskz.vpdpwssd.", 0x16u)
                        || !memcmp((const void *)(v8 + 9), "avx512.mask.vpdpwssds.", 0x16u)
                        || (unsigned __int64)s1a > 0x16
                        && !memcmp((const void *)(v8 + 9), "avx512.maskz.vpdpwssds.", 0x17u) )
                      {
                        goto LABEL_42;
                      }
                      goto LABEL_229;
                    }
                    if ( (unsigned __int64)s1a <= 0x13 )
                    {
LABEL_717:
                      if ( (unsigned __int64)s1a <= 0x12 )
                      {
                        if ( (unsigned __int64)s1a <= 0x10 )
                        {
                          if ( s1a == (void *)12 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "sse.cvtsi2ss", 0xCu) )
                              goto LABEL_42;
                            goto LABEL_253;
                          }
                          if ( s1a == (void *)14 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "sse.cvtsi642ss", 0xEu) )
                              goto LABEL_42;
                            goto LABEL_253;
                          }
                          if ( s1a == (void *)13 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "sse2.cvtsi2sd", 0xDu)
                              || !memcmp((const void *)(v8 + 9), "sse2.cvtss2sd", 0xDu)
                              || !memcmp((const void *)(v8 + 9), "sse2.cvtdq2pd", 0xDu)
                              || !memcmp((const void *)(v8 + 9), "sse2.cvtdq2ps", 0xDu)
                              || !memcmp((const void *)(v8 + 9), "sse2.cvtps2pd", 0xDu) )
                            {
                              goto LABEL_42;
                            }
                            goto LABEL_253;
                          }
                          if ( s1a == (void *)15 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "sse2.cvtsi642sd", 0xFu) )
                              goto LABEL_42;
                            goto LABEL_253;
                          }
LABEL_244:
                          if ( s1a == (void *)18 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "avx.cvt.ps2.pd.256", 0x12u)
                              || !memcmp((const void *)(v8 + 9), "avx.vinsertf128.", 0x10u) )
                            {
                              goto LABEL_42;
                            }
                          }
                          else
                          {
                            if ( (unsigned __int64)s1a > 0xF )
                            {
                              if ( !memcmp((const void *)(v8 + 9), "avx.vinsertf128.", 0x10u) )
                                goto LABEL_42;
                              if ( s1a == (void *)16 )
                              {
                                if ( !memcmp((const void *)(v8 + 9), "avx2.vinserti128", 0x10u) )
                                  goto LABEL_42;
                                goto LABEL_253;
                              }
                            }
                            if ( (unsigned __int64)s1a <= 0x11 )
                            {
LABEL_697:
                              if ( (unsigned __int64)s1a > 0xB )
                                goto LABEL_253;
LABEL_254:
                              if ( !memcmp((const void *)(v8 + 9), "avx.movnt.", 0xAu) )
                                goto LABEL_42;
                              if ( (unsigned __int64)s1a > 0xE )
                              {
                                if ( !memcmp((const void *)(v8 + 9), "avx512.storent.", 0xFu) )
                                  goto LABEL_42;
                                if ( s1a != (void *)15 )
                                  goto LABEL_258;
                                if ( !memcmp((const void *)(v8 + 9), "avx512.movntdqa", 0xFu) )
                                  goto LABEL_42;
LABEL_691:
                                if ( !memcmp((const void *)(v8 + 9), "sse.storeu.", 0xBu) )
                                  goto LABEL_42;
                                goto LABEL_260;
                              }
                              if ( s1a == (void *)14 )
                              {
                                if ( !memcmp((const void *)(v8 + 9), "sse41.movntdqa", 0xEu)
                                  || !memcmp((const void *)(v8 + 9), "sse2.storel.dq", 0xEu) )
                                {
                                  goto LABEL_42;
                                }
                                goto LABEL_691;
                              }
                              if ( s1a == (void *)13 )
                              {
                                if ( !memcmp((const void *)(v8 + 9), "avx2.movntdqa", 0xDu) )
                                  goto LABEL_42;
                                goto LABEL_691;
                              }
LABEL_695:
                              if ( (unsigned __int64)s1a <= 0xA )
                                goto LABEL_686;
LABEL_258:
                              if ( !memcmp((const void *)(v8 + 9), "sse.storeu.", 0xBu) )
                                goto LABEL_42;
                              if ( (unsigned __int64)s1a <= 0xB )
                              {
LABEL_261:
                                if ( !memcmp((const void *)(v8 + 9), "avx.storeu.", 0xBu) )
                                  goto LABEL_42;
                                if ( (unsigned __int64)s1a > 0x12 )
                                {
                                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.storeu.", 0x13u)
                                    || !memcmp((const void *)(v8 + 9), "avx512.mask.store.p", 0x13u)
                                    || s1a != (void *)19
                                    && (!memcmp((const void *)(v8 + 9), "avx512.mask.store.b.", 0x14u)
                                     || !memcmp((const void *)(v8 + 9), "avx512.mask.store.w.", 0x14u)
                                     || !memcmp((const void *)(v8 + 9), "avx512.mask.store.d.", 0x14u)
                                     || !memcmp((const void *)(v8 + 9), "avx512.mask.store.q.", 0x14u)
                                     || s1a == (void *)20
                                     && !memcmp((const void *)(v8 + 9), "avx512.mask.store.ss", 0x14u)) )
                                  {
                                    goto LABEL_42;
                                  }
LABEL_271:
                                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.loadu.", 0x12u)
                                    || !memcmp((const void *)(v8 + 9), "avx512.mask.load.", 0x11u) )
                                  {
                                    goto LABEL_42;
                                  }
                                  if ( (unsigned __int64)s1a > 0x17 )
                                  {
                                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.expand.load.", 0x18u)
                                      || (unsigned __int64)s1a > 0x1A
                                      && !memcmp((const void *)(v8 + 9), "avx512.mask.compress.store.", 0x1Bu)
                                      || !memcmp((const void *)(v8 + 9), "avx.vbroadcast.s", 0x10u) )
                                    {
                                      goto LABEL_42;
                                    }
                                    goto LABEL_278;
                                  }
                                  goto LABEL_674;
                                }
                                if ( (unsigned __int64)s1a > 0x11 )
                                  goto LABEL_271;
LABEL_686:
                                if ( s1a == (void *)17 )
                                {
                                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.load.", 0x11u) )
                                    goto LABEL_42;
                                  goto LABEL_681;
                                }
LABEL_674:
                                if ( s1a != (void *)16 )
                                {
                                  if ( (unsigned __int64)s1a > 0xF )
                                  {
                                    if ( !memcmp((const void *)(v8 + 9), "avx.vbroadcast.s", 0x10u) )
                                      goto LABEL_42;
                                    if ( (unsigned __int64)s1a > 0x12 )
                                    {
LABEL_278:
                                      if ( !memcmp((const void *)(v8 + 9), "avx512.vbroadcast.s", 0x13u)
                                        || (unsigned __int64)s1a > 0x13
                                        && !memcmp((const void *)(v8 + 9), "avx512.mask.palignr.", 0x14u)
                                        || !memcmp((const void *)(v8 + 9), "avx512.mask.valign.", 0x13u) )
                                      {
                                        goto LABEL_42;
                                      }
                                      goto LABEL_282;
                                    }
                                  }
                                  if ( (unsigned __int64)s1a <= 0xB )
                                  {
                                    if ( s1a == (void *)11 )
                                    {
                                      if ( !memcmp((const void *)(v8 + 9), "avx.blend.p", 0xBu) )
                                        goto LABEL_42;
                                    }
                                    else
                                    {
                                      if ( s1a != (void *)10 )
                                      {
LABEL_606:
                                        if ( s1a != (void *)14 )
                                        {
                                          if ( (unsigned __int64)s1a <= 0xE )
                                          {
                                            if ( (unsigned __int64)s1a <= 8 )
                                              goto LABEL_609;
LABEL_301:
                                            if ( memcmp((const void *)(v8 + 9), "xop.vpcom", 9u) )
                                              goto LABEL_303;
                                            goto LABEL_302;
                                          }
LABEL_300:
                                          if ( !memcmp((const void *)(v8 + 9), "avx512.cvtmask2", 0xFu) )
                                            goto LABEL_42;
                                          goto LABEL_301;
                                        }
                                        if ( !memcmp((const void *)(v8 + 9), "xop.vpcmov.256", 0xEu) )
                                          goto LABEL_42;
                                        if ( memcmp((const void *)(v8 + 9), "xop.vpcom", 9u) )
                                        {
LABEL_304:
                                          if ( !memcmp((const void *)(v8 + 9), "avx512.ptestm", 0xDu)
                                            || (unsigned __int64)s1a > 0xD
                                            && !memcmp((const void *)(v8 + 9), "avx512.ptestnm", 0xEu) )
                                          {
                                            goto LABEL_42;
                                          }
LABEL_307:
                                          if ( !memcmp((const void *)(v8 + 9), "sse2.pavg", 9u)
                                            || !memcmp((const void *)(v8 + 9), "avx2.pavg", 9u) )
                                          {
                                            goto LABEL_42;
                                          }
                                          if ( (unsigned __int64)s1a > 0xF )
                                          {
                                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pavg", 0x10u) )
                                              goto LABEL_42;
                                            if ( !memcmp((const void *)(v8 + 9), "sse41.ptest", 0xBu) )
                                            {
LABEL_312:
                                              if ( s1a == (void *)12 )
                                              {
                                                if ( !memcmp((const void *)(v8 + 9), "avx2.mpsadbw", 0xCu) )
                                                {
                                                  result = sub_1564350(a1, 0x1919u, a2);
                                                  goto LABEL_315;
                                                }
LABEL_614:
                                                if ( !memcmp((const void *)(v8 + 9), "xop.vfrcz.ss", 0xCu) )
                                                {
                                                  if ( *(_QWORD *)(a1 + 96) == 2 )
                                                  {
                                                    LOWORD(dest[0]) = 773;
                                                    v136 = (const char **)sub_1649960(a1);
                                                    v139 = &v136;
                                                    v137 = v108;
                                                    n = (size_t)".old";
                                                    sub_164B780(a1, &v139);
                                                    *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 7502, 0, 0);
                                                    return 1;
                                                  }
                                                }
                                                else if ( !memcmp((const void *)(v8 + 9), "xop.vfrcz.sd", 0xCu)
                                                       && *(_QWORD *)(a1 + 96) == 2 )
                                                {
                                                  LOWORD(dest[0]) = 773;
                                                  v136 = (const char **)sub_1649960(a1);
                                                  v139 = &v136;
                                                  v137 = v107;
                                                  n = (size_t)".old";
                                                  sub_164B780(a1, &v139);
                                                  *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), 7501, 0, 0);
                                                  return 1;
                                                }
                                                if ( !memcmp((const void *)(v8 + 9), "xop.vpermil2", 0xCu) )
                                                {
                                                  v100 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 24LL);
                                                  v101 = v100;
                                                  if ( *(_BYTE *)(v100 + 8) == 16 )
                                                    v101 = **(_QWORD **)(v100 + 16);
                                                  if ( (unsigned __int8)(*(_BYTE *)(v101 + 8) - 1) <= 5u )
                                                  {
                                                    LOWORD(dest[0]) = 773;
                                                    v136 = (const char **)sub_1649960(a1);
                                                    v139 = &v136;
                                                    v137 = v102;
                                                    n = (size_t)".old";
                                                    sub_164B780(a1, &v139);
                                                    v103 = sub_1643030(v100);
                                                    v104 = sub_16431D0(v100);
                                                    if ( v104 != 64 || (v105 = 7511, v103 != 128) )
                                                    {
                                                      if ( v104 != 32 || (v105 = 7513, v103 != 128) )
                                                      {
                                                        if ( v103 != 256 || (v105 = 7512, v104 != 64) )
                                                          v105 = 7514;
                                                      }
                                                    }
                                                    *a2 = (void *)sub_15E26F0(*(_QWORD *)(a1 + 40), v105, 0, 0);
                                                    return 1;
                                                  }
                                                }
                                                goto LABEL_11;
                                              }
LABEL_613:
                                              if ( (unsigned __int64)s1a <= 0xB )
                                                goto LABEL_11;
                                              goto LABEL_614;
                                            }
                                            goto LABEL_628;
                                          }
LABEL_609:
                                          if ( (unsigned __int64)s1a <= 0xA )
                                          {
                                            if ( s1a == (void *)10 )
                                            {
                                              if ( *(_QWORD *)(v8 + 9) == 0x70642E3134657373LL
                                                && *(_WORD *)(v8 + 17) == 25712 )
                                              {
                                                result = sub_1564350(a1, 0x1CE7u, a2);
                                              }
                                              else
                                              {
                                                if ( *(_QWORD *)(v8 + 9) != 0x70642E3134657373LL
                                                  || *(_WORD *)(v8 + 17) != 29552 )
                                                {
                                                  goto LABEL_11;
                                                }
                                                result = sub_1564350(a1, 0x1CE8u, a2);
                                              }
                                              goto LABEL_315;
                                            }
                                          }
                                          else
                                          {
                                            if ( *(_QWORD *)(v8 + 9) == 0x74702E3134657373LL
                                              && *(_WORD *)(v8 + 17) == 29541
                                              && *(_BYTE *)(v8 + 19) == 116 )
                                            {
                                              if ( v6 == 21 )
                                              {
                                                v106 = *(_BYTE *)(v8 + 20);
                                                if ( v106 == 99 )
                                                {
                                                  result = sub_15643F0(a1, 0x1CEEu, a2);
                                                }
                                                else
                                                {
                                                  if ( v106 != 122 )
                                                    goto LABEL_312;
                                                  result = sub_15643F0(a1, 0x1CF0u, a2);
                                                }
LABEL_315:
                                                if ( (_BYTE)result )
                                                  return result;
                                                goto LABEL_11;
                                              }
                                              if ( v6 == 23 )
                                              {
                                                if ( !memcmp((const void *)(v8 + 20), "nzc", 3u) )
                                                {
                                                  result = sub_15643F0(a1, 0x1CEFu, a2);
                                                  goto LABEL_315;
                                                }
                                                goto LABEL_612;
                                              }
                                            }
                                            if ( s1a == (void *)14 )
                                            {
LABEL_612:
                                              if ( *(_QWORD *)(v8 + 9) != 0x6E692E3134657373LL
                                                || *(_DWORD *)(v8 + 17) != 1953654131
                                                || *(_WORD *)(v8 + 21) != 29552 )
                                              {
                                                goto LABEL_613;
                                              }
                                              result = sub_1564350(a1, 0x1CE9u, a2);
                                              goto LABEL_315;
                                            }
                                          }
LABEL_628:
                                          if ( s1a != (void *)13 )
                                            goto LABEL_312;
                                          if ( !memcmp((const void *)(v8 + 9), "sse41.mpsadbw", 0xDu) )
                                          {
                                            result = sub_1564350(a1, 0x1CEAu, a2);
                                          }
                                          else
                                          {
                                            if ( memcmp((const void *)(v8 + 9), "avx.dp.ps.256", 0xDu) )
                                              goto LABEL_614;
                                            result = sub_1564350(a1, 0x18D4u, a2);
                                          }
                                          goto LABEL_315;
                                        }
LABEL_302:
                                        if ( *(_QWORD *)(a1 + 96) == 2 )
                                          goto LABEL_42;
LABEL_303:
                                        if ( (unsigned __int64)s1a <= 0xC )
                                          goto LABEL_307;
                                        goto LABEL_304;
                                      }
                                      if ( !memcmp((const void *)(v8 + 9), "xop.vpcmov", 0xAu) )
                                        goto LABEL_42;
                                    }
LABEL_662:
                                    if ( memcmp((const void *)(v8 + 9), "xop.vpcom", 9u) )
                                      goto LABEL_307;
                                    goto LABEL_302;
                                  }
LABEL_282:
                                  if ( !memcmp((const void *)(v8 + 9), "sse2.psll.dq", 0xCu)
                                    || !memcmp((const void *)(v8 + 9), "sse2.psrl.dq", 0xCu)
                                    || !memcmp((const void *)(v8 + 9), "avx2.psll.dq", 0xCu)
                                    || !memcmp((const void *)(v8 + 9), "avx2.psrl.dq", 0xCu) )
                                  {
                                    goto LABEL_42;
                                  }
                                  if ( (unsigned __int64)s1a <= 0xD )
                                  {
                                    if ( s1a == (void *)13 && !memcmp((const void *)(v8 + 9), "sse41.pblendw", 0xDu) )
                                      goto LABEL_42;
                                  }
                                  else if ( !memcmp((const void *)(v8 + 9), "avx512.psll.dq", 0xEu)
                                         || !memcmp((const void *)(v8 + 9), "avx512.psrl.dq", 0xEu) )
                                  {
                                    goto LABEL_42;
                                  }
                                  if ( !memcmp((const void *)(v8 + 9), "sse41.blendp", 0xCu)
                                    || !memcmp((const void *)(v8 + 9), "avx.blend.p", 0xBu) )
                                  {
                                    goto LABEL_42;
                                  }
                                  if ( s1a != (void *)12 )
                                  {
                                    if ( !memcmp((const void *)(v8 + 9), "avx2.pblendd.", 0xDu) )
                                      goto LABEL_42;
                                    if ( (unsigned __int64)s1a > 0x11 )
                                    {
                                      if ( !memcmp((const void *)(v8 + 9), "avx.vbroadcastf128", 0x12u) )
                                        goto LABEL_42;
                                      if ( s1a == (void *)19 )
                                      {
                                        if ( !memcmp((const void *)(v8 + 9), "avx2.vbroadcasti128", 0x13u) )
                                          goto LABEL_42;
                                      }
                                      else if ( (unsigned __int64)s1a > 0x15
                                             && (!memcmp((const void *)(v8 + 9), "avx512.mask.broadcastf", 0x16u)
                                              || !memcmp((const void *)(v8 + 9), "avx512.mask.broadcasti", 0x16u)) )
                                      {
                                        goto LABEL_42;
                                      }
                                      if ( !memcmp((const void *)(v8 + 9), "avx512.mask.move.s", 0x12u) )
                                        goto LABEL_42;
                                      goto LABEL_300;
                                    }
                                    goto LABEL_606;
                                  }
                                  if ( !memcmp((const void *)(v8 + 9), "avx2.pblendw", 0xCu) )
                                    goto LABEL_42;
                                  goto LABEL_662;
                                }
                                if ( !memcmp((const void *)(v8 + 9), "sse42.crc32.64.8", 0x10u) )
                                  goto LABEL_42;
LABEL_681:
                                if ( !memcmp((const void *)(v8 + 9), "avx.vbroadcast.s", 0x10u) )
                                  goto LABEL_42;
                                goto LABEL_282;
                              }
LABEL_260:
                              if ( !memcmp((const void *)(v8 + 9), "sse2.storeu.", 0xCu) )
                                goto LABEL_42;
                              goto LABEL_261;
                            }
                          }
                          if ( !memcmp((const void *)(v8 + 9), "avx512.mask.insert", 0x12u)
                            || !memcmp((const void *)(v8 + 9), "avx.vextractf128.", 0x11u) )
                          {
                            goto LABEL_42;
                          }
                          if ( (unsigned __int64)s1a > 0x13 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vextract", 0x14u) )
                              goto LABEL_42;
LABEL_253:
                            if ( !memcmp((const void *)(v8 + 9), "sse4a.movnt.", 0xCu) )
                              goto LABEL_42;
                            goto LABEL_254;
                          }
                          goto LABEL_697;
                        }
LABEL_232:
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.add.p", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.sub.p", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.mul.p", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.div.p", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.max.p", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.min.p", 0x11u) )
                        {
                          goto LABEL_42;
                        }
                        if ( (unsigned __int64)s1a <= 0x14 )
                        {
                          if ( (unsigned __int64)s1a <= 0x11 )
                          {
                            if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pror.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx512.mask.prol.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx.cvtdq2.pd.256", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx.cvtdq2.ps.256", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx.vinsertf128.", 0x10u)
                              || !memcmp((const void *)(v8 + 9), "avx.vextractf128.", 0x11u)
                              || !memcmp((const void *)(v8 + 9), "avx2.vextracti128", 0x11u) )
                            {
                              goto LABEL_42;
                            }
                            goto LABEL_253;
                          }
                        }
                        else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.fpclass.p", 0x15u) )
                        {
                          goto LABEL_42;
                        }
                        if ( !memcmp((const void *)(v8 + 9), "avx512.mask.prorv.", 0x12u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.pror.", 0x11u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.prolv.", 0x12u)
                          || !memcmp((const void *)(v8 + 9), "avx512.mask.prol.", 0x11u) )
                        {
                          goto LABEL_42;
                        }
                        goto LABEL_244;
                      }
                    }
                    else if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpmadd52", 0x14u) )
                    {
                      goto LABEL_42;
                    }
LABEL_230:
                    if ( !memcmp((const void *)(v8 + 9), "avx512.mask.vpshld.", 0x13u)
                      || !memcmp((const void *)(v8 + 9), "avx512.mask.vpshrd.", 0x13u) )
                    {
                      goto LABEL_42;
                    }
                    goto LABEL_232;
                  }
LABEL_139:
                  if ( !memcmp((const void *)(v8 + 9), "avx512.mask.pxor.", 0x11u)
                    || !memcmp((const void *)(v8 + 9), "avx512.mask.and.", 0x10u)
                    || !memcmp((const void *)(v8 + 9), "avx512.mask.andn.", 0x11u) )
                  {
                    goto LABEL_42;
                  }
                  goto LABEL_142;
                }
LABEL_114:
                if ( !memcmp((const void *)(v8 + 9), "avx.vpermil.", 0xCu) )
                  goto LABEL_42;
                goto LABEL_115;
              }
              if ( (unsigned __int64)s1a > 0xE )
              {
                if ( !memcmp((const void *)(v8 + 9), "avx.vperm2f128.", 0xFu) )
                  goto LABEL_42;
                if ( s1a == (void *)15 )
                {
                  if ( !memcmp((const void *)(v8 + 9), "avx2.vperm2i128", 0xFu) )
                    goto LABEL_42;
                  goto LABEL_107;
                }
              }
LABEL_823:
              if ( s1a == (void *)11 )
              {
                if ( !memcmp((const void *)(v8 + 9), "sse2.add.sd", 0xBu)
                  || !memcmp((const void *)(v8 + 9), "sse2.sub.sd", 0xBu)
                  || !memcmp((const void *)(v8 + 9), "sse2.mul.sd", 0xBu)
                  || !memcmp((const void *)(v8 + 9), "sse2.div.sd", 0xBu) )
                {
                  goto LABEL_42;
                }
                goto LABEL_107;
              }
              if ( s1a == (void *)12 )
              {
                if ( !memcmp((const void *)(v8 + 9), "sse41.pmaxsb", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse2.pmaxs.w", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pmaxsd", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse2.pmaxu.b", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pmaxuw", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pmaxud", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pminsb", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse2.pmins.w", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pminsd", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse2.pminu.b", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pminuw", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "sse41.pminud", 0xCu)
                  || !memcmp((const void *)(v8 + 9), "avx512.kor.w", 0xCu) )
                {
                  goto LABEL_42;
                }
                goto LABEL_107;
              }
              if ( s1a == (void *)13 )
              {
                if ( !memcmp((const void *)(v8 + 9), "avx512.kand.w", 0xDu)
                  || !memcmp((const void *)(v8 + 9), "avx512.knot.w", 0xDu)
                  || !memcmp((const void *)(v8 + 9), "avx512.kxor.w", 0xDu) )
                {
                  goto LABEL_42;
                }
                goto LABEL_107;
              }
              if ( s1a == (void *)14 )
              {
                if ( !memcmp((const void *)(v8 + 9), "avx512.kandn.w", 0xEu)
                  || !memcmp((const void *)(v8 + 9), "avx512.kxnor.w", 0xEu) )
                {
                  goto LABEL_42;
                }
                goto LABEL_107;
              }
              if ( s1a == (void *)17 )
              {
                if ( !memcmp((const void *)(v8 + 9), "avx512.kortestc.w", 0x11u)
                  || !memcmp((const void *)(v8 + 9), "avx512.kortestz.w", 0x11u) )
                {
                  goto LABEL_42;
                }
                goto LABEL_107;
              }
              goto LABEL_105;
            }
LABEL_831:
            if ( (unsigned __int64)s1a <= 0xB )
              goto LABEL_823;
            goto LABEL_97;
          }
          goto LABEL_598;
        }
        if ( *(_QWORD *)(v8 + 9) == 0x6D66762E34616D66LL
          && *(_DWORD *)(v8 + 17) == 778331233
          && *(_BYTE *)(v8 + 21) == 115 )
        {
          goto LABEL_42;
        }
        goto LABEL_466;
      default:
        goto LABEL_11;
    }
  }
  return result;
}
