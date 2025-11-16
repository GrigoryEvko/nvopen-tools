// Function: sub_358B4D0
// Address: 0x358b4d0
//
unsigned __int64 *__fastcall sub_358B4D0(unsigned __int64 *a1, __int64 a2, char ***a3, __int64 a4)
{
  unsigned __int64 v6; // rsi
  char **v7; // r12
  char **v8; // rbx
  __int64 v9; // r11
  bool v10; // zf
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  char *v13; // rdi
  size_t v14; // rdx
  char *v15; // rax
  char *v16; // rdi
  __int64 v17; // rax
  char *v18; // rdi
  void *v19; // r9
  size_t v20; // rdx
  char *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  char **v26; // rdx
  char *v27; // r10
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rax
  _QWORD *v30; // rdx
  unsigned int v31; // edi
  __int64 *v32; // rax
  __int64 v33; // rcx
  int v34; // r14d
  int v35; // eax
  int v36; // eax
  __int64 *v37; // r12
  __int64 *i; // rbx
  _BYTE *v39; // rsi
  __int64 v40; // rdi
  _BYTE *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // r14
  __int64 v44; // rdi
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rcx
  __int64 v47; // rax
  char **v49; // rax
  __int64 v50; // r11
  __int64 v51; // rbx
  unsigned int v52; // esi
  __int64 v53; // r8
  unsigned int v54; // r13d
  __int64 *v55; // rbx
  __int64 *v56; // r10
  __m128i *v57; // rsi
  unsigned int v58; // esi
  __int64 v59; // r13
  __int64 v60; // rdx
  unsigned int v61; // ecx
  unsigned int v62; // r14d
  unsigned int v63; // edi
  __int64 v64; // rax
  unsigned int v65; // r9d
  _QWORD *v66; // rax
  __int64 v67; // rdi
  unsigned int v68; // r9d
  _QWORD *v69; // rax
  __int64 v70; // r8
  __int64 v71; // rax
  int v72; // r8d
  _QWORD *v73; // r8
  int v74; // eax
  int v75; // eax
  int v76; // edx
  int v77; // edx
  __int64 v78; // rsi
  unsigned int v79; // r14d
  int v80; // eax
  _QWORD *v81; // rdi
  __int64 v82; // rcx
  int v83; // eax
  int v84; // edx
  int v85; // edx
  __int64 v86; // rsi
  _QWORD *v87; // r9
  unsigned int v88; // r14d
  int v89; // r8d
  __int64 v90; // rcx
  int v91; // r9d
  int v92; // r9d
  __int64 v93; // rsi
  unsigned int v94; // edx
  __int64 v95; // rcx
  _QWORD *v96; // rdi
  int v97; // r9d
  int v98; // r9d
  __int64 v99; // rsi
  unsigned int v100; // edx
  __int64 v101; // rcx
  int v102; // edx
  int v103; // ecx
  int v104; // r9d
  int v105; // r9d
  __int64 v106; // rdi
  __int64 v107; // rcx
  __int64 v108; // rsi
  int v109; // r13d
  _QWORD *v110; // r10
  int v111; // r8d
  int v112; // r8d
  __int64 v113; // rsi
  _QWORD *v114; // r9
  __int64 v115; // r13
  int v116; // r10d
  __int64 v117; // rcx
  int v118; // r9d
  _QWORD *v119; // r8
  __int64 v120; // [rsp+8h] [rbp-B8h]
  char **v122; // [rsp+10h] [rbp-B0h]
  __int64 *v124; // [rsp+20h] [rbp-A0h]
  int v125; // [rsp+20h] [rbp-A0h]
  __int64 *v126; // [rsp+20h] [rbp-A0h]
  __int64 *v127; // [rsp+20h] [rbp-A0h]
  __int64 *v128; // [rsp+20h] [rbp-A0h]
  __int64 *v129; // [rsp+20h] [rbp-A0h]
  size_t v130; // [rsp+28h] [rbp-98h]
  __int64 v131; // [rsp+28h] [rbp-98h]
  __int64 v132; // [rsp+28h] [rbp-98h]
  int v133; // [rsp+28h] [rbp-98h]
  __int64 v134; // [rsp+28h] [rbp-98h]
  __int64 v135; // [rsp+28h] [rbp-98h]
  int v136; // [rsp+28h] [rbp-98h]
  __int64 v137; // [rsp+28h] [rbp-98h]
  int v138; // [rsp+28h] [rbp-98h]
  unsigned __int64 v139; // [rsp+30h] [rbp-90h]
  size_t v140; // [rsp+30h] [rbp-90h]
  unsigned __int64 v141; // [rsp+30h] [rbp-90h]
  void *v142; // [rsp+30h] [rbp-90h]
  __int64 *v143; // [rsp+30h] [rbp-90h]
  char **v145; // [rsp+38h] [rbp-88h]
  unsigned int v146; // [rsp+38h] [rbp-88h]
  __int64 v147; // [rsp+38h] [rbp-88h]
  __int64 v148; // [rsp+38h] [rbp-88h]
  __m128i v149; // [rsp+40h] [rbp-80h] BYREF
  __m128i v150; // [rsp+50h] [rbp-70h] BYREF
  void *src; // [rsp+60h] [rbp-60h]
  _BYTE *v152; // [rsp+68h] [rbp-58h]
  __int64 v153; // [rsp+70h] [rbp-50h]
  void *v154; // [rsp+78h] [rbp-48h]
  _BYTE *v155; // [rsp+80h] [rbp-40h]
  __int64 v156; // [rsp+88h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  v6 = a3[1] - *a3;
  sub_26D0110(a1, v6);
  v7 = a3[1];
  v8 = *a3;
  v9 = a4;
  if ( v8 == v7 )
    goto LABEL_43;
  v145 = v7;
  v120 = v9;
  do
  {
    v23 = *(_QWORD *)(a2 + 16);
    v16 = *v8;
    v149.m128i_i64[1] = 0;
    v150.m128i_i16[0] = 1;
    v150.m128i_i64[1] = 0;
    src = 0;
    v152 = 0;
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    v24 = *(_QWORD *)(v23 + 8);
    v25 = *(unsigned int *)(v23 + 24);
    if ( (_DWORD)v25 )
    {
      v6 = ((_DWORD)v25 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v26 = (char **)(v24 + 16 * v6);
      v27 = *v26;
      if ( v16 == *v26 )
      {
LABEL_21:
        if ( v26 != (char **)(v24 + 16 * v25) )
        {
          v150.m128i_i8[0] = 0;
          v149.m128i_i64[1] = (__int64)v26[1];
        }
      }
      else
      {
        v102 = 1;
        while ( v27 != (char *)-4096LL )
        {
          v103 = v102 + 1;
          v6 = ((_DWORD)v25 - 1) & (unsigned int)(v102 + v6);
          v26 = (char **)(v24 + 16LL * (unsigned int)v6);
          v27 = *v26;
          if ( v16 == *v26 )
            goto LABEL_21;
          v102 = v103;
        }
      }
    }
    v28 = a1[1];
    v29 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v28 - *a1) >> 4);
    v149.m128i_i64[0] = v29;
    if ( v28 == a1[2] )
    {
      sub_26D33A0(a1, (_QWORD *)v28, (__int64)&v149);
      v19 = v154;
      v6 = v156 - (_QWORD)v154;
    }
    else
    {
      if ( !v28 )
      {
        a1[1] = 80;
        goto LABEL_16;
      }
      *(_QWORD *)v28 = v29;
      *(_QWORD *)(v28 + 8) = v149.m128i_i64[1];
      *(_WORD *)(v28 + 16) = v150.m128i_i16[0];
      *(_QWORD *)(v28 + 24) = v150.m128i_i64[1];
      v11 = v152 - (_BYTE *)src;
      v10 = v152 == src;
      *(_QWORD *)(v28 + 32) = 0;
      *(_QWORD *)(v28 + 40) = 0;
      *(_QWORD *)(v28 + 48) = 0;
      if ( v10 )
      {
        v13 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_186;
        v139 = v11;
        v12 = sub_22077B0(v11);
        v11 = v139;
        v13 = (char *)v12;
      }
      *(_QWORD *)(v28 + 32) = v13;
      *(_QWORD *)(v28 + 48) = &v13[v11];
      *(_QWORD *)(v28 + 40) = v13;
      v6 = (unsigned __int64)src;
      v14 = v152 - (_BYTE *)src;
      if ( v152 != src )
      {
        v140 = v152 - (_BYTE *)src;
        v15 = (char *)memmove(v13, src, v14);
        v14 = v140;
        v13 = v15;
      }
      v16 = &v13[v14];
      *(_QWORD *)(v28 + 40) = v16;
      v11 = v155 - (_BYTE *)v154;
      v10 = v155 == v154;
      *(_QWORD *)(v28 + 56) = 0;
      *(_QWORD *)(v28 + 64) = 0;
      *(_QWORD *)(v28 + 72) = 0;
      if ( v10 )
      {
        v18 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_186:
          sub_4261EA(v16, v6, v11);
        v141 = v11;
        v17 = sub_22077B0(v11);
        v11 = v141;
        v18 = (char *)v17;
      }
      *(_QWORD *)(v28 + 56) = v18;
      *(_QWORD *)(v28 + 72) = &v18[v11];
      *(_QWORD *)(v28 + 64) = v18;
      v19 = v154;
      v20 = v155 - (_BYTE *)v154;
      if ( v155 != v154 )
      {
        v130 = v155 - (_BYTE *)v154;
        v142 = v154;
        v21 = (char *)memmove(v18, v154, v20);
        v20 = v130;
        v19 = v142;
        v18 = v21;
      }
      *(_QWORD *)(v28 + 64) = &v18[v20];
      v22 = v156;
      a1[1] += 80LL;
      v6 = v22 - (_QWORD)v19;
    }
    if ( v19 )
      j_j___libc_free_0((unsigned __int64)v19);
LABEL_16:
    if ( src )
    {
      v6 = v153 - (_QWORD)src;
      j_j___libc_free_0((unsigned __int64)src);
    }
    ++v8;
  }
  while ( v145 != v8 );
  v49 = *a3;
  v122 = a3[1];
  if ( v122 != v49 )
  {
    v143 = (__int64 *)v49;
    while ( 1 )
    {
      v50 = *v143;
      v51 = *(_QWORD *)(a2 + 8);
      v52 = *(_DWORD *)(v51 + 24);
      if ( !v52 )
        break;
      v53 = *(_QWORD *)(v51 + 8);
      v34 = 1;
      v54 = ((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4);
      v31 = (v52 - 1) & v54;
      v30 = 0;
      v32 = (__int64 *)(v53 + 88LL * v31);
      v33 = *v32;
      if ( v50 != *v32 )
      {
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v30 )
            v30 = v32;
          v31 = (v52 - 1) & (v34 + v31);
          v32 = (__int64 *)(v53 + 88LL * v31);
          v33 = *v32;
          if ( v50 == *v32 )
            goto LABEL_67;
          ++v34;
        }
        if ( !v30 )
          v30 = v32;
        v35 = *(_DWORD *)(v51 + 16);
        ++*(_QWORD *)v51;
        v36 = v35 + 1;
        if ( 4 * v36 < 3 * v52 )
        {
          if ( v52 - *(_DWORD *)(v51 + 20) - v36 <= v52 >> 3 )
          {
            v148 = v50;
            sub_358A610(v51, v52);
            v111 = *(_DWORD *)(v51 + 24);
            if ( !v111 )
            {
LABEL_193:
              ++*(_DWORD *)(v51 + 16);
              BUG();
            }
            v112 = v111 - 1;
            v113 = *(_QWORD *)(v51 + 8);
            v50 = v148;
            v114 = 0;
            LODWORD(v115) = v112 & v54;
            v116 = 1;
            v30 = (_QWORD *)(v113 + 88LL * (unsigned int)v115);
            v117 = *v30;
            v36 = *(_DWORD *)(v51 + 16) + 1;
            if ( v148 != *v30 )
            {
              while ( v117 != -4096 )
              {
                if ( !v114 && v117 == -8192 )
                  v114 = v30;
                v115 = v112 & (unsigned int)(v115 + v116);
                v30 = (_QWORD *)(v113 + 88 * v115);
                v117 = *v30;
                if ( v148 == *v30 )
                  goto LABEL_39;
                ++v116;
              }
              if ( v114 )
                v30 = v114;
            }
          }
          goto LABEL_39;
        }
LABEL_136:
        v147 = v50;
        sub_358A610(v51, 2 * v52);
        v104 = *(_DWORD *)(v51 + 24);
        if ( !v104 )
          goto LABEL_193;
        v50 = v147;
        v105 = v104 - 1;
        v106 = *(_QWORD *)(v51 + 8);
        LODWORD(v107) = v105 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
        v30 = (_QWORD *)(v106 + 88LL * (unsigned int)v107);
        v108 = *v30;
        v36 = *(_DWORD *)(v51 + 16) + 1;
        if ( v147 != *v30 )
        {
          v109 = 1;
          v110 = 0;
          while ( v108 != -4096 )
          {
            if ( !v110 && v108 == -8192 )
              v110 = v30;
            v107 = v105 & (unsigned int)(v107 + v109);
            v30 = (_QWORD *)(v106 + 88 * v107);
            v108 = *v30;
            if ( v147 == *v30 )
              goto LABEL_39;
            ++v109;
          }
          if ( v110 )
            v30 = v110;
        }
LABEL_39:
        *(_DWORD *)(v51 + 16) = v36;
        if ( *v30 != -4096 )
          --*(_DWORD *)(v51 + 20);
        *v30 = v50;
        v30[1] = v30 + 3;
        v30[2] = 0x800000000LL;
        goto LABEL_42;
      }
LABEL_67:
      v55 = (__int64 *)v32[1];
      v56 = &v55[*((unsigned int *)v32 + 4)];
      if ( v56 != v55 )
      {
        v146 = ((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4);
        while ( 1 )
        {
          v58 = *(_DWORD *)(v120 + 24);
          v59 = *v55;
          v60 = *(_QWORD *)(v120 + 8);
          if ( v58 )
            break;
LABEL_72:
          if ( v56 == ++v55 )
            goto LABEL_42;
        }
        v61 = v58 - 1;
        v62 = ((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4);
        v63 = (v58 - 1) & v62;
        v64 = *(_QWORD *)(v60 + 16LL * v63);
        if ( v59 != v64 )
        {
          v72 = 1;
          while ( v64 != -4096 )
          {
            v63 = v61 & (v72 + v63);
            v64 = *(_QWORD *)(v60 + 16LL * v63);
            if ( v59 == v64 )
              goto LABEL_75;
            ++v72;
          }
          goto LABEL_72;
        }
LABEL_75:
        v150.m128i_i64[0] = 0;
        v150.m128i_i16[4] = 1;
        v65 = v61 & v146;
        src = 0;
        v66 = (_QWORD *)(v60 + 16LL * (v61 & v146));
        v67 = *v66;
        if ( v50 == *v66 )
        {
LABEL_76:
          v149.m128i_i64[0] = v66[1];
          goto LABEL_77;
        }
        v125 = 1;
        v73 = 0;
        while ( v67 != -4096 )
        {
          if ( v67 == -8192 && !v73 )
            v73 = v66;
          v65 = v61 & (v125 + v65);
          v66 = (_QWORD *)(v60 + 16LL * v65);
          v67 = *v66;
          if ( v50 == *v66 )
            goto LABEL_76;
          ++v125;
        }
        if ( !v73 )
          v73 = v66;
        v74 = *(_DWORD *)(v120 + 16);
        ++*(_QWORD *)v120;
        v75 = v74 + 1;
        if ( 4 * v75 >= 3 * v58 )
        {
          v128 = v56;
          v135 = v50;
          sub_2E3E470(v120, 2 * v58);
          v91 = *(_DWORD *)(v120 + 24);
          if ( !v91 )
            goto LABEL_192;
          v92 = v91 - 1;
          v93 = *(_QWORD *)(v120 + 8);
          v50 = v135;
          v94 = v92 & v146;
          v56 = v128;
          v75 = *(_DWORD *)(v120 + 16) + 1;
          v73 = (_QWORD *)(v93 + 16LL * (v92 & v146));
          v95 = *v73;
          if ( v135 != *v73 )
          {
            v136 = 1;
            v96 = 0;
            while ( v95 != -4096 )
            {
              if ( !v96 && v95 == -8192 )
                v96 = v73;
              v94 = v92 & (v136 + v94);
              v73 = (_QWORD *)(v93 + 16LL * v94);
              v95 = *v73;
              if ( v50 == *v73 )
                goto LABEL_91;
              ++v136;
            }
LABEL_117:
            if ( v96 )
              v73 = v96;
          }
        }
        else if ( v58 - *(_DWORD *)(v120 + 20) - v75 <= v58 >> 3 )
        {
          v129 = v56;
          v137 = v50;
          sub_2E3E470(v120, v58);
          v97 = *(_DWORD *)(v120 + 24);
          if ( !v97 )
          {
LABEL_192:
            ++*(_DWORD *)(v120 + 16);
            BUG();
          }
          v98 = v97 - 1;
          v99 = *(_QWORD *)(v120 + 8);
          v50 = v137;
          v100 = v98 & v146;
          v56 = v129;
          v75 = *(_DWORD *)(v120 + 16) + 1;
          v73 = (_QWORD *)(v99 + 16LL * (v98 & v146));
          v101 = *v73;
          if ( v137 != *v73 )
          {
            v138 = 1;
            v96 = 0;
            while ( v101 != -4096 )
            {
              if ( v101 == -8192 && !v96 )
                v96 = v73;
              v100 = v98 & (v138 + v100);
              v73 = (_QWORD *)(v99 + 16LL * v100);
              v101 = *v73;
              if ( v50 == *v73 )
                goto LABEL_91;
              ++v138;
            }
            goto LABEL_117;
          }
        }
LABEL_91:
        *(_DWORD *)(v120 + 16) = v75;
        if ( *v73 != -4096 )
          --*(_DWORD *)(v120 + 20);
        *v73 = v50;
        v73[1] = 0;
        v58 = *(_DWORD *)(v120 + 24);
        v149.m128i_i64[0] = 0;
        v60 = *(_QWORD *)(v120 + 8);
        if ( !v58 )
        {
          ++*(_QWORD *)v120;
          goto LABEL_95;
        }
        v61 = v58 - 1;
LABEL_77:
        v68 = v61 & v62;
        v69 = (_QWORD *)(v60 + 16LL * (v61 & v62));
        v70 = *v69;
        if ( v59 == *v69 )
        {
LABEL_78:
          v71 = v69[1];
        }
        else
        {
          v133 = 1;
          v81 = 0;
          while ( v70 != -4096 )
          {
            if ( v70 == -8192 && !v81 )
              v81 = v69;
            v68 = v61 & (v133 + v68);
            v69 = (_QWORD *)(v60 + 16LL * v68);
            v70 = *v69;
            if ( v59 == *v69 )
              goto LABEL_78;
            ++v133;
          }
          if ( !v81 )
            v81 = v69;
          v83 = *(_DWORD *)(v120 + 16);
          ++*(_QWORD *)v120;
          v80 = v83 + 1;
          if ( 4 * v80 >= 3 * v58 )
          {
LABEL_95:
            v126 = v56;
            v132 = v50;
            sub_2E3E470(v120, 2 * v58);
            v76 = *(_DWORD *)(v120 + 24);
            if ( !v76 )
              goto LABEL_192;
            v77 = v76 - 1;
            v78 = *(_QWORD *)(v120 + 8);
            v79 = v77 & v62;
            v50 = v132;
            v56 = v126;
            v80 = *(_DWORD *)(v120 + 16) + 1;
            v81 = (_QWORD *)(v78 + 16LL * v79);
            v82 = *v81;
            if ( v59 != *v81 )
            {
              v118 = 1;
              v119 = 0;
              while ( v82 != -4096 )
              {
                if ( v82 != -8192 || v119 )
                  v81 = v119;
                v79 = v77 & (v118 + v79);
                v82 = *(_QWORD *)(v78 + 16LL * v79);
                if ( v59 == v82 )
                {
                  v81 = (_QWORD *)(v78 + 16LL * v79);
                  goto LABEL_97;
                }
                ++v118;
                v119 = v81;
                v81 = (_QWORD *)(v78 + 16LL * v79);
              }
              if ( v119 )
                v81 = v119;
            }
          }
          else if ( v58 - (v80 + *(_DWORD *)(v120 + 20)) <= v58 >> 3 )
          {
            v127 = v56;
            v134 = v50;
            sub_2E3E470(v120, v58);
            v84 = *(_DWORD *)(v120 + 24);
            if ( !v84 )
              goto LABEL_192;
            v85 = v84 - 1;
            v86 = *(_QWORD *)(v120 + 8);
            v87 = 0;
            v88 = v85 & v62;
            v50 = v134;
            v56 = v127;
            v89 = 1;
            v80 = *(_DWORD *)(v120 + 16) + 1;
            v81 = (_QWORD *)(v86 + 16LL * v88);
            v90 = *v81;
            if ( v59 != *v81 )
            {
              while ( v90 != -4096 )
              {
                if ( !v87 && v90 == -8192 )
                  v87 = v81;
                v88 = v85 & (v89 + v88);
                v81 = (_QWORD *)(v86 + 16LL * v88);
                v90 = *v81;
                if ( v59 == *v81 )
                  goto LABEL_97;
                ++v89;
              }
              if ( v87 )
                v81 = v87;
            }
          }
LABEL_97:
          *(_DWORD *)(v120 + 16) = v80;
          if ( *v81 != -4096 )
            --*(_DWORD *)(v120 + 20);
          *v81 = v59;
          v71 = 0;
          v81[1] = 0;
        }
        v149.m128i_i64[1] = v71;
        v57 = (__m128i *)a1[4];
        if ( v57 == (__m128i *)a1[5] )
        {
          v124 = v56;
          v131 = v50;
          sub_26D37E0(a1 + 3, v57, &v149);
          v50 = v131;
          v56 = v124;
        }
        else
        {
          if ( v57 )
          {
            *v57 = _mm_loadu_si128(&v149);
            v57[1] = _mm_loadu_si128(&v150);
            v57[2].m128i_i64[0] = (__int64)src;
            v57 = (__m128i *)a1[4];
          }
          a1[4] = (unsigned __int64)&v57[2].m128i_u64[1];
        }
        goto LABEL_72;
      }
LABEL_42:
      if ( v122 == (char **)++v143 )
        goto LABEL_43;
    }
    ++*(_QWORD *)v51;
    goto LABEL_136;
  }
LABEL_43:
  v37 = (__int64 *)a1[4];
  for ( i = (__int64 *)a1[3]; v37 != i; *(_QWORD *)(v40 + 64) = v41 + 8 )
  {
    while ( 1 )
    {
      v42 = *i;
      v43 = i[1];
      v149.m128i_i64[0] = (__int64)i;
      v44 = *a1 + 80 * v42;
      v39 = *(_BYTE **)(v44 + 40);
      if ( v39 == *(_BYTE **)(v44 + 48) )
      {
        sub_26D7E10(v44 + 32, v39, &v149);
      }
      else
      {
        if ( v39 )
        {
          *(_QWORD *)v39 = i;
          v39 = *(_BYTE **)(v44 + 40);
        }
        *(_QWORD *)(v44 + 40) = v39 + 8;
      }
      v149.m128i_i64[0] = (__int64)i;
      v40 = *a1 + 80 * v43;
      v41 = *(_BYTE **)(v40 + 64);
      if ( v41 != *(_BYTE **)(v40 + 72) )
        break;
      i += 5;
      sub_26D7E10(v40 + 56, v41, &v149);
      if ( v37 == i )
        goto LABEL_55;
    }
    if ( v41 )
    {
      *(_QWORD *)v41 = i;
      v41 = *(_BYTE **)(v40 + 64);
    }
    i += 5;
  }
LABEL_55:
  v45 = 0;
  v46 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[1] - *a1) >> 4);
  v47 = *a1;
  if ( v46 )
  {
    while ( *(_QWORD *)(v47 + 64) != *(_QWORD *)(v47 + 56) )
    {
      ++v45;
      v47 += 80;
      if ( v45 == v46 )
        goto LABEL_133;
    }
    a1[6] = v45;
    if ( !*(_QWORD *)(v47 + 8) )
    {
LABEL_60:
      if ( !*(_BYTE *)(v47 + 16) )
        *(_QWORD *)(v47 + 8) = 1;
    }
  }
  else
  {
LABEL_133:
    v47 = *a1 + 80 * a1[6];
    if ( !*(_QWORD *)(v47 + 8) )
      goto LABEL_60;
  }
  return a1;
}
