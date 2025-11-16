// Function: sub_392FE90
// Address: 0x392fe90
//
void __fastcall sub_392FE90(_DWORD *a1, __int64 *a2, __int64 **a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  _DWORD *v6; // r15
  __int64 v7; // rdx
  char v8; // al
  char v9; // al
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 *v15; // r14
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // r15
  bool v19; // r12
  bool v20; // bl
  bool v21; // r13
  int v22; // edx
  __int64 v23; // rcx
  int v24; // esi
  unsigned int v25; // eax
  __int64 v26; // rdi
  bool v27; // bl
  char v28; // al
  __int64 v29; // rcx
  __int64 v30; // r10
  __int64 v31; // r13
  _QWORD *v32; // r15
  __int64 v33; // r12
  unsigned __int64 v34; // rax
  __int64 v35; // r15
  char *v36; // r12
  __int64 v37; // rbx
  unsigned __int64 v38; // rax
  unsigned int v39; // eax
  __int64 *v40; // r14
  int i; // r15d
  unsigned int v42; // esi
  __int64 v43; // rbx
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  int v47; // ebx
  __m128i *v48; // rax
  __m128i *v49; // r13
  __m128i *v50; // r12
  char *v51; // r13
  __int64 v52; // r15
  unsigned __int64 v53; // rax
  unsigned int v54; // eax
  int v55; // edx
  __int64 v56; // rdx
  _QWORD *v57; // r12
  __int64 v58; // rbx
  unsigned __int64 *v59; // rax
  unsigned __int64 *v60; // rsi
  unsigned __int64 v61; // rcx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // r14
  unsigned int *v64; // rbx
  _QWORD *v65; // r13
  _DWORD *v66; // r12
  __int64 v67; // r13
  __int64 v68; // rax
  unsigned int *v69; // r15
  __int32 v70; // eax
  __int64 v71; // rdi
  unsigned __int32 v72; // esi
  _QWORD *v73; // r14
  __int64 v74; // rbx
  unsigned __int64 *v75; // rax
  unsigned __int64 *v76; // rsi
  unsigned __int64 v77; // rcx
  unsigned __int64 v78; // rdx
  __int64 v79; // rdi
  unsigned __int64 v80; // rax
  int v81; // ebx
  unsigned __int64 v82; // rax
  char v83; // al
  __int64 *v84; // r13
  __int64 v85; // r12
  _QWORD *v86; // r13
  unsigned __int64 v87; // rax
  __m128i *v88; // rsi
  __m128i *v89; // rsi
  unsigned __int64 v90; // rdx
  __int64 v91; // rax
  unsigned __int64 v92; // rax
  int v93; // eax
  int v94; // esi
  __int64 v95; // rcx
  unsigned int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // rdi
  int v99; // eax
  int v100; // r8d
  unsigned __int64 v101; // rdx
  __int64 v102; // rax
  unsigned __int64 v103; // rax
  __int64 v104; // rdx
  unsigned __int64 v105; // rax
  int v106; // esi
  __int64 v107; // rdi
  int v108; // esi
  unsigned int v109; // ecx
  __int64 v110; // r10
  __int64 v111; // rbx
  int v112; // eax
  int v113; // eax
  int v114; // r8d
  __int64 v118; // [rsp+28h] [rbp-148h]
  char v119; // [rsp+37h] [rbp-139h]
  __int64 v120; // [rsp+38h] [rbp-138h]
  int v121; // [rsp+38h] [rbp-138h]
  __int64 v123; // [rsp+48h] [rbp-128h]
  _DWORD *v124; // [rsp+48h] [rbp-128h]
  _DWORD *v125; // [rsp+48h] [rbp-128h]
  char v126; // [rsp+58h] [rbp-118h]
  __int64 v127; // [rsp+58h] [rbp-118h]
  __int64 v128; // [rsp+58h] [rbp-118h]
  __int64 *v129; // [rsp+60h] [rbp-110h]
  __int64 v130; // [rsp+60h] [rbp-110h]
  char *v131; // [rsp+60h] [rbp-110h]
  _DWORD *v132; // [rsp+60h] [rbp-110h]
  __int64 v135; // [rsp+70h] [rbp-100h]
  __m128i *v136; // [rsp+70h] [rbp-100h]
  __m128i *v137; // [rsp+70h] [rbp-100h]
  unsigned __int64 v138; // [rsp+88h] [rbp-E8h] BYREF
  void *base; // [rsp+90h] [rbp-E0h] BYREF
  __m128i *v140; // [rsp+98h] [rbp-D8h]
  const __m128i *v141; // [rsp+A0h] [rbp-D0h]
  void *v142; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i *v143; // [rsp+B8h] [rbp-B8h]
  const __m128i *v144; // [rsp+C0h] [rbp-B0h]
  _QWORD v145[2]; // [rsp+D0h] [rbp-A0h] BYREF
  char v146; // [rsp+E0h] [rbp-90h]
  char v147; // [rsp+E1h] [rbp-8Fh]
  __m128i v148; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v149; // [rsp+100h] [rbp-70h] BYREF
  _DWORD *v150; // [rsp+110h] [rbp-60h] BYREF
  bool v151; // [rsp+118h] [rbp-58h]
  unsigned __int64 v152; // [rsp+120h] [rbp-50h]
  unsigned int *v153; // [rsp+128h] [rbp-48h]
  __int64 v154; // [rsp+130h] [rbp-40h]
  int v155; // [rsp+138h] [rbp-38h]

  v6 = a1;
  v7 = *(_QWORD *)a1;
  v120 = *a2;
  v8 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL);
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v151 = (v8 & 2) != 0;
  v9 = *(_BYTE *)(*(_QWORD *)(v7 + 8) + 12LL);
  v145[0] = ".symtab";
  v149.m128i_i16[0] = 257;
  v150 = a1;
  v147 = 1;
  v146 = 3;
  v10 = sub_38C3B80(v120, (__int64)v145, 2, 0, (v9 & 2) == 0 ? 16 : 24, (__int64)&v148, -1, 0);
  v138 = v10;
  *(_DWORD *)(v10 + 24) = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0 ? 4 : 8;
  a1[24] = sub_392F0E0(a1, v10);
  sub_392DC80((__int64)a1, *(_DWORD *)(v138 + 24));
  v11 = (_QWORD *)*((_QWORD *)a1 + 1);
  v118 = (*(__int64 (__fastcall **)(_QWORD *))(*v11 + 64LL))(v11) + v11[3] - v11[1];
  sub_392F510((__int64)&v150, 0, 0, 0, 0, 0, 0, 0);
  v12 = a2[8];
  v13 = (__int64 *)a2[7];
  v14 = (__int64)(a1 + 8);
  v119 = 0;
  base = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v129 = (__int64 *)v12;
  if ( v13 == (__int64 *)v12 )
  {
LABEL_22:
    v121 = 0;
    goto LABEL_23;
  }
  v123 = (__int64)(a1 + 8);
  v15 = v13;
  do
  {
    while ( 1 )
    {
      v18 = *v15;
      v19 = (*(_BYTE *)(*v15 + 9) & 2) != 0;
      v126 = *(_BYTE *)(*v15 + 9) & 2;
      v20 = sub_38E2770(*v15);
      v21 = sub_38E2790(v18);
      v22 = *(_DWORD *)(*(_QWORD *)a1 + 72LL);
      if ( v22 )
      {
        v23 = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
        v24 = v22 - 1;
        v25 = (v22 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v26 = *(_QWORD *)(v23 + 16LL * v25);
        v22 = 1;
        if ( v18 != v26 )
        {
          while ( 1 )
          {
            if ( v26 == -8 )
            {
              v22 = 0;
              goto LABEL_17;
            }
            v25 = v24 & (v22 + v25);
            v26 = *(_QWORD *)(v23 + 16LL * v25);
            if ( v18 == v26 )
              break;
            ++v22;
          }
          v22 = 1;
        }
      }
LABEL_17:
      v27 = v19 || v21 || v20;
      v28 = *(_BYTE *)(v18 + 9) & 0xC;
      if ( v28 == 8 )
      {
        v29 = *(_QWORD *)(v18 + 24);
        *(_BYTE *)(v18 + 8) |= 4u;
        if ( *(_DWORD *)v29 == 2 && *(_WORD *)(v29 + 16) == 27 )
          break;
      }
      if ( v27 )
        goto LABEL_10;
      if ( v22 )
        goto LABEL_14;
      v16 = *(_QWORD *)v18;
      if ( v28 == 8 )
      {
        if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          *(_BYTE *)(v18 + 8) |= 4u;
          v90 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24));
          v91 = v90 | *(_QWORD *)v18 & 7LL;
          *(_QWORD *)v18 = v91;
          if ( !v90 )
          {
            sub_38CF550(a3, v18);
            goto LABEL_14;
          }
          if ( (v91 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
            if ( (*(_BYTE *)(v18 + 9) & 0xC) != 8
              || (*(_BYTE *)(v18 + 8) |= 4u,
                  v92 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24)),
                  *(_QWORD *)v18 = v92 | *(_QWORD *)v18 & 7LL,
                  !v92) )
            {
LABEL_7:
              if ( !sub_38E27B0(v18) )
                goto LABEL_14;
            }
          }
        }
      }
      else if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        goto LABEL_7;
      }
      if ( (*(_BYTE *)(v18 + 8) & 1) != 0 || (unsigned int)sub_38E2700(v18) == 3 )
        goto LABEL_14;
LABEL_10:
      v17 = *(_BYTE *)(v18 + 8);
      if ( (v17 & 1) != 0 && (*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        if ( (*(_BYTE *)(v18 + 9) & 0xC) != 8
          || (v79 = *(_QWORD *)(v18 + 24),
              *(_BYTE *)(v18 + 8) = v17 | 4,
              v80 = (unsigned __int64)sub_38CE440(v79),
              *(_QWORD *)v18 = v80 | *(_QWORD *)v18 & 7LL,
              !v80) )
        {
          v149.m128i_i16[0] = 259;
          v148.m128i_i64[0] = (__int64)"Undefined temporary symbol";
          sub_38BE3D0(v120, 0, (__int64)&v148);
          goto LABEL_14;
        }
      }
      v149 = 0u;
      v148.m128i_i64[0] = v18;
      v81 = sub_38E27C0(v18);
      v82 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v82 )
      {
        v82 = 0;
        if ( (*(_BYTE *)(v18 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v18 + 8) |= 4u;
          v82 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24));
          *(_QWORD *)v18 = v82 | *(_QWORD *)v18 & 7LL;
        }
      }
      if ( off_4CF6DB8 == (_UNKNOWN *)v82 )
      {
        v148.m128i_i32[2] = 65521;
        goto LABEL_83;
      }
      v83 = *(_BYTE *)(v18 + 9) & 0xC;
      if ( v83 == 12 )
      {
        v148.m128i_i32[2] = 65522;
LABEL_83:
        if ( (*(_BYTE *)v18 & 4) != 0 )
        {
          v84 = *(__int64 **)(v18 - 8);
          v85 = *v84;
          v86 = v84 + 2;
        }
        else
        {
          v85 = 0;
          v86 = 0;
        }
        if ( (unsigned int)sub_38E2700(v18) != 3 )
        {
          v149.m128i_i64[0] = (__int64)v86;
          v149.m128i_i64[1] = v85;
          v87 = sub_16D3930(v86, v85);
          sub_1680880(v123, (__int64)v86, (v87 << 32) | (unsigned int)v85);
        }
        if ( v81 )
        {
          v89 = v143;
          if ( v143 == v144 )
          {
            sub_392D490((unsigned __int64 *)&v142, v143, &v148);
          }
          else
          {
            if ( v143 )
            {
              *v143 = _mm_loadu_si128(&v148);
              v89[1] = _mm_loadu_si128(&v149);
              v89 = v143;
            }
            v143 = v89 + 2;
          }
        }
        else
        {
          v88 = v140;
          if ( v140 == v141 )
          {
            sub_392D490((unsigned __int64 *)&base, v140, &v148);
          }
          else
          {
            if ( v140 )
            {
              *v140 = _mm_loadu_si128(&v148);
              v88[1] = _mm_loadu_si128(&v149);
              v88 = v140;
            }
            v140 = v88 + 2;
          }
        }
        goto LABEL_14;
      }
      if ( (*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v103 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_130:
        v104 = *(_QWORD *)(v103 + 24);
        goto LABEL_122;
      }
      if ( v83 != 8
        || (*(_BYTE *)(v18 + 8) |= 4u,
            v101 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24)),
            v102 = v101 | *(_QWORD *)v18 & 7LL,
            *(_QWORD *)v18 = v102,
            !v101) )
      {
        if ( !v126 && v21 )
        {
          v93 = *(_DWORD *)(a5 + 24);
          if ( v93 )
          {
            v94 = v93 - 1;
            v95 = *(_QWORD *)(a5 + 8);
            v96 = (v93 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v97 = (__int64 *)(v95 + 16LL * v96);
            v98 = *v97;
            if ( v18 != *v97 )
            {
              v99 = 1;
              while ( v98 != -8 )
              {
                v100 = v99 + 1;
                v96 = v94 & (v99 + v96);
                v97 = (__int64 *)(v95 + 16LL * v96);
                v98 = *v97;
                if ( v18 == *v97 )
                  goto LABEL_127;
                v99 = v100;
              }
              goto LABEL_101;
            }
LABEL_127:
            v148.m128i_i32[2] = *((_DWORD *)v97 + 2);
            if ( v148.m128i_i32[2] > 0xFEFFu )
              v119 = 1;
            goto LABEL_83;
          }
        }
LABEL_101:
        v148.m128i_i32[2] = 0;
        goto LABEL_83;
      }
      v103 = v102 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v103 )
        goto LABEL_130;
      if ( (*(_BYTE *)(v18 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v18 + 8) |= 4u;
        v103 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24));
        *(_QWORD *)v18 = v103 | *(_QWORD *)v18 & 7LL;
        if ( v103 )
          goto LABEL_130;
      }
      v104 = 0;
LABEL_122:
      if ( a1[6] != 1
        || (v105 = *(_QWORD *)(v104 + 160), v105 <= 3)
        || *(_DWORD *)(*(_QWORD *)(v104 + 152) + v105 - 4) != 1870095406 )
      {
        v106 = *(_DWORD *)(a4 + 24);
        if ( v106 )
        {
          v107 = *(_QWORD *)(a4 + 8);
          v108 = v106 - 1;
          v109 = v108 & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
          v97 = (__int64 *)(v107 + 16LL * v109);
          v110 = *v97;
          if ( v104 == *v97 )
            goto LABEL_127;
          v113 = 1;
          while ( v110 != -8 )
          {
            v114 = v113 + 1;
            v109 = v108 & (v113 + v109);
            v97 = (__int64 *)(v107 + 16LL * v109);
            v110 = *v97;
            if ( v104 == *v97 )
              goto LABEL_127;
            v113 = v114;
          }
        }
        goto LABEL_101;
      }
LABEL_14:
      if ( v129 == ++v15 )
        goto LABEL_21;
    }
    ++v15;
  }
  while ( v129 != v15 );
LABEL_21:
  v14 = v123;
  v6 = a1;
  if ( !v119 )
    goto LABEL_22;
  v149.m128i_i16[0] = 257;
  v147 = 1;
  v145[0] = ".symtab_shndxr";
  v146 = 3;
  v111 = sub_38C3B80(v120, (__int64)v145, 18, 0, 4, (__int64)&v148, -1, 0);
  v112 = sub_392F0E0(a1, v111);
  *(_DWORD *)(v111 + 24) = 4;
  v121 = v112;
LABEL_23:
  v30 = a2[19];
  v135 = a2[20];
  v130 = (v135 - v30) >> 5;
  if ( v30 == v135 )
  {
    sub_1680590(v14);
  }
  else
  {
    v127 = a2[19];
    v31 = v127;
    v124 = v6;
    do
    {
      v32 = *(_QWORD **)v31;
      v33 = *(_QWORD *)(v31 + 8);
      v31 += 32;
      v34 = sub_16D3930(v32, v33);
      sub_1680880(v14, (__int64)v32, (v34 << 32) | (unsigned int)v33);
    }
    while ( v135 != v31 );
    sub_1680590(v14);
    v35 = v127;
    do
    {
      v36 = *(char **)v35;
      v37 = *(_QWORD *)(v35 + 8);
      v35 += 32;
      v38 = sub_16D3930(v36, v37);
      v39 = sub_167FE60(v14, v36, (unsigned int)v37 | (v38 << 32));
      sub_392F510((__int64)&v150, v39, 4, 0, 0, 0, 0xFFF1u, 1);
    }
    while ( v135 != v35 );
    v6 = v124;
  }
  if ( (char *)v140 - (_BYTE *)base > 32 )
    qsort(base, ((char *)v140 - (_BYTE *)base) >> 5, 0x20u, (__compar_fn_t)sub_392D6E0);
  if ( (char *)v143 - (_BYTE *)v142 > 32 )
    qsort(v142, ((char *)v143 - (_BYTE *)v142) >> 5, 0x20u, (__compar_fn_t)sub_392D6E0);
  v136 = v140;
  if ( v140 == base )
  {
    v47 = v130 + 1;
  }
  else
  {
    v128 = v14;
    v40 = (__int64 *)base;
    v125 = v6;
    for ( i = v130 + 1; ; ++i )
    {
      v42 = 0;
      if ( (unsigned int)sub_38E2700(*v40) != 3 )
      {
        v43 = v40[3];
        v131 = (char *)v40[2];
        v44 = sub_16D3930(v131, v43);
        v42 = sub_167FE60(v128, v131, (v44 << 32) | (unsigned int)v43);
      }
      v45 = *v40;
      v46 = (__int64)v40;
      v47 = i + 1;
      v40 += 4;
      *(_DWORD *)(v45 + 16) = i;
      sub_392F830((__int64)&v150, v42, v46, a3);
      if ( v136 == (__m128i *)v40 )
        break;
    }
    v14 = v128;
    v6 = v125;
  }
  v48 = v143;
  v49 = (__m128i *)v142;
  v6[22] = v47;
  v137 = v48;
  if ( v48 != v49 )
  {
    v132 = v6;
    v50 = v49;
    do
    {
      v51 = (char *)v50[1].m128i_i64[0];
      v52 = v50[1].m128i_i64[1];
      v53 = sub_16D3930(v51, v52);
      v54 = sub_167FE60(v14, v51, (v53 << 32) | (unsigned int)v52);
      v55 = v47++;
      *(_DWORD *)(v50->m128i_i64[0] + 16) = v55;
      v56 = (__int64)v50;
      v50 += 2;
      sub_392F830((__int64)&v150, v54, v56, a3);
    }
    while ( v137 != v50 );
    v6 = v132;
  }
  v57 = (_QWORD *)*((_QWORD *)v6 + 1);
  v58 = (*(__int64 (__fastcall **)(_QWORD *))(*v57 + 64LL))(v57) + v57[3] - v57[1];
  v59 = (unsigned __int64 *)a6[2];
  v60 = a6 + 1;
  if ( !v59 )
    goto LABEL_109;
  do
  {
    while ( 1 )
    {
      v61 = v59[2];
      v62 = v59[3];
      if ( v59[4] >= v138 )
        break;
      v59 = (unsigned __int64 *)v59[3];
      if ( !v62 )
        goto LABEL_49;
    }
    v60 = v59;
    v59 = (unsigned __int64 *)v59[2];
  }
  while ( v61 );
LABEL_49:
  if ( v60 == a6 + 1 || v60[4] > v138 )
  {
LABEL_109:
    v148.m128i_i64[0] = (__int64)&v138;
    v60 = sub_392FDD0(a6, (__int64)v60, (unsigned __int64 **)&v148);
  }
  v63 = v152;
  v60[6] = v58;
  v64 = v153;
  v60[5] = v118;
  if ( (unsigned int *)v63 != v64 )
  {
    v65 = (_QWORD *)*((_QWORD *)v6 + 1);
    v66 = v6;
    v67 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64 *, unsigned __int64, unsigned __int64))(*v65 + 64LL))(
            v65,
            v60,
            v62,
            v61)
        + v65[3]
        - v65[1];
    v68 = *((_QWORD *)v6 + 13);
    v69 = (unsigned int *)v63;
    v145[0] = *(_QWORD *)(v68 + 8LL * (unsigned int)(v121 - 1));
    do
    {
      v70 = *v69;
      v71 = *((_QWORD *)v66 + 1);
      v72 = _byteswap_ulong(*v69);
      if ( (unsigned int)(v66[4] - 1) > 1 )
        v70 = v72;
      ++v69;
      v148.m128i_i32[0] = v70;
      sub_16E7EE0(v71, v148.m128i_i8, 4u);
    }
    while ( v64 != v69 );
    v73 = (_QWORD *)*((_QWORD *)v66 + 1);
    v74 = (*(__int64 (__fastcall **)(_QWORD *))(*v73 + 64LL))(v73) + v73[3] - v73[1];
    v75 = (unsigned __int64 *)a6[2];
    if ( !v75 )
    {
      v76 = a6 + 1;
      goto LABEL_63;
    }
    v76 = a6 + 1;
    do
    {
      while ( 1 )
      {
        v77 = v75[2];
        v78 = v75[3];
        if ( v75[4] >= v145[0] )
          break;
        v75 = (unsigned __int64 *)v75[3];
        if ( !v78 )
          goto LABEL_61;
      }
      v76 = v75;
      v75 = (unsigned __int64 *)v75[2];
    }
    while ( v77 );
LABEL_61:
    if ( a6 + 1 == v76 || v76[4] > v145[0] )
    {
LABEL_63:
      v148.m128i_i64[0] = (__int64)v145;
      v76 = sub_392FDD0(a6, (__int64)v76, (unsigned __int64 **)&v148);
    }
    v76[5] = v67;
    v76[6] = v74;
  }
  if ( v142 )
    j_j___libc_free_0((unsigned __int64)v142);
  if ( base )
    j_j___libc_free_0((unsigned __int64)base);
  if ( v152 )
    j_j___libc_free_0(v152);
}
