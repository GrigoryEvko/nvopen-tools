// Function: sub_262F360
// Address: 0x262f360
//
__int64 __fastcall sub_262F360(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  size_t v3; // r15
  int v4; // eax
  unsigned __int64 v5; // rdx
  char *v6; // rsi
  char *v7; // rbx
  char *v8; // r13
  char *v9; // rdi
  char v10; // al
  __int64 v11; // rax
  void *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  char *v15; // r14
  __int64 v16; // rbx
  unsigned __int64 v17; // rbx
  char *v18; // r14
  __int64 v19; // rbx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  char *v22; // r14
  __int64 v23; // rbx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  const __m128i *v26; // rbx
  const __m128i *v27; // rax
  __m128i *v28; // r14
  const __m128i *v29; // r12
  const __m128i *v30; // rbx
  unsigned __int64 v31; // r13
  char *v32; // rdi
  __int64 v33; // r13
  __int64 v34; // rax
  const __m128i *v35; // rbx
  const __m128i *v36; // r15
  __m128i *v37; // r14
  unsigned __int64 v38; // r12
  char *v39; // rcx
  __int64 v40; // r12
  const __m128i *v41; // rsi
  __int64 v42; // rbx
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // rdi
  const __m128i *v45; // r12
  unsigned __int64 v46; // rcx
  char *v47; // r8
  const __m128i *v48; // r13
  __int64 v49; // rbx
  unsigned __int64 v50; // r14
  unsigned __int64 v51; // rdi
  __int64 v52; // rbx
  unsigned __int64 v53; // r14
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // rcx
  char *v62; // r12
  __int64 v63; // rax
  _QWORD *v64; // r14
  _QWORD *v65; // r15
  char *v66; // r8
  __int64 v67; // rdx
  __int64 v68; // rbx
  signed __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // rcx
  bool v72; // cf
  unsigned __int64 v73; // rax
  unsigned __int64 v74; // r13
  char *v75; // rcx
  int v76; // edx
  __int64 v77; // rax
  const __m128i *v78; // rsi
  __int64 v79; // rbx
  unsigned __int64 v80; // r12
  unsigned __int64 v81; // rdi
  char *v82; // rax
  __int64 v83; // rbx
  unsigned __int64 v84; // rdi
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // r12
  __int64 v87; // rbx
  unsigned __int64 v88; // rdi
  unsigned __int64 v89; // r13
  __int64 v90; // rax
  __int64 v92; // [rsp+10h] [rbp-170h]
  _QWORD *v93; // [rsp+18h] [rbp-168h]
  char *v94; // [rsp+20h] [rbp-160h]
  const __m128i *v95; // [rsp+30h] [rbp-150h]
  char *v96; // [rsp+30h] [rbp-150h]
  char *v97; // [rsp+30h] [rbp-150h]
  char *v98; // [rsp+30h] [rbp-150h]
  __int64 v99; // [rsp+30h] [rbp-150h]
  size_t na; // [rsp+38h] [rbp-148h]
  size_t nb; // [rsp+38h] [rbp-148h]
  size_t nc; // [rsp+38h] [rbp-148h]
  size_t nd; // [rsp+38h] [rbp-148h]
  size_t ne; // [rsp+38h] [rbp-148h]
  size_t n; // [rsp+38h] [rbp-148h]
  size_t *v106; // [rsp+40h] [rbp-140h]
  size_t *v107; // [rsp+48h] [rbp-138h]
  char v108; // [rsp+5Fh] [rbp-121h] BYREF
  __int64 v109; // [rsp+60h] [rbp-120h] BYREF
  char v110; // [rsp+74h] [rbp-10Ch] BYREF
  _BYTE v111[11]; // [rsp+75h] [rbp-10Bh] BYREF
  const __m128i *v112; // [rsp+80h] [rbp-100h] BYREF
  const __m128i *v113; // [rsp+88h] [rbp-F8h]
  const __m128i *v114; // [rsp+90h] [rbp-F0h]
  __m128i v115[11]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v116; // [rsp+150h] [rbp-30h] BYREF

  result = *(_QWORD *)(a2 + 24);
  v92 = a2 + 8;
  v93 = (_QWORD *)result;
  if ( result != a2 + 8 )
  {
    while ( 1 )
    {
      v112 = 0;
      v113 = 0;
      v114 = 0;
      v106 = (size_t *)v93[8];
      if ( v106 != (size_t *)v93[7] )
        break;
LABEL_105:
      result = sub_220EEE0((__int64)v93);
      v93 = (_QWORD *)result;
      if ( v92 == result )
        return result;
    }
    v107 = (size_t *)v93[7];
    while ( 1 )
    {
      v3 = *v107;
      v4 = *(_DWORD *)(*v107 + 8);
      if ( v4 != 1 )
      {
        if ( !v4 && *(_QWORD *)(v3 + 64) )
        {
          v76 = *(_BYTE *)(v3 + 12) & 0xF;
          v115[0].m128i_i32[0] = v76;
          v115[0].m128i_i32[1] = (*(_BYTE *)(v3 + 12) >> 4) & 3;
          v115[0].m128i_i8[8] = (*(_BYTE *)(v3 + 12) & 0x40) != 0;
          v115[0].m128i_i8[9] = *(_BYTE *)(v3 + 12) >> 7;
          v115[0].m128i_i8[10] = *(_BYTE *)(v3 + 13) & 1;
          v115[0].m128i_i8[11] = (*(_BYTE *)(v3 + 13) & 2) != 0;
          v115[0].m128i_i32[3] = (*(_BYTE *)(v3 + 13) & 4) != 0;
          v77 = *(_QWORD *)(*(_QWORD *)(v3 + 56) & 0xFFFFFFFFFFFFFFF8LL);
          v115[1].m128i_i8[8] = 1;
          memset(&v115[2], 0, 144);
          v115[1].m128i_i64[0] = v77;
          v78 = v113;
          if ( v113 == v114 )
          {
            sub_2628F50((unsigned __int64 *)&v112, v113, v115);
            v86 = v115[10].m128i_u64[0];
            v87 = v115[9].m128i_i64[1];
            if ( v115[9].m128i_i64[1] != v115[10].m128i_i64[0] )
            {
              do
              {
                v88 = *(_QWORD *)(v87 + 16);
                if ( v88 )
                  j_j___libc_free_0(v88);
                v87 += 40;
              }
              while ( v86 != v87 );
              v86 = v115[9].m128i_u64[1];
            }
            if ( v86 )
              j_j___libc_free_0(v86);
          }
          else
          {
            if ( v113 )
            {
              v113->m128i_i32[0] = v76;
              *(__int64 *)((char *)v78->m128i_i64 + 4) = *(__int64 *)((char *)v115[0].m128i_i64 + 4);
              v78->m128i_i32[3] = v115[0].m128i_i32[3];
              v78[1] = _mm_loadu_si128(&v115[1]);
              v78[2] = v115[2];
              v78[3].m128i_i64[0] = v115[3].m128i_i64[0];
              memset(&v115[2], 0, 24);
              v78[3].m128i_i64[1] = v115[3].m128i_i64[1];
              v78[4] = v115[4];
              v115[4] = 0u;
              v115[3].m128i_i64[1] = 0;
              v78[5] = v115[5];
              v78[6].m128i_i64[0] = v115[6].m128i_i64[0];
              memset(&v115[5], 0, 24);
              v78[6].m128i_i64[1] = v115[6].m128i_i64[1];
              v78[7] = v115[7];
              v115[7] = 0u;
              v115[6].m128i_i64[1] = 0;
              v78[8] = v115[8];
              v78[9].m128i_i64[0] = v115[9].m128i_i64[0];
              memset(&v115[8], 0, 24);
              v78[9].m128i_i64[1] = v115[9].m128i_i64[1];
              v78[10] = v115[10];
              v78 = v113;
            }
            v113 = v78 + 11;
          }
          v79 = v115[8].m128i_i64[1];
          v80 = v115[8].m128i_u64[0];
          if ( v115[8].m128i_i64[1] != v115[8].m128i_i64[0] )
          {
            do
            {
              v81 = *(_QWORD *)(v80 + 16);
              if ( v81 )
                j_j___libc_free_0(v81);
              v80 += 40LL;
            }
            while ( v79 != v80 );
            v80 = v115[8].m128i_u64[0];
          }
          if ( v80 )
            j_j___libc_free_0(v80);
          if ( v115[6].m128i_i64[1] )
            j_j___libc_free_0(v115[6].m128i_u64[1]);
          if ( v115[5].m128i_i64[0] )
            j_j___libc_free_0(v115[5].m128i_u64[0]);
          if ( v115[3].m128i_i64[1] )
            j_j___libc_free_0(v115[3].m128i_u64[1]);
          if ( v115[2].m128i_i64[0] )
            j_j___libc_free_0(v115[2].m128i_u64[0]);
        }
        goto LABEL_70;
      }
      v5 = *(unsigned int *)(v3 + 48);
      if ( !*(_DWORD *)(v3 + 48) )
      {
        v6 = 0;
        v7 = 0;
        v8 = 0;
        goto LABEL_7;
      }
      v59 = 8 * v5;
      v60 = sub_22077B0(8 * v5);
      v61 = *(_QWORD *)(v3 + 40);
      v7 = (char *)v60;
      v62 = (char *)(v60 + v59);
      v63 = *(unsigned int *)(v3 + 48);
      if ( v61 + 8 * v63 != v61 )
      {
        n = v3;
        v64 = *(_QWORD **)(v3 + 40);
        v65 = (_QWORD *)(v61 + 8 * v63);
        v66 = v7;
        while ( 1 )
        {
          while ( 1 )
          {
            v67 = *(_QWORD *)(*v64 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v7 == v62 )
              break;
            if ( v7 )
              *(_QWORD *)v7 = v67;
            ++v64;
            v7 += 8;
            if ( v65 == v64 )
            {
LABEL_124:
              v3 = n;
              v8 = v66;
              v6 = (char *)(v7 - v66);
              v5 = v7 - v66;
              goto LABEL_7;
            }
          }
          v68 = v7 - v66;
          v69 = v68;
          v70 = v68 >> 3;
          if ( v68 >> 3 == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"vector::_M_realloc_insert");
          v71 = 1;
          if ( v70 )
            v71 = v68 >> 3;
          v72 = __CFADD__(v71, v70);
          v73 = v71 + v70;
          if ( v72 )
          {
            v89 = 0x7FFFFFFFFFFFFFF8LL;
          }
          else
          {
            if ( !v73 )
            {
              v74 = 0;
              v75 = 0;
              goto LABEL_119;
            }
            if ( v73 > 0xFFFFFFFFFFFFFFFLL )
              v73 = 0xFFFFFFFFFFFFFFFLL;
            v89 = 8 * v73;
          }
          v94 = v66;
          v99 = *(_QWORD *)(*v64 & 0xFFFFFFFFFFFFFFF8LL);
          v90 = sub_22077B0(v89);
          v67 = v99;
          v69 = v68;
          v66 = v94;
          v75 = (char *)v90;
          v74 = v90 + v89;
LABEL_119:
          if ( &v75[v69] )
            *(_QWORD *)&v75[v69] = v67;
          v7 = &v75[v69 + 8];
          if ( v69 > 0 )
          {
            v97 = v66;
            v82 = (char *)memmove(v75, v66, v69);
            v66 = v97;
            v75 = v82;
LABEL_147:
            v98 = v75;
            j_j___libc_free_0((unsigned __int64)v66);
            v75 = v98;
            goto LABEL_123;
          }
          if ( v66 )
            goto LABEL_147;
LABEL_123:
          ++v64;
          v62 = (char *)v74;
          v66 = v75;
          if ( v65 == v64 )
            goto LABEL_124;
        }
      }
      v8 = v7;
      v5 = 0;
      v6 = 0;
LABEL_7:
      memset(v115, 0, sizeof(v115));
      v9 = (char *)&v116;
      v115[0].m128i_i32[0] = *(_BYTE *)(v3 + 12) & 0xF;
      v115[0].m128i_i32[1] = (*(_BYTE *)(v3 + 12) >> 4) & 3;
      v115[0].m128i_i8[8] = (*(_BYTE *)(v3 + 12) & 0x40) != 0;
      v115[0].m128i_i8[9] = *(_BYTE *)(v3 + 12) >> 7;
      v115[0].m128i_i8[10] = *(_BYTE *)(v3 + 13) & 1;
      v115[0].m128i_i8[11] = (*(_BYTE *)(v3 + 13) & 2) != 0;
      v10 = *(_BYTE *)(v3 + 13);
      memset(&v115[2], 0, 24);
      v115[0].m128i_i32[3] = (v10 & 4) != 0;
      if ( v6 )
      {
        if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_179:
          sub_4261EA(v9, v6, v5);
        na = v5;
        v11 = sub_22077B0(v5);
        v5 = na;
        v12 = (void *)v11;
      }
      else
      {
        v12 = 0;
      }
      v13 = (__int64)v12 + v5;
      v115[2].m128i_i64[0] = (__int64)v12;
      v115[3].m128i_i64[0] = (__int64)v12 + v5;
      if ( v7 != v8 )
      {
        v6 = v8;
        memcpy(v12, v8, v5);
      }
      v115[2].m128i_i64[1] = v13;
      v14 = *(_QWORD *)(v3 + 80);
      if ( !v14 )
        goto LABEL_148;
      v15 = *(char **)v14;
      v16 = *(_QWORD *)(v14 + 8);
      v115[3].m128i_i64[1] = 0;
      v115[4] = 0u;
      v17 = v16 - (_QWORD)v15;
      if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_178;
      if ( v17 )
      {
        v6 = v15;
        v115[3].m128i_i64[1] = sub_22077B0(v17);
        v115[4].m128i_i64[1] = v115[3].m128i_i64[1] + v17;
        nb = v115[3].m128i_i64[1] + v17;
        memcpy((void *)v115[3].m128i_i64[1], v15, v17);
        v14 = *(_QWORD *)(v3 + 80);
        v115[4].m128i_i64[0] = nb;
        if ( !v14 )
          goto LABEL_149;
      }
      else
      {
LABEL_148:
        v115[3].m128i_i64[1] = 0;
        v115[4] = 0u;
        if ( !v14 )
          goto LABEL_149;
      }
      v18 = *(char **)(v14 + 24);
      v19 = *(_QWORD *)(v14 + 32);
      memset(&v115[5], 0, 24);
      v20 = v19 - (_QWORD)v18;
      if ( v20 > 0x7FFFFFFFFFFFFFF0LL )
        goto LABEL_178;
      if ( !v20 )
      {
LABEL_149:
        memset(&v115[5], 0, 24);
        v21 = *(_QWORD *)(v3 + 80);
        if ( !v21 )
          goto LABEL_150;
        goto LABEL_19;
      }
      v6 = v18;
      v115[5].m128i_i64[0] = sub_22077B0(v20);
      v115[6].m128i_i64[0] = v115[5].m128i_i64[0] + v20;
      nc = v115[5].m128i_i64[0] + v20;
      memcpy((void *)v115[5].m128i_i64[0], v18, v20);
      v115[5].m128i_i64[1] = nc;
      v21 = *(_QWORD *)(v3 + 80);
      if ( !v21 )
        goto LABEL_150;
LABEL_19:
      v22 = *(char **)(v21 + 48);
      v23 = *(_QWORD *)(v21 + 56);
      v115[6].m128i_i64[1] = 0;
      v115[7] = 0u;
      v24 = v23 - (_QWORD)v22;
      if ( v24 > 0x7FFFFFFFFFFFFFF0LL )
        goto LABEL_178;
      if ( v24 )
      {
        v6 = v22;
        v115[6].m128i_i64[1] = sub_22077B0(v24);
        v115[7].m128i_i64[1] = v115[6].m128i_i64[1] + v24;
        nd = v115[6].m128i_i64[1] + v24;
        memcpy((void *)v115[6].m128i_i64[1], v22, v24);
        v115[7].m128i_i64[0] = nd;
        v25 = *(_QWORD *)(v3 + 80);
        if ( !v25 )
          goto LABEL_151;
        goto LABEL_22;
      }
LABEL_150:
      v115[6].m128i_i64[1] = 0;
      v115[7] = 0u;
      v25 = *(_QWORD *)(v3 + 80);
      if ( !v25 )
        goto LABEL_151;
LABEL_22:
      v26 = *(const __m128i **)(v25 + 72);
      v27 = *(const __m128i **)(v25 + 80);
      memset(&v115[8], 0, 24);
      v95 = v27;
      if ( (unsigned __int64)((char *)v27 - (char *)v26) > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_178;
      if ( v27 == v26 )
      {
LABEL_151:
        memset(&v115[8], 0, 24);
        v34 = *(_QWORD *)(v3 + 80);
        if ( !v34 )
          goto LABEL_152;
        goto LABEL_37;
      }
      v9 = (char *)((char *)v27 - (char *)v26);
      ne = (char *)v27 - (char *)v26;
      v115[8].m128i_i64[0] = sub_22077B0((char *)v27 - (char *)v26);
      v28 = (__m128i *)v115[8].m128i_i64[0];
      v115[9].m128i_i64[0] = v115[8].m128i_i64[0] + ne;
      if ( v26 != v95 )
      {
        v29 = v26;
        v30 = v95;
        v96 = v8;
        do
        {
          if ( v28 )
          {
            *v28 = _mm_loadu_si128(v29);
            v5 = v29[1].m128i_i64[1] - v29[1].m128i_i64[0];
            v28[1].m128i_i64[0] = 0;
            v28[1].m128i_i64[1] = 0;
            v28[2].m128i_i64[0] = 0;
            if ( v5 )
            {
              v31 = v5;
              if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
                goto LABEL_179;
              v32 = (char *)sub_22077B0(v5);
            }
            else
            {
              v31 = 0;
              v32 = 0;
            }
            v28[1].m128i_i64[0] = (__int64)v32;
            v28[1].m128i_i64[1] = (__int64)v32;
            v28[2].m128i_i64[0] = (__int64)&v32[v31];
            v6 = (char *)v29[1].m128i_i64[0];
            v33 = v29[1].m128i_i64[1] - (_QWORD)v6;
            if ( (char *)v29[1].m128i_i64[1] != v6 )
              v32 = (char *)memmove(v32, v6, v29[1].m128i_i64[1] - (_QWORD)v6);
            v9 = &v32[v33];
            v28[1].m128i_i64[1] = (__int64)v9;
          }
          v29 = (const __m128i *)((char *)v29 + 40);
          v28 = (__m128i *)((char *)v28 + 40);
        }
        while ( v30 != v29 );
        v8 = v96;
      }
      v115[8].m128i_i64[1] = (__int64)v28;
      v34 = *(_QWORD *)(v3 + 80);
      if ( !v34 )
        goto LABEL_152;
LABEL_37:
      v35 = *(const __m128i **)(v34 + 104);
      v36 = *(const __m128i **)(v34 + 96);
      v115[9].m128i_i64[1] = 0;
      v115[10] = 0u;
      if ( (unsigned __int64)((char *)v35 - (char *)v36) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_178:
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      if ( v35 == v36 )
      {
LABEL_152:
        v115[9].m128i_i64[1] = 0;
        v37 = 0;
        v115[10].m128i_i64[1] = 0;
        goto LABEL_50;
      }
      v9 = (char *)((char *)v35 - (char *)v36);
      v115[9].m128i_i64[1] = sub_22077B0((char *)v35 - (char *)v36);
      v37 = (__m128i *)v115[9].m128i_i64[1];
      for ( v115[10].m128i_i64[1] = v115[9].m128i_i64[1] + (char *)v35 - (char *)v36;
            v35 != v36;
            v37 = (__m128i *)((char *)v37 + 40) )
      {
        if ( v37 )
        {
          *v37 = _mm_loadu_si128(v36);
          v5 = v36[1].m128i_i64[1] - v36[1].m128i_i64[0];
          v37[1].m128i_i64[0] = 0;
          v37[1].m128i_i64[1] = 0;
          v37[2].m128i_i64[0] = 0;
          if ( v5 )
          {
            v38 = v5;
            if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_179;
            v9 = (char *)v5;
            v39 = (char *)sub_22077B0(v5);
          }
          else
          {
            v38 = 0;
            v39 = 0;
          }
          v37[1].m128i_i64[0] = (__int64)v39;
          v37[1].m128i_i64[1] = (__int64)v39;
          v37[2].m128i_i64[0] = (__int64)&v39[v38];
          v6 = (char *)v36[1].m128i_i64[0];
          v40 = v36[1].m128i_i64[1] - (_QWORD)v6;
          if ( (char *)v36[1].m128i_i64[1] != v6 )
          {
            v9 = v39;
            v39 = (char *)memmove(v39, v6, v36[1].m128i_i64[1] - (_QWORD)v6);
          }
          v37[1].m128i_i64[1] = (__int64)&v39[v40];
        }
        v36 = (const __m128i *)((char *)v36 + 40);
      }
LABEL_50:
      v115[10].m128i_i64[0] = (__int64)v37;
      v41 = v113;
      if ( v113 == v114 )
      {
        sub_2628F50((unsigned __int64 *)&v112, v113, v115);
        v37 = (__m128i *)v115[10].m128i_i64[0];
        v83 = v115[9].m128i_i64[1];
        goto LABEL_155;
      }
      if ( !v113 )
      {
        v113 = (const __m128i *)176;
        v83 = v115[9].m128i_i64[1];
LABEL_155:
        if ( (__m128i *)v83 != v37 )
        {
          do
          {
            v84 = *(_QWORD *)(v83 + 16);
            if ( v84 )
              j_j___libc_free_0(v84);
            v83 += 40;
          }
          while ( (__m128i *)v83 != v37 );
          v37 = (__m128i *)v115[9].m128i_i64[1];
        }
        if ( v37 )
          j_j___libc_free_0((unsigned __int64)v37);
        goto LABEL_53;
      }
      v113->m128i_i64[0] = v115[0].m128i_i64[0];
      v41->m128i_i64[1] = v115[0].m128i_i64[1];
      v41[1] = _mm_loadu_si128(&v115[1]);
      v41[2] = v115[2];
      v41[3].m128i_i64[0] = v115[3].m128i_i64[0];
      memset(&v115[2], 0, 24);
      v41[3].m128i_i64[1] = v115[3].m128i_i64[1];
      v41[4] = v115[4];
      v115[4] = 0u;
      v115[3].m128i_i64[1] = 0;
      v41[5] = v115[5];
      v41[6].m128i_i64[0] = v115[6].m128i_i64[0];
      memset(&v115[5], 0, 24);
      v41[6].m128i_i64[1] = v115[6].m128i_i64[1];
      v41[7] = v115[7];
      v115[7] = 0u;
      v115[6].m128i_i64[1] = 0;
      v41[8] = v115[8];
      v41[9].m128i_i64[0] = v115[9].m128i_i64[0];
      memset(&v115[8], 0, 24);
      v41[9].m128i_i64[1] = v115[9].m128i_i64[1];
      v41[10].m128i_i64[0] = v115[10].m128i_i64[0];
      v113 += 11;
      v41[10].m128i_i64[1] = v115[10].m128i_i64[1];
LABEL_53:
      v42 = v115[8].m128i_i64[1];
      v43 = v115[8].m128i_u64[0];
      if ( v115[8].m128i_i64[1] != v115[8].m128i_i64[0] )
      {
        do
        {
          v44 = *(_QWORD *)(v43 + 16);
          if ( v44 )
            j_j___libc_free_0(v44);
          v43 += 40LL;
        }
        while ( v42 != v43 );
        v43 = v115[8].m128i_u64[0];
      }
      if ( v43 )
        j_j___libc_free_0(v43);
      if ( v115[6].m128i_i64[1] )
        j_j___libc_free_0(v115[6].m128i_u64[1]);
      if ( v115[5].m128i_i64[0] )
        j_j___libc_free_0(v115[5].m128i_u64[0]);
      if ( v115[3].m128i_i64[1] )
        j_j___libc_free_0(v115[3].m128i_u64[1]);
      if ( v115[2].m128i_i64[0] )
        j_j___libc_free_0(v115[2].m128i_u64[0]);
      if ( v8 )
        j_j___libc_free_0((unsigned __int64)v8);
LABEL_70:
      if ( v106 == ++v107 )
      {
        v45 = v112;
        if ( v113 != v112 )
        {
          v46 = v93[4];
          if ( v46 )
          {
            v47 = v111;
            do
            {
              *--v47 = v46 % 0xA + 48;
              v85 = v46;
              v46 /= 0xAu;
            }
            while ( v85 > 9 );
          }
          else
          {
            v110 = 48;
            v47 = &v110;
          }
          v115[0].m128i_i64[0] = (__int64)v115[1].m128i_i64;
          sub_261A960(v115[0].m128i_i64, v47, (__int64)v111);
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                 a1,
                 v115[0].m128i_i64[0],
                 1,
                 0,
                 &v108,
                 &v109) )
          {
            sub_262F0A0(a1, (__int64 *)&v112);
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v109);
          }
          if ( (__m128i *)v115[0].m128i_i64[0] != &v115[1] )
            j_j___libc_free_0(v115[0].m128i_u64[0]);
          v48 = v113;
          v45 = v112;
          if ( v113 != v112 )
          {
            do
            {
              v49 = v45[10].m128i_i64[0];
              v50 = v45[9].m128i_u64[1];
              if ( v49 != v50 )
              {
                do
                {
                  v51 = *(_QWORD *)(v50 + 16);
                  if ( v51 )
                    j_j___libc_free_0(v51);
                  v50 += 40LL;
                }
                while ( v49 != v50 );
                v50 = v45[9].m128i_u64[1];
              }
              if ( v50 )
                j_j___libc_free_0(v50);
              v52 = v45[8].m128i_i64[1];
              v53 = v45[8].m128i_u64[0];
              if ( v52 != v53 )
              {
                do
                {
                  v54 = *(_QWORD *)(v53 + 16);
                  if ( v54 )
                    j_j___libc_free_0(v54);
                  v53 += 40LL;
                }
                while ( v52 != v53 );
                v53 = v45[8].m128i_u64[0];
              }
              if ( v53 )
                j_j___libc_free_0(v53);
              v55 = v45[6].m128i_u64[1];
              if ( v55 )
                j_j___libc_free_0(v55);
              v56 = v45[5].m128i_u64[0];
              if ( v56 )
                j_j___libc_free_0(v56);
              v57 = v45[3].m128i_u64[1];
              if ( v57 )
                j_j___libc_free_0(v57);
              v58 = v45[2].m128i_u64[0];
              if ( v58 )
                j_j___libc_free_0(v58);
              v45 += 11;
            }
            while ( v48 != v45 );
            v45 = v112;
          }
        }
        if ( v45 )
          j_j___libc_free_0((unsigned __int64)v45);
        goto LABEL_105;
      }
    }
  }
  return result;
}
