// Function: sub_C908F0
// Address: 0xc908f0
//
__int64 __fastcall sub_C908F0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r13
  __m128i *v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // rcx
  bool v9; // cf
  unsigned int v10; // eax
  unsigned __int64 v11; // rbx
  __m128i *v12; // r15
  bool v13; // cc
  unsigned __int64 v14; // rbx
  __m128i *v15; // r10
  __m128i *v16; // r12
  __int64 v17; // r14
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rax
  __m128i *v20; // r8
  bool v21; // cf
  unsigned int v22; // eax
  __m128i v23; // xmm1
  __m128i *v24; // rax
  size_t v25; // rax
  __m128i *v26; // r13
  __m128i v27; // xmm2
  __int64 v28; // rax
  _BYTE *v29; // rax
  __m128i *v30; // rax
  __m128i *v31; // rdi
  size_t v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // r8
  size_t v35; // rbx
  size_t v36; // r12
  size_t v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // rbx
  unsigned int v40; // eax
  unsigned __int64 v41; // rax
  size_t v42; // r15
  size_t v43; // rcx
  size_t v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // r15
  size_t v47; // rdx
  size_t v48; // rdx
  unsigned __int64 v49; // r15
  size_t v50; // r9
  size_t v51; // rdx
  unsigned int v52; // eax
  __int64 v53; // r15
  __m128i *v54; // r8
  unsigned __int64 v55; // rax
  size_t v56; // rbx
  size_t v57; // r15
  size_t v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // rbx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // r15
  const __m128i *v67; // r12
  __int64 v68; // r14
  __int64 v69; // rsi
  size_t v70; // rcx
  __m128i v71; // xmm0
  __m128i *v72; // rdx
  __m128i v73; // xmm5
  __m128i *v74; // r8
  __int64 v75; // r14
  __m128i *v76; // r15
  size_t v77; // rax
  __m128i v78; // xmm6
  __int64 v79; // rax
  _BYTE *v80; // rax
  __m128i v81; // xmm7
  __int64 v82; // r12
  __m128i v83; // xmm5
  __m128i *v84; // rax
  size_t v85; // rdx
  size_t v86; // r12
  size_t v87; // r10
  size_t v88; // rdx
  unsigned int v89; // eax
  size_t v90; // r10
  size_t v91; // rcx
  size_t v92; // rdx
  signed __int64 v93; // rax
  __int64 v94; // r12
  size_t v95; // rbx
  size_t v96; // r12
  size_t v97; // rdx
  unsigned int v98; // eax
  __int64 v99; // rbx
  __m128i *v100; // [rsp+8h] [rbp-C8h]
  __m128i *dest; // [rsp+10h] [rbp-C0h]
  __m128i *v102; // [rsp+18h] [rbp-B8h]
  __int64 v103; // [rsp+20h] [rbp-B0h]
  __m128i *v104; // [rsp+28h] [rbp-A8h]
  __m128i *v105; // [rsp+28h] [rbp-A8h]
  size_t v106; // [rsp+28h] [rbp-A8h]
  size_t v107; // [rsp+28h] [rbp-A8h]
  __m128i *v108; // [rsp+30h] [rbp-A0h]
  __m128i *v109; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v110; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v111; // [rsp+30h] [rbp-A0h]
  size_t v112; // [rsp+30h] [rbp-A0h]
  __m128i *v113; // [rsp+38h] [rbp-98h]
  size_t v114; // [rsp+38h] [rbp-98h]
  __m128i *v115; // [rsp+38h] [rbp-98h]
  __m128i *v116; // [rsp+38h] [rbp-98h]
  size_t v117; // [rsp+38h] [rbp-98h]
  __m128i *v118; // [rsp+38h] [rbp-98h]
  unsigned __int64 v119; // [rsp+38h] [rbp-98h]
  unsigned __int64 v120; // [rsp+38h] [rbp-98h]
  unsigned __int64 v121; // [rsp+38h] [rbp-98h]
  unsigned __int64 v122; // [rsp+38h] [rbp-98h]
  __m128i v123; // [rsp+40h] [rbp-90h] BYREF
  __m128i *v124; // [rsp+50h] [rbp-80h]
  size_t v125; // [rsp+58h] [rbp-78h]
  __m128i v126; // [rsp+60h] [rbp-70h] BYREF
  __m128i v127; // [rsp+70h] [rbp-60h] BYREF
  __m128i *v128; // [rsp+80h] [rbp-50h]
  size_t n; // [rsp+88h] [rbp-48h]
  _OWORD src[4]; // [rsp+90h] [rbp-40h] BYREF

  result = (char *)a2 - (char *)a1;
  v103 = a3;
  if ( (char *)a2 - (char *)a1 <= 768 )
    return result;
  v4 = (__int64)a1;
  v5 = a2;
  if ( !a3 )
  {
    v54 = a2;
    goto LABEL_100;
  }
  v100 = a1 + 3;
  dest = a1 + 5;
  while ( 2 )
  {
    v6 = *(_QWORD *)(v4 + 48);
    --v103;
    v7 = v4
       + 16
       * (((0xAAAAAAAAAAAAAAABLL * (result >> 4)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 4)) >> 1));
    v8 = *(_QWORD *)v7;
    v9 = v6 < *(_QWORD *)v7;
    if ( v6 == *(_QWORD *)v7
      && (v55 = *(_QWORD *)(v7 + 8), v9 = *(_QWORD *)(v4 + 56) < v55, *(_QWORD *)(v4 + 56) == v55) )
    {
      v56 = *(_QWORD *)(v4 + 72);
      v57 = *(_QWORD *)(v7 + 24);
      v58 = v57;
      if ( v56 <= v57 )
        v58 = *(_QWORD *)(v4 + 72);
      if ( v58
        && (v110 = *(_QWORD *)v7,
            v119 = v7,
            v59 = memcmp(*(const void **)(v4 + 64), *(const void **)(v7 + 16), v58),
            v7 = v119,
            v8 = v110,
            v59) )
      {
        v10 = v59 >> 31;
      }
      else
      {
        v60 = v56 - v57;
        if ( v60 >= 0x80000000LL )
        {
          v11 = v5[-3].m128i_u64[0];
          v12 = v5 - 3;
          goto LABEL_81;
        }
        if ( v60 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v11 = v5[-3].m128i_u64[0];
          v12 = v5 - 3;
LABEL_7:
          if ( v11 == v8 )
          {
            v63 = v5[-3].m128i_u64[1];
            if ( *(_QWORD *)(v7 + 8) != v63 )
            {
              if ( *(_QWORD *)(v7 + 8) < v63 )
                goto LABEL_87;
              v13 = v11 <= v6;
              if ( v11 != v6 )
              {
LABEL_10:
                if ( !v13 )
                  goto LABEL_11;
                goto LABEL_96;
              }
LABEL_94:
              v64 = v5[-3].m128i_u64[1];
              if ( *(_QWORD *)(v4 + 56) == v64 )
              {
                v35 = *(_QWORD *)(v4 + 72);
                v36 = v5[-2].m128i_u64[1];
                v37 = v36;
                if ( v35 <= v36 )
                  v37 = *(_QWORD *)(v4 + 72);
                if ( v37 && (v38 = memcmp(*(const void **)(v4 + 64), (const void *)v5[-2].m128i_i64[0], v37)) != 0 )
                {
                  v40 = v38 >> 31;
                }
                else
                {
                  v39 = v35 - v36;
                  if ( v39 >= 0x80000000LL )
                    goto LABEL_96;
                  if ( v39 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                    goto LABEL_11;
                  LOBYTE(v40) = (int)v39 < 0;
                }
                if ( (_BYTE)v40 )
                  goto LABEL_11;
              }
              else if ( *(_QWORD *)(v4 + 56) < v64 )
              {
                goto LABEL_11;
              }
              goto LABEL_96;
            }
            v90 = *(_QWORD *)(v7 + 24);
            v91 = v5[-2].m128i_u64[1];
            v92 = v91;
            if ( v90 <= v91 )
              v92 = *(_QWORD *)(v7 + 24);
            if ( !v92
              || (v107 = v5[-2].m128i_u64[1],
                  v112 = *(_QWORD *)(v7 + 24),
                  v121 = v7,
                  LODWORD(v93) = memcmp(*(const void **)(v7 + 16), (const void *)v5[-2].m128i_i64[0], v92),
                  v7 = v121,
                  v90 = v112,
                  v91 = v107,
                  !(_DWORD)v93) )
            {
              v93 = v90 - v91;
              if ( (__int64)(v90 - v91) >= 0x80000000LL )
                goto LABEL_9;
              if ( v93 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                goto LABEL_87;
            }
            if ( (int)v93 < 0 )
              goto LABEL_87;
          }
          else if ( v11 > v8 )
          {
LABEL_87:
            sub_C90720((__m128i *)v4, (__m128i *)v7);
            goto LABEL_12;
          }
LABEL_9:
          v13 = v11 <= v6;
          if ( v11 != v6 )
            goto LABEL_10;
          goto LABEL_94;
        }
        LOBYTE(v10) = (int)v60 < 0;
      }
    }
    else
    {
      LOBYTE(v10) = v9;
    }
    v11 = v5[-3].m128i_u64[0];
    v12 = v5 - 3;
    if ( (_BYTE)v10 )
      goto LABEL_7;
LABEL_81:
    if ( v11 != v6 )
    {
      LOBYTE(v61) = v11 > v6;
      goto LABEL_83;
    }
    v61 = v5[-3].m128i_u64[1];
    if ( *(_QWORD *)(v4 + 56) != v61 )
    {
      LOBYTE(v61) = *(_QWORD *)(v4 + 56) < v61;
      goto LABEL_83;
    }
    v86 = *(_QWORD *)(v4 + 72);
    v87 = v5[-2].m128i_u64[1];
    v88 = v87;
    if ( v86 <= v87 )
      v88 = *(_QWORD *)(v4 + 72);
    if ( v88 )
    {
      v106 = v5[-2].m128i_u64[1];
      v111 = v8;
      v120 = v7;
      v89 = memcmp(*(const void **)(v4 + 64), (const void *)v5[-2].m128i_i64[0], v88);
      v7 = v120;
      v8 = v111;
      v87 = v106;
      if ( v89 )
      {
        LODWORD(v61) = v89 >> 31;
        goto LABEL_83;
      }
    }
    v94 = v86 - v87;
    if ( v94 >= 0x80000000LL )
    {
LABEL_84:
      if ( v11 == v8 )
      {
        v62 = v5[-3].m128i_u64[1];
        if ( *(_QWORD *)(v7 + 8) == v62 )
        {
          v95 = *(_QWORD *)(v7 + 24);
          v96 = v5[-2].m128i_u64[1];
          v97 = v96;
          if ( v95 <= v96 )
            v97 = *(_QWORD *)(v7 + 24);
          if ( v97
            && (v122 = v7,
                v98 = memcmp(*(const void **)(v7 + 16), (const void *)v5[-2].m128i_i64[0], v97),
                v7 = v122,
                v98) )
          {
            LODWORD(v62) = v98 >> 31;
          }
          else
          {
            v99 = v95 - v96;
            if ( v99 >= 0x80000000LL )
              goto LABEL_87;
            if ( v99 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_11;
            LOBYTE(v62) = (int)v99 < 0;
          }
        }
        else
        {
          LOBYTE(v62) = *(_QWORD *)(v7 + 8) < v62;
        }
      }
      else
      {
        LOBYTE(v62) = v11 > v8;
      }
      if ( !(_BYTE)v62 )
        goto LABEL_87;
LABEL_11:
      sub_C90720((__m128i *)v4, v12);
      goto LABEL_12;
    }
    if ( v94 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      LOBYTE(v61) = (int)v94 < 0;
LABEL_83:
      if ( !(_BYTE)v61 )
        goto LABEL_84;
    }
LABEL_96:
    sub_C90720((__m128i *)v4, v100);
LABEL_12:
    v14 = (unsigned __int64)v5;
    v15 = (__m128i *)src;
    v102 = v5;
    v16 = dest;
    v17 = v4;
    v18 = *(_QWORD *)v4;
    while ( 1 )
    {
      v20 = v16 - 2;
      if ( v18 != v16[-2].m128i_i64[0] )
      {
        LOBYTE(v19) = v18 > v16[-2].m128i_i64[0];
        goto LABEL_14;
      }
      v19 = *(_QWORD *)(v17 + 8);
      if ( v16[-2].m128i_i64[1] != v19 )
      {
        LOBYTE(v19) = v16[-2].m128i_i64[1] < v19;
        goto LABEL_14;
      }
      v49 = v16[-1].m128i_u64[1];
      v50 = *(_QWORD *)(v17 + 24);
      v51 = v50;
      if ( v49 <= v50 )
        v51 = v16[-1].m128i_u64[1];
      if ( v51 )
      {
        v109 = v15;
        v117 = *(_QWORD *)(v17 + 24);
        v52 = memcmp((const void *)v16[-1].m128i_i64[0], *(const void **)(v17 + 16), v51);
        v50 = v117;
        v15 = v109;
        v20 = v16 - 2;
        if ( v52 )
        {
          LODWORD(v19) = v52 >> 31;
          goto LABEL_14;
        }
      }
      v53 = v49 - v50;
      if ( v53 >= 0x80000000LL )
        goto LABEL_19;
      if ( v53 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        break;
LABEL_15:
      v16 += 3;
    }
    LODWORD(v19) = (unsigned int)v53 >> 31;
LABEL_14:
    if ( (_BYTE)v19 )
      goto LABEL_15;
    while ( 1 )
    {
LABEL_19:
      while ( 1 )
      {
        v14 -= 48LL;
        v21 = v18 < *(_QWORD *)v14;
        if ( v18 == *(_QWORD *)v14 )
        {
          v41 = *(_QWORD *)(v14 + 8);
          v21 = *(_QWORD *)(v17 + 8) < v41;
          if ( *(_QWORD *)(v17 + 8) == v41 )
            break;
        }
        LOBYTE(v22) = v21;
LABEL_21:
        if ( !(_BYTE)v22 )
          goto LABEL_22;
      }
      v42 = *(_QWORD *)(v17 + 24);
      v43 = *(_QWORD *)(v14 + 24);
      v44 = v43;
      if ( v42 <= v43 )
        v44 = *(_QWORD *)(v17 + 24);
      if ( v44 )
      {
        v104 = v15;
        v108 = v20;
        v114 = *(_QWORD *)(v14 + 24);
        v45 = memcmp(*(const void **)(v17 + 16), *(const void **)(v14 + 16), v44);
        v43 = v114;
        v20 = v108;
        v15 = v104;
        if ( v45 )
        {
          v22 = v45 >> 31;
          goto LABEL_21;
        }
      }
      v46 = v42 - v43;
      if ( v46 >= 0x80000000LL )
        break;
      if ( v46 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v22 = (unsigned int)v46 >> 31;
        goto LABEL_21;
      }
    }
LABEL_22:
    if ( v14 > (unsigned __int64)v20 )
    {
      v23 = _mm_loadu_si128(v16 - 2);
      v24 = (__m128i *)v16[-1].m128i_i64[0];
      v128 = v15;
      v127 = v23;
      if ( v16 == v24 )
      {
        src[0] = _mm_loadu_si128(v16);
      }
      else
      {
        v128 = v24;
        *(_QWORD *)&src[0] = v16->m128i_i64[0];
      }
      v25 = v16[-1].m128i_u64[1];
      v16[-1].m128i_i64[0] = (__int64)v16;
      v26 = (__m128i *)(v14 + 32);
      v16[-1].m128i_i64[1] = 0;
      v16->m128i_i8[0] = 0;
      v27 = _mm_loadu_si128((const __m128i *)v14);
      n = v25;
      v16[-2] = v27;
      v28 = *(_QWORD *)(v14 + 16);
      if ( v28 == v14 + 32 )
      {
        v48 = *(_QWORD *)(v14 + 24);
        if ( v48 )
        {
          if ( v48 == 1 )
          {
            v16->m128i_i8[0] = *(_BYTE *)(v14 + 32);
            v48 = *(_QWORD *)(v14 + 24);
          }
          else
          {
            v116 = v15;
            memcpy(v16, (const void *)(v14 + 32), v48);
            v48 = *(_QWORD *)(v14 + 24);
            v15 = v116;
          }
        }
        v16[-1].m128i_i64[1] = v48;
        v16->m128i_i8[v48] = 0;
        v29 = *(_BYTE **)(v14 + 16);
      }
      else
      {
        v16[-1].m128i_i64[0] = v28;
        v16[-1].m128i_i64[1] = *(_QWORD *)(v14 + 24);
        v16->m128i_i64[0] = *(_QWORD *)(v14 + 32);
        v29 = (_BYTE *)(v14 + 32);
        *(_QWORD *)(v14 + 16) = v26;
      }
      *(_QWORD *)(v14 + 24) = 0;
      *v29 = 0;
      v30 = v128;
      v31 = *(__m128i **)(v14 + 16);
      *(__m128i *)v14 = _mm_load_si128(&v127);
      if ( v30 == v15 )
      {
        v47 = n;
        if ( n )
        {
          if ( n == 1 )
          {
            v31->m128i_i8[0] = src[0];
            v47 = n;
            v31 = *(__m128i **)(v14 + 16);
          }
          else
          {
            v115 = v15;
            memcpy(v31, v15, n);
            v47 = n;
            v31 = *(__m128i **)(v14 + 16);
            v15 = v115;
          }
        }
        *(_QWORD *)(v14 + 24) = v47;
        v31->m128i_i8[v47] = 0;
        v31 = v128;
        goto LABEL_31;
      }
      v32 = n;
      v33 = *(_QWORD *)&src[0];
      if ( v31 == v26 )
      {
        *(_QWORD *)(v14 + 16) = v30;
        *(_QWORD *)(v14 + 24) = v32;
        *(_QWORD *)(v14 + 32) = v33;
      }
      else
      {
        v34 = *(_QWORD *)(v14 + 32);
        *(_QWORD *)(v14 + 16) = v30;
        *(_QWORD *)(v14 + 24) = v32;
        *(_QWORD *)(v14 + 32) = v33;
        if ( v31 )
        {
          v128 = v31;
          *(_QWORD *)&src[0] = v34;
LABEL_31:
          n = 0;
          v31->m128i_i8[0] = 0;
          if ( v128 != v15 )
          {
            v113 = v15;
            j_j___libc_free_0(v128, *(_QWORD *)&src[0] + 1LL);
            v15 = v113;
          }
          v18 = *(_QWORD *)v17;
          goto LABEL_15;
        }
      }
      v128 = v15;
      v31 = v15;
      goto LABEL_31;
    }
    v4 = v17;
    v118 = v20;
    sub_C908F0(v20, v102, v103);
    v54 = v118;
    result = (__int64)v118->m128i_i64 - v17;
    if ( (__int64)v118->m128i_i64 - v17 <= 768 )
      return result;
    if ( v103 )
    {
      v5 = v118;
      continue;
    }
    break;
  }
LABEL_100:
  v105 = v54;
  v65 = 0xAAAAAAAAAAAAAAABLL * (result >> 4);
  v66 = (__int64)(v65 - 2) >> 1;
  v67 = (const __m128i *)(v4 + 16 * (v66 + ((v65 - 2) & 0xFFFFFFFFFFFFFFFELL)) + 32);
  v68 = v65;
  while ( 2 )
  {
    v71 = _mm_loadu_si128(v67 - 2);
    v72 = (__m128i *)v67[-1].m128i_i64[0];
    v123 = v71;
    if ( v72 == v67 )
    {
      v73 = _mm_loadu_si128(v67);
      v67->m128i_i8[0] = 0;
      v70 = v67[-1].m128i_u64[1];
      v127 = v71;
      v67[-1].m128i_i64[1] = 0;
      v128 = (__m128i *)src;
      v126 = v73;
LABEL_111:
      src[0] = _mm_load_si128(&v126);
    }
    else
    {
      v69 = v67->m128i_i64[0];
      v124 = v72;
      v70 = v67[-1].m128i_u64[1];
      v67[-1].m128i_i64[0] = (__int64)v67;
      v126.m128i_i64[0] = v69;
      v67[-1].m128i_i64[1] = 0;
      v67->m128i_i8[0] = 0;
      v128 = (__m128i *)src;
      v127 = v71;
      if ( v72 == &v126 )
        goto LABEL_111;
      v128 = v72;
      *(_QWORD *)&src[0] = v69;
    }
    n = v70;
    v124 = &v126;
    v125 = 0;
    v126.m128i_i8[0] = 0;
    sub_C8E1E0(v4, v66, v68, &v127);
    if ( v128 != (__m128i *)src )
      j_j___libc_free_0(v128, *(_QWORD *)&src[0] + 1LL);
    if ( v66 )
    {
      --v66;
      if ( v124 != &v126 )
        j_j___libc_free_0(v124, v126.m128i_i64[0] + 1);
      v67 -= 3;
      continue;
    }
    break;
  }
  v74 = v105;
  if ( v124 != &v126 )
  {
    j_j___libc_free_0(v124, v126.m128i_i64[0] + 1);
    v74 = v105;
  }
  v75 = v4 + 32;
  v76 = v74 - 1;
  do
  {
    v83 = _mm_loadu_si128(v76 - 2);
    v84 = (__m128i *)v76[-1].m128i_i64[0];
    v124 = &v126;
    v123 = v83;
    if ( v84 == v76 )
    {
      v126 = _mm_loadu_si128(v76);
    }
    else
    {
      v124 = v84;
      v126.m128i_i64[0] = v76->m128i_i64[0];
    }
    v77 = v76[-1].m128i_u64[1];
    v76[-1].m128i_i64[0] = (__int64)v76;
    v76[-1].m128i_i64[1] = 0;
    v76->m128i_i8[0] = 0;
    v78 = _mm_loadu_si128((const __m128i *)v4);
    v125 = v77;
    v76[-2] = v78;
    v79 = *(_QWORD *)(v4 + 16);
    if ( v79 == v75 )
    {
      v85 = *(_QWORD *)(v4 + 24);
      if ( v85 )
      {
        if ( v85 == 1 )
          v76->m128i_i8[0] = *(_BYTE *)(v4 + 32);
        else
          memcpy(v76, (const void *)(v4 + 32), v85);
        v85 = *(_QWORD *)(v4 + 24);
      }
      v76[-1].m128i_i64[1] = v85;
      v76->m128i_i8[v85] = 0;
      v80 = *(_BYTE **)(v4 + 16);
    }
    else
    {
      v76[-1].m128i_i64[0] = v79;
      v76[-1].m128i_i64[1] = *(_QWORD *)(v4 + 24);
      v76->m128i_i64[0] = *(_QWORD *)(v4 + 32);
      v80 = (_BYTE *)(v4 + 32);
      *(_QWORD *)(v4 + 16) = v75;
    }
    *(_QWORD *)(v4 + 24) = 0;
    *v80 = 0;
    v81 = _mm_load_si128(&v123);
    v128 = (__m128i *)src;
    v127 = v81;
    if ( v124 == &v126 )
    {
      src[0] = _mm_load_si128(&v126);
    }
    else
    {
      v128 = v124;
      *(_QWORD *)&src[0] = v126.m128i_i64[0];
    }
    v82 = (__int64)v76[-2].m128i_i64 - v4;
    v124 = &v126;
    n = v125;
    v125 = 0;
    v126.m128i_i8[0] = 0;
    result = sub_C8E1E0(v4, 0, 0xAAAAAAAAAAAAAAABLL * (v82 >> 4), &v127);
    if ( v128 != (__m128i *)src )
      result = j_j___libc_free_0(v128, *(_QWORD *)&src[0] + 1LL);
    if ( v124 != &v126 )
      result = j_j___libc_free_0(v124, v126.m128i_i64[0] + 1);
    v76 -= 3;
  }
  while ( v82 > 48 );
  return result;
}
