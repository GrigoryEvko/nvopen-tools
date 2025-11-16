// Function: sub_16CDB80
// Address: 0x16cdb80
//
__int64 __fastcall sub_16CDB80(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v4; // r8
  __int64 v5; // r11
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  __m128i *v10; // rcx
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  bool v14; // cf
  __m128i *v15; // r9
  __m128i *v16; // r12
  __int64 v17; // rax
  __m128i *v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  size_t v21; // r9
  size_t v22; // r11
  size_t v23; // rdx
  int v24; // eax
  bool v25; // sf
  __int64 v26; // r9
  size_t v27; // rdx
  _BYTE *v28; // rdi
  __int64 v29; // r13
  __m128i *v30; // rdx
  size_t v31; // r10
  __int64 v32; // rbx
  __int64 v33; // r14
  __int64 v34; // r13
  unsigned int v35; // eax
  __m128i *v36; // rcx
  __m128i *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r12
  bool v41; // cf
  size_t v42; // rcx
  size_t v43; // rdx
  unsigned int v44; // eax
  __int64 v45; // rcx
  __int64 result; // rax
  __m128i *v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // rcx
  size_t v50; // rdx
  __int64 v51; // rax
  __m128i *v52; // rdi
  __int64 v53; // r13
  const __m128i *v54; // r12
  const __m128i *v55; // rsi
  __int64 v56; // rdx
  size_t v57; // rdx
  __m128i *v58; // [rsp+0h] [rbp-90h]
  size_t v59; // [rsp+8h] [rbp-88h]
  size_t v60; // [rsp+10h] [rbp-80h]
  const __m128i *v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  const __m128i *v63; // [rsp+18h] [rbp-78h]
  __m128i *v65; // [rsp+20h] [rbp-70h]
  __int64 v66; // [rsp+20h] [rbp-70h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  size_t v69; // [rsp+28h] [rbp-68h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __m128i v71; // [rsp+30h] [rbp-60h] BYREF
  void *s2; // [rsp+40h] [rbp-50h]
  size_t v73; // [rsp+48h] [rbp-48h]
  _OWORD src[4]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a1;
  v5 = a2;
  v6 = (a3 - 1) / 2;
  v7 = a1 + 48 * a2;
  if ( a2 >= v6 )
  {
    v15 = (__m128i *)(v7 + 32);
    v29 = a2;
    goto LABEL_27;
  }
  v8 = a2;
  v61 = a4;
  v10 = (__m128i *)(v7 + 32);
  while ( 1 )
  {
    v11 = 2 * (v8 + 1);
    v12 = 96 * (v8 + 1);
    v13 = a1 + v12 - 48;
    v7 = a1 + v12;
    v14 = *(_QWORD *)v7 < *(_QWORD *)v13;
    if ( *(_QWORD *)v7 != *(_QWORD *)v13
      || (v20 = *(_QWORD *)(v13 + 8), v14 = *(_QWORD *)(v7 + 8) < v20, *(_QWORD *)(v7 + 8) != v20) )
    {
      if ( !v14 )
        goto LABEL_6;
LABEL_5:
      --v11;
      v7 = a1 + 48 * v11;
      goto LABEL_6;
    }
    v21 = *(_QWORD *)(v7 + 24);
    v22 = *(_QWORD *)(v13 + 24);
    v23 = v22;
    if ( v21 <= v22 )
      v23 = *(_QWORD *)(v7 + 24);
    if ( !v23 )
      goto LABEL_17;
    v60 = *(_QWORD *)(v7 + 24);
    v58 = v10;
    v59 = *(_QWORD *)(v13 + 24);
    v24 = memcmp(*(const void **)(v7 + 16), *(const void **)(v13 + 16), v23);
    v21 = v60;
    v22 = v59;
    v25 = v24 < 0;
    v10 = v58;
    if ( !v24 )
    {
LABEL_17:
      v26 = v21 - v22;
      if ( v26 >= 0x80000000LL )
        goto LABEL_6;
      if ( v26 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_5;
      v25 = (int)v26 < 0;
    }
    if ( v25 )
      goto LABEL_5;
LABEL_6:
    v15 = (__m128i *)(v7 + 32);
    v16 = (__m128i *)(a1 + 48 * v8);
    *v16 = _mm_loadu_si128((const __m128i *)v7);
    v17 = *(_QWORD *)(v7 + 16);
    v18 = (__m128i *)v16[1].m128i_i64[0];
    if ( v17 == v7 + 32 )
      break;
    if ( v18 == v10 )
    {
      v16[1].m128i_i64[0] = v17;
      v16[1].m128i_i64[1] = *(_QWORD *)(v7 + 24);
      v16[2].m128i_i64[0] = *(_QWORD *)(v7 + 32);
    }
    else
    {
      v16[1].m128i_i64[0] = v17;
      v19 = v16[2].m128i_i64[0];
      v16[1].m128i_i64[1] = *(_QWORD *)(v7 + 24);
      v16[2].m128i_i64[0] = *(_QWORD *)(v7 + 32);
      if ( v18 )
      {
        *(_QWORD *)(v7 + 16) = v18;
        *(_QWORD *)(v7 + 32) = v19;
        goto LABEL_10;
      }
    }
    *(_QWORD *)(v7 + 16) = v15;
    v18 = (__m128i *)(v7 + 32);
LABEL_10:
    *(_QWORD *)(v7 + 24) = 0;
    v18->m128i_i8[0] = 0;
    if ( v11 >= v6 )
      goto LABEL_26;
LABEL_11:
    v10 = v15;
    v8 = v11;
  }
  v27 = *(_QWORD *)(v7 + 24);
  if ( v27 )
  {
    if ( v27 == 1 )
    {
      v18->m128i_i8[0] = *(_BYTE *)(v7 + 32);
      v27 = *(_QWORD *)(v7 + 24);
      v18 = (__m128i *)v16[1].m128i_i64[0];
    }
    else
    {
      memcpy(v18, (const void *)(v7 + 32), v27);
      v27 = *(_QWORD *)(v7 + 24);
      v18 = (__m128i *)v16[1].m128i_i64[0];
      v15 = (__m128i *)(v7 + 32);
    }
  }
  v16[1].m128i_i64[1] = v27;
  v18->m128i_i8[v27] = 0;
  v28 = *(_BYTE **)(v7 + 16);
  *(_QWORD *)(v7 + 24) = 0;
  *v28 = 0;
  if ( v11 < v6 )
    goto LABEL_11;
LABEL_26:
  v5 = a2;
  a4 = v61;
  v4 = a1;
  v29 = v11;
LABEL_27:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v29 )
  {
    v51 = v29 + 1;
    v52 = *(__m128i **)(v7 + 16);
    v53 = 2 * (v29 + 1);
    v54 = (const __m128i *)(v4 + 16 * (v53 + 4 * v51) - 48);
    *(__m128i *)v7 = _mm_loadu_si128(v54);
    v55 = (const __m128i *)v54[1].m128i_i64[0];
    if ( v55 == &v54[2] )
    {
      v57 = v54[1].m128i_u64[1];
      if ( v57 )
      {
        if ( v57 == 1 )
        {
          v52->m128i_i8[0] = v54[2].m128i_i8[0];
          v57 = v54[1].m128i_u64[1];
          v52 = *(__m128i **)(v7 + 16);
        }
        else
        {
          v63 = a4;
          v67 = v5;
          v70 = v4;
          memcpy(v52, v55, v57);
          v57 = v54[1].m128i_u64[1];
          v52 = *(__m128i **)(v7 + 16);
          a4 = v63;
          v5 = v67;
          v4 = v70;
        }
      }
      *(_QWORD *)(v7 + 24) = v57;
      v52->m128i_i8[v57] = 0;
      v52 = (__m128i *)v54[1].m128i_i64[0];
      goto LABEL_83;
    }
    if ( v15 == v52 )
    {
      *(_QWORD *)(v7 + 16) = v55;
      *(_QWORD *)(v7 + 24) = v54[1].m128i_i64[1];
      *(_QWORD *)(v7 + 32) = v54[2].m128i_i64[0];
    }
    else
    {
      *(_QWORD *)(v7 + 16) = v55;
      v56 = *(_QWORD *)(v7 + 32);
      *(_QWORD *)(v7 + 24) = v54[1].m128i_i64[1];
      *(_QWORD *)(v7 + 32) = v54[2].m128i_i64[0];
      if ( v52 )
      {
        v54[1].m128i_i64[0] = (__int64)v52;
        v54[2].m128i_i64[0] = v56;
LABEL_83:
        v54[1].m128i_i64[1] = 0;
        v29 = v53 - 1;
        v52->m128i_i8[0] = 0;
        v7 = v4 + 48 * v29;
        v15 = (__m128i *)(v7 + 32);
        goto LABEL_29;
      }
    }
    v54[1].m128i_i64[0] = (__int64)v54[2].m128i_i64;
    v52 = (__m128i *)&v54[2];
    goto LABEL_83;
  }
LABEL_29:
  v30 = (__m128i *)a4[1].m128i_i64[0];
  s2 = src;
  v71 = _mm_loadu_si128(a4);
  if ( v30 == &a4[2] )
  {
    src[0] = _mm_loadu_si128(a4 + 2);
  }
  else
  {
    s2 = v30;
    *(_QWORD *)&src[0] = a4[2].m128i_i64[0];
  }
  a4[1].m128i_i64[0] = (__int64)a4[2].m128i_i64;
  v31 = a4[1].m128i_u64[1];
  a4[2].m128i_i8[0] = 0;
  a4[1].m128i_i64[1] = 0;
  v73 = v31;
  if ( v29 > v5 )
  {
    v32 = v29;
    v33 = (v29 - 1) / 2;
    v34 = v4;
    while ( 1 )
    {
      v40 = v34 + 48 * v33;
      v41 = *(_QWORD *)v40 < v71.m128i_i64[0];
      if ( *(_QWORD *)v40 == v71.m128i_i64[0]
        && (v41 = *(_QWORD *)(v40 + 8) < v71.m128i_i64[1], *(_QWORD *)(v40 + 8) == v71.m128i_i64[1]) )
      {
        v42 = *(_QWORD *)(v40 + 24);
        v31 = v73;
        v43 = v73;
        if ( v42 <= v73 )
          v43 = *(_QWORD *)(v40 + 24);
        if ( !v43 )
          goto LABEL_47;
        v62 = v5;
        v65 = v15;
        v69 = *(_QWORD *)(v40 + 24);
        v44 = memcmp(*(const void **)(v40 + 16), s2, v43);
        v42 = v69;
        v15 = v65;
        v5 = v62;
        v31 = v73;
        if ( v44 )
        {
          v35 = v44 >> 31;
        }
        else
        {
LABEL_47:
          v45 = v42 - v31;
          if ( v45 >= 0x80000000LL )
          {
            v7 = v34 + 48 * v32;
            break;
          }
          if ( v45 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v7 = v34 + 48 * v32;
            goto LABEL_35;
          }
          LOBYTE(v35) = (int)v45 < 0;
        }
      }
      else
      {
        LOBYTE(v35) = v41;
      }
      v7 = v34 + 48 * v32;
      if ( !(_BYTE)v35 )
      {
        v31 = v73;
        break;
      }
LABEL_35:
      v36 = (__m128i *)(v40 + 32);
      v37 = *(__m128i **)(v7 + 16);
      *(__m128i *)v7 = _mm_loadu_si128((const __m128i *)v40);
      v38 = *(_QWORD *)(v40 + 16);
      if ( v38 == v40 + 32 )
      {
        v50 = *(_QWORD *)(v40 + 24);
        if ( v50 )
        {
          if ( v50 == 1 )
          {
            v37->m128i_i8[0] = *(_BYTE *)(v40 + 32);
            v50 = *(_QWORD *)(v40 + 24);
            v37 = *(__m128i **)(v7 + 16);
          }
          else
          {
            v66 = v5;
            memcpy(v37, (const void *)(v40 + 32), v50);
            v50 = *(_QWORD *)(v40 + 24);
            v37 = *(__m128i **)(v7 + 16);
            v5 = v66;
            v36 = (__m128i *)(v40 + 32);
          }
        }
        *(_QWORD *)(v7 + 24) = v50;
        v37->m128i_i8[v50] = 0;
        v37 = *(__m128i **)(v40 + 16);
      }
      else
      {
        if ( v15 == v37 )
        {
          *(_QWORD *)(v7 + 16) = v38;
          *(_QWORD *)(v7 + 24) = *(_QWORD *)(v40 + 24);
          *(_QWORD *)(v7 + 32) = *(_QWORD *)(v40 + 32);
        }
        else
        {
          *(_QWORD *)(v7 + 16) = v38;
          v39 = *(_QWORD *)(v7 + 32);
          *(_QWORD *)(v7 + 24) = *(_QWORD *)(v40 + 24);
          *(_QWORD *)(v7 + 32) = *(_QWORD *)(v40 + 32);
          if ( v37 )
          {
            *(_QWORD *)(v40 + 16) = v37;
            *(_QWORD *)(v40 + 32) = v39;
            goto LABEL_39;
          }
        }
        *(_QWORD *)(v40 + 16) = v36;
        v37 = (__m128i *)(v40 + 32);
      }
LABEL_39:
      *(_QWORD *)(v40 + 24) = 0;
      v32 = v33;
      v37->m128i_i8[0] = 0;
      if ( v5 >= v33 )
      {
        v31 = v73;
        v15 = v36;
        v7 = v34 + 48 * v33;
        break;
      }
      v15 = v36;
      v33 = (v33 - 1) / 2;
    }
  }
  result = (__int64)s2;
  v47 = *(__m128i **)(v7 + 16);
  *(__m128i *)v7 = _mm_load_si128(&v71);
  if ( (_OWORD *)result == src )
  {
    if ( v31 )
    {
      if ( v31 == 1 )
      {
        result = LOBYTE(src[0]);
        v47->m128i_i8[0] = src[0];
      }
      else
      {
        result = (__int64)memcpy(v47, src, v31);
      }
      v31 = v73;
      v47 = *(__m128i **)(v7 + 16);
    }
    *(_QWORD *)(v7 + 24) = v31;
    v47->m128i_i8[v31] = 0;
    v47 = (__m128i *)s2;
  }
  else
  {
    v48 = *(_QWORD *)&src[0];
    if ( v15 == v47 )
    {
      *(_QWORD *)(v7 + 16) = result;
      *(_QWORD *)(v7 + 24) = v31;
      *(_QWORD *)(v7 + 32) = v48;
    }
    else
    {
      v49 = *(_QWORD *)(v7 + 32);
      *(_QWORD *)(v7 + 16) = result;
      *(_QWORD *)(v7 + 24) = v31;
      *(_QWORD *)(v7 + 32) = v48;
      if ( v47 )
      {
        s2 = v47;
        *(_QWORD *)&src[0] = v49;
        goto LABEL_57;
      }
    }
    s2 = src;
    v47 = (__m128i *)src;
  }
LABEL_57:
  v73 = 0;
  v47->m128i_i8[0] = 0;
  if ( s2 != src )
    return j_j___libc_free_0(s2, *(_QWORD *)&src[0] + 1LL);
  return result;
}
