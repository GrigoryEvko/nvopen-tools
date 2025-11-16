// Function: sub_E45B90
// Address: 0xe45b90
//
__int64 __fastcall sub_E45B90(__int64 a1, __int64 a2)
{
  const void *v4; // rsi
  _BYTE *v5; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rdi
  const void *v8; // rsi
  __int64 v9; // rdx
  const void *v10; // rsi
  _BYTE *v11; // rdi
  size_t v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r10
  __int64 result; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // r15
  const __m128i *v19; // r14
  __int64 v20; // r8
  unsigned __int64 v21; // r9
  int v22; // r13d
  const __m128i *v23; // r8
  __m128i *v24; // rdx
  const __m128i *v25; // r14
  const __m128i *v26; // rax
  __m128i *v27; // rsi
  __m128i v28; // xmm0
  const __m128i *v29; // rcx
  __int64 v30; // r13
  __int64 v31; // r12
  __int64 v32; // rdi
  __int64 v33; // r13
  __int64 v34; // rdi
  size_t v35; // rdx
  size_t v36; // rdx
  __m128i *v37; // r14
  __m128i *v38; // rcx
  __int64 v39; // r8
  __m128i *v40; // r9
  __int64 v41; // rdx
  __m128i *v42; // rdi
  __m128i *v43; // rax
  size_t v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // rdi
  __int64 v48; // r13
  __int64 v49; // r12
  __int64 v50; // rdi
  __int64 v51; // r14
  __int64 v52; // rdi
  __m128i *v53; // r14
  __int64 v54; // r8
  __m128i *v55; // r15
  __int64 v56; // r9
  __int64 v57; // rdx
  __m128i *v58; // rdi
  __m128i *v59; // rax
  size_t v60; // rdx
  __int64 v61; // [rsp+8h] [rbp-48h]
  __int64 v62; // [rsp+8h] [rbp-48h]
  __int64 v63; // [rsp+8h] [rbp-48h]
  __m128i *v64; // [rsp+10h] [rbp-40h]
  unsigned __int64 v65; // [rsp+10h] [rbp-40h]
  __int64 v66; // [rsp+10h] [rbp-40h]
  __int64 v67; // [rsp+18h] [rbp-38h]
  __m128i *v68; // [rsp+18h] [rbp-38h]
  __int64 v69; // [rsp+18h] [rbp-38h]
  __int64 v70; // [rsp+18h] [rbp-38h]
  __int64 v71; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  v4 = *(const void **)(a2 + 16);
  v5 = *(_BYTE **)(a1 + 16);
  if ( v4 == (const void *)(a2 + 32) )
  {
    v35 = *(_QWORD *)(a2 + 24);
    if ( v35 )
    {
      if ( v35 == 1 )
        *v5 = *(_BYTE *)(a2 + 32);
      else
        memcpy(v5, v4, v35);
      v35 = *(_QWORD *)(a2 + 24);
      v5 = *(_BYTE **)(a1 + 16);
    }
    *(_QWORD *)(a1 + 24) = v35;
    v5[v35] = 0;
    v5 = *(_BYTE **)(a2 + 16);
  }
  else
  {
    if ( v5 == (_BYTE *)(a1 + 32) )
    {
      *(_QWORD *)(a1 + 16) = v4;
      *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_QWORD *)(a1 + 16) = v4;
      v6 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
      if ( v5 )
      {
        *(_QWORD *)(a2 + 16) = v5;
        *(_QWORD *)(a2 + 32) = v6;
        goto LABEL_5;
      }
    }
    *(_QWORD *)(a2 + 16) = a2 + 32;
    v5 = (_BYTE *)(a2 + 32);
  }
LABEL_5:
  *(_QWORD *)(a2 + 24) = 0;
  *v5 = 0;
  v7 = *(_BYTE **)(a1 + 64);
  *(_DWORD *)(a1 + 48) = *(_DWORD *)(a2 + 48);
  *(_DWORD *)(a1 + 52) = *(_DWORD *)(a2 + 52);
  *(_DWORD *)(a1 + 56) = *(_DWORD *)(a2 + 56);
  v8 = *(const void **)(a2 + 64);
  if ( v8 == (const void *)(a2 + 80) )
  {
    v36 = *(_QWORD *)(a2 + 72);
    if ( v36 )
    {
      if ( v36 == 1 )
        *v7 = *(_BYTE *)(a2 + 80);
      else
        memcpy(v7, v8, v36);
      v36 = *(_QWORD *)(a2 + 72);
      v7 = *(_BYTE **)(a1 + 64);
    }
    *(_QWORD *)(a1 + 72) = v36;
    v7[v36] = 0;
    v7 = *(_BYTE **)(a2 + 64);
  }
  else
  {
    if ( v7 == (_BYTE *)(a1 + 80) )
    {
      *(_QWORD *)(a1 + 64) = v8;
      *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
      *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
    }
    else
    {
      *(_QWORD *)(a1 + 64) = v8;
      v9 = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
      *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
      if ( v7 )
      {
        *(_QWORD *)(a2 + 64) = v7;
        *(_QWORD *)(a2 + 80) = v9;
        goto LABEL_9;
      }
    }
    *(_QWORD *)(a2 + 64) = a2 + 80;
    v7 = (_BYTE *)(a2 + 80);
  }
LABEL_9:
  *(_QWORD *)(a2 + 72) = 0;
  *v7 = 0;
  v10 = *(const void **)(a2 + 96);
  v11 = *(_BYTE **)(a1 + 96);
  if ( v10 == (const void *)(a2 + 112) )
  {
    v12 = *(_QWORD *)(a2 + 104);
    if ( v12 )
    {
      if ( v12 == 1 )
        *v11 = *(_BYTE *)(a2 + 112);
      else
        memcpy(v11, v10, v12);
      v12 = *(_QWORD *)(a2 + 104);
      v11 = *(_BYTE **)(a1 + 96);
    }
    *(_QWORD *)(a1 + 104) = v12;
    v11[v12] = 0;
    v11 = *(_BYTE **)(a2 + 96);
  }
  else
  {
    if ( v11 == (_BYTE *)(a1 + 112) )
    {
      *(_QWORD *)(a1 + 96) = v10;
      *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
      v12 = *(_QWORD *)(a2 + 112);
      *(_QWORD *)(a1 + 112) = v12;
    }
    else
    {
      *(_QWORD *)(a1 + 96) = v10;
      v12 = *(_QWORD *)(a1 + 112);
      *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
      *(_QWORD *)(a1 + 112) = *(_QWORD *)(a2 + 112);
      if ( v11 )
      {
        *(_QWORD *)(a2 + 96) = v11;
        *(_QWORD *)(a2 + 112) = v12;
        goto LABEL_13;
      }
    }
    *(_QWORD *)(a2 + 96) = a2 + 112;
    v11 = (_BYTE *)(a2 + 112);
  }
LABEL_13:
  *(_QWORD *)(a2 + 104) = 0;
  *v11 = 0;
  v13 = *(_QWORD *)(a1 + 128);
  v14 = *(_QWORD *)(a1 + 144);
  *(_QWORD *)(a1 + 128) = *(_QWORD *)(a2 + 128);
  *(_QWORD *)(a1 + 136) = *(_QWORD *)(a2 + 136);
  *(_QWORD *)(a1 + 144) = *(_QWORD *)(a2 + 144);
  *(_QWORD *)(a2 + 128) = 0;
  *(_QWORD *)(a2 + 136) = 0;
  *(_QWORD *)(a2 + 144) = 0;
  if ( v13 )
  {
    v14 -= v13;
    j_j___libc_free_0(v13, v14);
  }
  v15 = a1 + 152;
  result = a2 + 152;
  if ( a1 + 152 == a2 + 152 )
    return result;
  v17 = *(_QWORD *)(a1 + 152);
  v18 = *(unsigned int *)(a1 + 160);
  v19 = (const __m128i *)(a2 + 168);
  v20 = v17;
  if ( *(_QWORD *)(a2 + 152) != a2 + 168 )
  {
    v33 = v17 + 48 * v18;
    if ( v33 != v17 )
    {
      do
      {
        v33 -= 48;
        v34 = *(_QWORD *)(v33 + 16);
        if ( v34 != v33 + 32 )
        {
          v67 = v17;
          v14 = *(_QWORD *)(v33 + 32) + 1LL;
          j_j___libc_free_0(v34, v14);
          v17 = v67;
        }
      }
      while ( v33 != v17 );
      v20 = *(_QWORD *)(a1 + 152);
    }
    if ( v20 != a1 + 168 )
      _libc_free(v20, v14);
    *(_QWORD *)(a1 + 152) = *(_QWORD *)(a2 + 152);
    *(_DWORD *)(a1 + 160) = *(_DWORD *)(a2 + 160);
    result = *(unsigned int *)(a2 + 164);
    *(_DWORD *)(a1 + 164) = result;
    *(_QWORD *)(a2 + 152) = v19;
    *(_QWORD *)(a2 + 160) = 0;
    return result;
  }
  v21 = *(unsigned int *)(a2 + 160);
  v22 = *(_DWORD *)(a2 + 160);
  if ( v21 > v18 )
  {
    if ( v21 > *(unsigned int *)(a1 + 164) )
    {
      v51 = v17 + 48 * v18;
      while ( v51 != v17 )
      {
        while ( 1 )
        {
          v51 -= 48;
          v52 = *(_QWORD *)(v51 + 16);
          if ( v52 == v51 + 32 )
            break;
          v62 = v17;
          v65 = v21;
          v70 = v15;
          j_j___libc_free_0(v52, *(_QWORD *)(v51 + 32) + 1LL);
          v17 = v62;
          v21 = v65;
          v15 = v70;
          if ( v51 == v62 )
            goto LABEL_89;
        }
      }
LABEL_89:
      v18 = 0;
      *(_DWORD *)(a1 + 160) = 0;
      sub_C8F9C0(v15, v21, v12, v17, v20, v21);
      v19 = *(const __m128i **)(a2 + 152);
      v21 = *(unsigned int *)(a2 + 160);
      v17 = *(_QWORD *)(a1 + 152);
      v23 = v19;
      goto LABEL_20;
    }
    v23 = (const __m128i *)(a2 + 168);
    if ( !*(_DWORD *)(a1 + 160) )
      goto LABEL_20;
    v37 = (__m128i *)(a2 + 200);
    v38 = (__m128i *)(v17 + 32);
    v39 = 48 * v18;
    v18 = v39;
    v40 = (__m128i *)(a2 + 200 + v39);
    while ( 1 )
    {
      v42 = (__m128i *)v38[-1].m128i_i64[0];
      v38[-2] = _mm_loadu_si128(v37 - 2);
      v43 = (__m128i *)v37[-1].m128i_i64[0];
      if ( v43 == v37 )
      {
        v44 = v37[-1].m128i_u64[1];
        if ( v44 )
        {
          if ( v44 == 1 )
          {
            v42->m128i_i8[0] = v37->m128i_i8[0];
            v44 = v37[-1].m128i_u64[1];
            v42 = (__m128i *)v38[-1].m128i_i64[0];
          }
          else
          {
            v61 = v39;
            v64 = v40;
            v68 = v38;
            memcpy(v42, v37, v44);
            v38 = v68;
            v44 = v37[-1].m128i_u64[1];
            v39 = v61;
            v40 = v64;
            v42 = (__m128i *)v68[-1].m128i_i64[0];
          }
        }
        v38[-1].m128i_i64[1] = v44;
        v42->m128i_i8[v44] = 0;
        v42 = (__m128i *)v37[-1].m128i_i64[0];
        goto LABEL_67;
      }
      if ( v42 == v38 )
        break;
      v38[-1].m128i_i64[0] = (__int64)v43;
      v41 = v38->m128i_i64[0];
      v38[-1].m128i_i64[1] = v37[-1].m128i_i64[1];
      v38->m128i_i64[0] = v37->m128i_i64[0];
      if ( !v42 )
        goto LABEL_84;
      v37[-1].m128i_i64[0] = (__int64)v42;
      v37->m128i_i64[0] = v41;
LABEL_67:
      v37[-1].m128i_i64[1] = 0;
      v37 += 3;
      v38 += 3;
      v42->m128i_i8[0] = 0;
      if ( v37 == v40 )
      {
        v19 = *(const __m128i **)(a2 + 152);
        v21 = *(unsigned int *)(a2 + 160);
        v17 = *(_QWORD *)(a1 + 152);
        v23 = (const __m128i *)((char *)v19 + v39);
LABEL_20:
        v24 = (__m128i *)(v17 + v18);
        v25 = &v19[3 * v21];
        if ( v25 != v23 )
        {
          v26 = v23 + 2;
          v27 = &v24[3
                   * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v25 - (char *)v23 - 48) >> 4))
                    & 0xFFFFFFFFFFFFFFFLL)
                   + 3];
          do
          {
            if ( v24 )
            {
              v28 = _mm_loadu_si128(v26 - 2);
              v24[1].m128i_i64[0] = (__int64)v24[2].m128i_i64;
              *v24 = v28;
              v29 = (const __m128i *)v26[-1].m128i_i64[0];
              if ( v26 == v29 )
              {
                v24[2] = _mm_loadu_si128(v26);
              }
              else
              {
                v24[1].m128i_i64[0] = (__int64)v29;
                v24[2].m128i_i64[0] = v26->m128i_i64[0];
              }
              v24[1].m128i_i64[1] = v26[-1].m128i_i64[1];
              v26[-1].m128i_i64[0] = (__int64)v26;
              v26[-1].m128i_i64[1] = 0;
              v26->m128i_i8[0] = 0;
            }
            v24 += 3;
            v26 += 3;
          }
          while ( v24 != v27 );
        }
        *(_DWORD *)(a1 + 160) = v22;
        result = *(unsigned int *)(a2 + 160);
        v30 = *(_QWORD *)(a2 + 152);
        v31 = v30 + 48 * result;
        while ( v30 != v31 )
        {
          v31 -= 48;
          v32 = *(_QWORD *)(v31 + 16);
          result = v31 + 32;
          if ( v32 != v31 + 32 )
            result = j_j___libc_free_0(v32, *(_QWORD *)(v31 + 32) + 1LL);
        }
LABEL_34:
        *(_DWORD *)(a2 + 160) = 0;
        return result;
      }
    }
    v38[-1].m128i_i64[0] = (__int64)v43;
    v38[-1].m128i_i64[1] = v37[-1].m128i_i64[1];
    v38->m128i_i64[0] = v37->m128i_i64[0];
LABEL_84:
    v37[-1].m128i_i64[0] = (__int64)v37;
    v42 = v37;
    goto LABEL_67;
  }
  v45 = *(_QWORD *)(a1 + 152);
  if ( *(_DWORD *)(a2 + 160) )
  {
    v53 = (__m128i *)(a2 + 200);
    v54 = 48 * v21;
    v55 = (__m128i *)(v17 + 32);
    v56 = a2 + 200 + 48 * v21;
    while ( 1 )
    {
      v58 = (__m128i *)v55[-1].m128i_i64[0];
      v55[-2] = _mm_loadu_si128(v53 - 2);
      v59 = (__m128i *)v53[-1].m128i_i64[0];
      if ( v59 == v53 )
      {
        v60 = v53[-1].m128i_u64[1];
        if ( v60 )
        {
          if ( v60 == 1 )
          {
            v58->m128i_i8[0] = v53->m128i_i8[0];
            v60 = v53[-1].m128i_u64[1];
            v58 = (__m128i *)v55[-1].m128i_i64[0];
          }
          else
          {
            v63 = v17;
            v66 = v54;
            v71 = v56;
            memcpy(v58, v53, v60);
            v60 = v53[-1].m128i_u64[1];
            v58 = (__m128i *)v55[-1].m128i_i64[0];
            v17 = v63;
            v54 = v66;
            v56 = v71;
          }
        }
        v55[-1].m128i_i64[1] = v60;
        v58->m128i_i8[v60] = 0;
        v58 = (__m128i *)v53[-1].m128i_i64[0];
        goto LABEL_95;
      }
      if ( v55 == v58 )
        break;
      v55[-1].m128i_i64[0] = (__int64)v59;
      v57 = v55->m128i_i64[0];
      v55[-1].m128i_i64[1] = v53[-1].m128i_i64[1];
      v55->m128i_i64[0] = v53->m128i_i64[0];
      if ( !v58 )
        goto LABEL_102;
      v53[-1].m128i_i64[0] = (__int64)v58;
      v53->m128i_i64[0] = v57;
LABEL_95:
      v53[-1].m128i_i64[1] = 0;
      v53 += 3;
      v55 += 3;
      v58->m128i_i8[0] = 0;
      if ( (__m128i *)v56 == v53 )
      {
        v45 = *(_QWORD *)(a1 + 152);
        v18 = *(unsigned int *)(a1 + 160);
        v20 = v17 + v54;
        goto LABEL_74;
      }
    }
    v55[-1].m128i_i64[0] = (__int64)v59;
    v55[-1].m128i_i64[1] = v53[-1].m128i_i64[1];
    v55->m128i_i64[0] = v53->m128i_i64[0];
LABEL_102:
    v53[-1].m128i_i64[0] = (__int64)v53;
    v58 = v53;
    goto LABEL_95;
  }
LABEL_74:
  v46 = v45 + 48 * v18;
  while ( v20 != v46 )
  {
    v46 -= 48;
    v47 = *(_QWORD *)(v46 + 16);
    if ( v47 != v46 + 32 )
    {
      v69 = v20;
      j_j___libc_free_0(v47, *(_QWORD *)(v46 + 32) + 1LL);
      v20 = v69;
    }
  }
  *(_DWORD *)(a1 + 160) = v22;
  result = *(unsigned int *)(a2 + 160);
  v48 = *(_QWORD *)(a2 + 152);
  v49 = v48 + 48 * result;
  if ( v48 == v49 )
    goto LABEL_34;
  do
  {
    v49 -= 48;
    v50 = *(_QWORD *)(v49 + 16);
    result = v49 + 32;
    if ( v50 != v49 + 32 )
      result = j_j___libc_free_0(v50, *(_QWORD *)(v49 + 32) + 1LL);
  }
  while ( v48 != v49 );
  *(_DWORD *)(a2 + 160) = 0;
  return result;
}
