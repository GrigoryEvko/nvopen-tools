// Function: sub_166DE00
// Address: 0x166de00
//
__int64 __fastcall sub_166DE00(__int64 a1, __int64 a2)
{
  const void *v4; // rsi
  _BYTE *v5; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rdi
  const void *v8; // rsi
  __int64 v9; // rdx
  const void *v10; // rsi
  _BYTE *v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r10
  __int64 result; // rax
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rcx
  const __m128i *v19; // r14
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // r13
  const __m128i *v24; // r8
  __m128i *v25; // rax
  const __m128i *v26; // r14
  const __m128i *v27; // rdx
  __m128i *v28; // rsi
  __m128i v29; // xmm0
  const __m128i *v30; // rcx
  size_t v31; // rdx
  size_t v32; // rdx
  size_t v33; // rdx
  unsigned __int64 v34; // r14
  __int64 v35; // rdi
  __int64 v36; // r13
  __int64 v37; // r12
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // rdi
  __int64 v42; // r13
  __int64 v43; // r12
  __int64 v44; // rdi
  __m128i *v45; // r14
  __m128i *v46; // rcx
  __int64 v47; // r13
  __m128i *v48; // r9
  __int64 v49; // rdx
  __m128i *v50; // rdi
  __m128i *v51; // rax
  size_t v52; // rdx
  __m128i *v53; // r14
  __m128i *v54; // r15
  __int64 v55; // r8
  __m128i *v56; // r13
  __int64 v57; // rdx
  __m128i *v58; // rdi
  __m128i *v59; // rax
  size_t v60; // rdx
  __m128i *v61; // [rsp+8h] [rbp-48h]
  __int64 v62; // [rsp+8h] [rbp-48h]
  unsigned int v63; // [rsp+14h] [rbp-3Ch]
  __int64 v64; // [rsp+18h] [rbp-38h]
  unsigned __int64 v65; // [rsp+18h] [rbp-38h]
  __m128i *v66; // [rsp+18h] [rbp-38h]
  unsigned __int64 v67; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  v4 = *(const void **)(a2 + 16);
  v5 = *(_BYTE **)(a1 + 16);
  if ( v4 == (const void *)(a2 + 32) )
  {
    v31 = *(_QWORD *)(a2 + 24);
    if ( v31 )
    {
      if ( v31 == 1 )
        *v5 = *(_BYTE *)(a2 + 32);
      else
        memcpy(v5, v4, v31);
      v31 = *(_QWORD *)(a2 + 24);
      v5 = *(_BYTE **)(a1 + 16);
    }
    *(_QWORD *)(a1 + 24) = v31;
    v5[v31] = 0;
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
    v32 = *(_QWORD *)(a2 + 72);
    if ( v32 )
    {
      if ( v32 == 1 )
        *v7 = *(_BYTE *)(a2 + 80);
      else
        memcpy(v7, v8, v32);
      v32 = *(_QWORD *)(a2 + 72);
      v7 = *(_BYTE **)(a1 + 64);
    }
    *(_QWORD *)(a1 + 72) = v32;
    v7[v32] = 0;
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
    v33 = *(_QWORD *)(a2 + 104);
    if ( v33 )
    {
      if ( v33 == 1 )
        *v11 = *(_BYTE *)(a2 + 112);
      else
        memcpy(v11, v10, v33);
      v33 = *(_QWORD *)(a2 + 104);
      v11 = *(_BYTE **)(a1 + 96);
    }
    *(_QWORD *)(a1 + 104) = v33;
    v11[v33] = 0;
    v11 = *(_BYTE **)(a2 + 96);
  }
  else
  {
    if ( v11 == (_BYTE *)(a1 + 112) )
    {
      *(_QWORD *)(a1 + 96) = v10;
      *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
      *(_QWORD *)(a1 + 112) = *(_QWORD *)(a2 + 112);
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
    j_j___libc_free_0(v13, v14 - v13);
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
    v21 = v17 + 48 * v18;
    if ( v21 != v17 )
    {
      do
      {
        v21 -= 48LL;
        v22 = *(_QWORD *)(v21 + 16);
        if ( v22 != v21 + 32 )
          j_j___libc_free_0(v22, *(_QWORD *)(v21 + 32) + 1LL);
      }
      while ( v21 != v17 );
      v20 = *(_QWORD *)(a1 + 152);
    }
    if ( v20 != a1 + 168 )
      _libc_free(v20);
    *(_QWORD *)(a1 + 152) = *(_QWORD *)(a2 + 152);
    *(_DWORD *)(a1 + 160) = *(_DWORD *)(a2 + 160);
    result = *(unsigned int *)(a2 + 164);
    *(_DWORD *)(a1 + 164) = result;
    *(_QWORD *)(a2 + 152) = v19;
    *(_QWORD *)(a2 + 160) = 0;
    return result;
  }
  v63 = *(_DWORD *)(a2 + 160);
  v23 = v63;
  if ( v63 > v18 )
  {
    if ( v63 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
    {
      v34 = v17 + 48 * v18;
      while ( v34 != v17 )
      {
        while ( 1 )
        {
          v34 -= 48LL;
          v35 = *(_QWORD *)(v34 + 16);
          if ( v35 == v34 + 32 )
            break;
          v64 = v15;
          j_j___libc_free_0(v35, *(_QWORD *)(v34 + 32) + 1LL);
          v15 = v64;
          if ( v34 == v17 )
            goto LABEL_62;
        }
      }
LABEL_62:
      *(_DWORD *)(a1 + 160) = 0;
      sub_166DC00(v15, v63);
      v19 = *(const __m128i **)(a2 + 152);
      v23 = *(unsigned int *)(a2 + 160);
      v18 = 0;
      v17 = *(_QWORD *)(a1 + 152);
      v24 = v19;
      goto LABEL_31;
    }
    v24 = (const __m128i *)(a2 + 168);
    if ( !*(_DWORD *)(a1 + 160) )
      goto LABEL_31;
    v53 = (__m128i *)(a2 + 200);
    v54 = (__m128i *)(v17 + 32);
    v55 = 48 * v18;
    v18 = v55;
    v56 = (__m128i *)(a2 + 200 + v55);
    while ( 1 )
    {
      v58 = (__m128i *)v54[-1].m128i_i64[0];
      v54[-2] = _mm_loadu_si128(v53 - 2);
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
            v58 = (__m128i *)v54[-1].m128i_i64[0];
          }
          else
          {
            v62 = v55;
            v67 = v18;
            memcpy(v58, v53, v60);
            v60 = v53[-1].m128i_u64[1];
            v58 = (__m128i *)v54[-1].m128i_i64[0];
            v55 = v62;
            v18 = v67;
          }
        }
        v54[-1].m128i_i64[1] = v60;
        v58->m128i_i8[v60] = 0;
        v58 = (__m128i *)v53[-1].m128i_i64[0];
        goto LABEL_98;
      }
      if ( v58 == v54 )
        break;
      v54[-1].m128i_i64[0] = (__int64)v59;
      v57 = v54->m128i_i64[0];
      v54[-1].m128i_i64[1] = v53[-1].m128i_i64[1];
      v54->m128i_i64[0] = v53->m128i_i64[0];
      if ( !v58 )
        goto LABEL_105;
      v53[-1].m128i_i64[0] = (__int64)v58;
      v53->m128i_i64[0] = v57;
LABEL_98:
      v53[-1].m128i_i64[1] = 0;
      v53 += 3;
      v54 += 3;
      v58->m128i_i8[0] = 0;
      if ( v53 == v56 )
      {
        v19 = *(const __m128i **)(a2 + 152);
        v23 = *(unsigned int *)(a2 + 160);
        v17 = *(_QWORD *)(a1 + 152);
        v24 = (const __m128i *)((char *)v19 + v55);
LABEL_31:
        v25 = (__m128i *)(v17 + v18);
        v26 = &v19[3 * v23];
        if ( v26 != v24 )
        {
          v27 = v24 + 2;
          v28 = &v25[3
                   * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v26 - (char *)v24 - 48) >> 4))
                    & 0xFFFFFFFFFFFFFFFLL)
                   + 3];
          do
          {
            if ( v25 )
            {
              v29 = _mm_loadu_si128(v27 - 2);
              v25[1].m128i_i64[0] = (__int64)v25[2].m128i_i64;
              *v25 = v29;
              v30 = (const __m128i *)v27[-1].m128i_i64[0];
              if ( v27 == v30 )
              {
                v25[2] = _mm_loadu_si128(v27);
              }
              else
              {
                v25[1].m128i_i64[0] = (__int64)v30;
                v25[2].m128i_i64[0] = v27->m128i_i64[0];
              }
              v25[1].m128i_i64[1] = v27[-1].m128i_i64[1];
              v27[-1].m128i_i64[0] = (__int64)v27;
              v27[-1].m128i_i64[1] = 0;
              v27->m128i_i8[0] = 0;
            }
            v25 += 3;
            v27 += 3;
          }
          while ( v25 != v28 );
        }
        *(_DWORD *)(a1 + 160) = v63;
        result = *(unsigned int *)(a2 + 160);
        v36 = *(_QWORD *)(a2 + 152);
        v37 = v36 + 48 * result;
        while ( v36 != v37 )
        {
          v37 -= 48;
          v38 = *(_QWORD *)(v37 + 16);
          result = v37 + 32;
          if ( v38 != v37 + 32 )
            result = j_j___libc_free_0(v38, *(_QWORD *)(v37 + 32) + 1LL);
        }
LABEL_67:
        *(_DWORD *)(a2 + 160) = 0;
        return result;
      }
    }
    v54[-1].m128i_i64[0] = (__int64)v59;
    v54[-1].m128i_i64[1] = v53[-1].m128i_i64[1];
    v54->m128i_i64[0] = v53->m128i_i64[0];
LABEL_105:
    v53[-1].m128i_i64[0] = (__int64)v53;
    v58 = v53;
    goto LABEL_98;
  }
  v39 = *(_QWORD *)(a1 + 152);
  if ( v63 )
  {
    v45 = (__m128i *)(a2 + 200);
    v46 = (__m128i *)(v17 + 32);
    v47 = 48LL * v63;
    v48 = (__m128i *)(a2 + 200 + v47);
    while ( 1 )
    {
      v50 = (__m128i *)v46[-1].m128i_i64[0];
      v46[-2] = _mm_loadu_si128(v45 - 2);
      v51 = (__m128i *)v45[-1].m128i_i64[0];
      if ( v51 == v45 )
      {
        v52 = v45[-1].m128i_u64[1];
        if ( v52 )
        {
          if ( v52 == 1 )
          {
            v50->m128i_i8[0] = v45->m128i_i8[0];
            v52 = v45[-1].m128i_u64[1];
            v50 = (__m128i *)v46[-1].m128i_i64[0];
          }
          else
          {
            v61 = v46;
            v66 = v48;
            memcpy(v50, v45, v52);
            v46 = v61;
            v52 = v45[-1].m128i_u64[1];
            v48 = v66;
            v50 = (__m128i *)v61[-1].m128i_i64[0];
          }
        }
        v46[-1].m128i_i64[1] = v52;
        v50->m128i_i8[v52] = 0;
        v50 = (__m128i *)v45[-1].m128i_i64[0];
        goto LABEL_85;
      }
      if ( v46 == v50 )
        break;
      v46[-1].m128i_i64[0] = (__int64)v51;
      v49 = v46->m128i_i64[0];
      v46[-1].m128i_i64[1] = v45[-1].m128i_i64[1];
      v46->m128i_i64[0] = v45->m128i_i64[0];
      if ( !v50 )
        goto LABEL_92;
      v45[-1].m128i_i64[0] = (__int64)v50;
      v45->m128i_i64[0] = v49;
LABEL_85:
      v45[-1].m128i_i64[1] = 0;
      v45 += 3;
      v46 += 3;
      v50->m128i_i8[0] = 0;
      if ( v48 == v45 )
      {
        v39 = *(_QWORD *)(a1 + 152);
        v18 = *(unsigned int *)(a1 + 160);
        v20 = v17 + v47;
        goto LABEL_69;
      }
    }
    v46[-1].m128i_i64[0] = (__int64)v51;
    v46[-1].m128i_i64[1] = v45[-1].m128i_i64[1];
    v46->m128i_i64[0] = v45->m128i_i64[0];
LABEL_92:
    v45[-1].m128i_i64[0] = (__int64)v45;
    v50 = v45;
    goto LABEL_85;
  }
LABEL_69:
  v40 = v39 + 48 * v18;
  while ( v20 != v40 )
  {
    v40 -= 48;
    v41 = *(_QWORD *)(v40 + 16);
    if ( v41 != v40 + 32 )
    {
      v65 = v20;
      j_j___libc_free_0(v41, *(_QWORD *)(v40 + 32) + 1LL);
      v20 = v65;
    }
  }
  *(_DWORD *)(a1 + 160) = v63;
  result = *(unsigned int *)(a2 + 160);
  v42 = *(_QWORD *)(a2 + 152);
  v43 = v42 + 48 * result;
  if ( v42 == v43 )
    goto LABEL_67;
  do
  {
    v43 -= 48;
    v44 = *(_QWORD *)(v43 + 16);
    result = v43 + 32;
    if ( v44 != v43 + 32 )
      result = j_j___libc_free_0(v44, *(_QWORD *)(v43 + 32) + 1LL);
  }
  while ( v42 != v43 );
  *(_DWORD *)(a2 + 160) = 0;
  return result;
}
