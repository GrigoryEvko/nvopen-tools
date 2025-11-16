// Function: sub_C96F60
// Address: 0xc96f60
//
__int64 __fastcall sub_C96F60(
        __int64 a1,
        __m128i *a2,
        void (__fastcall *a3)(__m128i **, __int64),
        __int64 a4,
        __int32 a5)
{
  __int64 v7; // rax
  __m128i *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __m128i *v11; // rax
  size_t v12; // rax
  __m128i *v13; // rax
  __int64 v14; // r9
  __m128i *v15; // r14
  __m128i *v16; // rax
  __int64 v17; // rax
  __m128i *v18; // rdi
  __m128i *v19; // rcx
  size_t v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  __m128i **v24; // rdx
  __m128i *v26; // rax
  _QWORD *v27; // r13
  _QWORD *v28; // r12
  _QWORD *v29; // rdi
  _QWORD *v30; // rdi
  _QWORD *v31; // rdi
  __m128i *v32; // rdi
  __m128i *v33; // rdi
  __m128i *v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 *v37; // rdx
  _QWORD *v38; // rax
  _QWORD *v39; // rcx
  _QWORD *v40; // r12
  _QWORD *v41; // r13
  _QWORD *v42; // r15
  _QWORD *v43; // rdi
  _QWORD *v44; // rdi
  _QWORD *v45; // rdi
  _QWORD *v46; // rdi
  _QWORD *v47; // rdi
  _QWORD *v48; // rdi
  int v49; // r12d
  __int64 v50; // [rsp+10h] [rbp-B0h]
  __int64 *v51; // [rsp+18h] [rbp-A8h]
  __int64 v52; // [rsp+28h] [rbp-98h]
  __int64 *v53; // [rsp+28h] [rbp-98h]
  __m128i *v54; // [rsp+30h] [rbp-90h] BYREF
  size_t v55; // [rsp+38h] [rbp-88h]
  __m128i v56; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v57; // [rsp+50h] [rbp-70h]
  __int64 v58; // [rsp+58h] [rbp-68h]
  __m128i v59; // [rsp+60h] [rbp-60h] BYREF
  __m128i *v60; // [rsp+70h] [rbp-50h] BYREF
  size_t n; // [rsp+78h] [rbp-48h]
  _OWORD src[4]; // [rsp+80h] [rbp-40h] BYREF

  a3(&v54, a4);
  v7 = sub_220F880();
  v8 = (__m128i *)a2->m128i_i64[0];
  v57 = &v59;
  v9 = v7;
  if ( v8 == &a2[1] )
  {
    v59 = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v57 = v8;
    v59.m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v10 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v11 = v54;
  a2[1].m128i_i8[0] = 0;
  v58 = v10;
  a2->m128i_i64[1] = 0;
  v60 = (__m128i *)src;
  if ( v11 == &v56 )
  {
    src[0] = _mm_load_si128(&v56);
  }
  else
  {
    v60 = v11;
    *(_QWORD *)&src[0] = v56.m128i_i64[0];
  }
  v12 = v55;
  v52 = v9;
  v55 = 0;
  n = v12;
  v56.m128i_i8[0] = 0;
  v54 = &v56;
  v13 = (__m128i *)sub_22077B0(152);
  v15 = v13;
  if ( v13 )
  {
    v13->m128i_i64[0] = v52;
    v16 = v13 + 2;
    v16[-2].m128i_i64[1] = 0;
    v15[1].m128i_i64[0] = (__int64)v16;
    if ( v57 == &v59 )
    {
      v15[2] = _mm_load_si128(&v59);
    }
    else
    {
      v15[1].m128i_i64[0] = (__int64)v57;
      v15[2].m128i_i64[0] = v59.m128i_i64[0];
    }
    v17 = v58;
    v18 = v60;
    v19 = v15 + 4;
    v15[7].m128i_i32[2] = a5;
    v20 = n;
    v15[1].m128i_i64[1] = v17;
    v57 = &v59;
    v58 = 0;
    v59.m128i_i8[0] = 0;
    v15[7].m128i_i64[0] = 0;
    v15[3].m128i_i64[0] = (__int64)v15[4].m128i_i64;
    v15[3].m128i_i64[1] = 0;
    v15[5].m128i_i64[0] = (__int64)v15[6].m128i_i64;
    v15[5].m128i_i64[1] = 0;
    v15[4] = 0;
    v15[6] = 0;
    if ( v18 == (__m128i *)src )
    {
      if ( v20 )
      {
        if ( v20 == 1 )
        {
          v15[4].m128i_i8[0] = src[0];
        }
        else
        {
          v26 = (__m128i *)memcpy(&v15[4], src, v20);
          v18 = v60;
          v19 = v26;
        }
      }
      v15[3].m128i_i64[1] = v20;
      v19->m128i_i8[v20] = 0;
    }
    else
    {
      v21 = *(_QWORD *)&src[0];
      v15[3].m128i_i64[1] = v20;
      v60 = (__m128i *)src;
      v15[4].m128i_i64[0] = v21;
      v15[3].m128i_i64[0] = (__int64)v18;
      v18 = (__m128i *)src;
    }
    n = 0;
    LOBYTE(src[0]) = 0;
    v15[8].m128i_i64[0] = 0;
    v15[8].m128i_i64[1] = 0;
    v15[9].m128i_i64[0] = 0;
  }
  else
  {
    v18 = v60;
  }
  if ( v18 != (__m128i *)src )
    j_j___libc_free_0(v18, *(_QWORD *)&src[0] + 1LL);
  if ( v57 != &v59 )
    j_j___libc_free_0(v57, v59.m128i_i64[0] + 1);
  v22 = *(unsigned int *)(a1 + 8);
  v23 = v22;
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v22 )
  {
    v35 = a1 + 16;
    v50 = sub_C8D7D0(a1, a1 + 16, 0, 8u, (unsigned __int64 *)&v60, v14);
    v36 = 8LL * *(unsigned int *)(a1 + 8);
    if ( v36 + v50 )
    {
      *(_QWORD *)(v36 + v50) = v15;
      v15 = 0;
      v36 = 8LL * *(unsigned int *)(a1 + 8);
    }
    v37 = *(__int64 **)a1;
    v53 = (__int64 *)(*(_QWORD *)a1 + v36);
    if ( *(__int64 **)a1 != v53 )
    {
      v38 = (_QWORD *)v50;
      v39 = (_QWORD *)(v50 + v36);
      do
      {
        if ( v38 )
        {
          v35 = *v37;
          *v38 = *v37;
          *v37 = 0;
        }
        ++v38;
        ++v37;
      }
      while ( v38 != v39 );
      v51 = *(__int64 **)a1;
      v53 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      if ( *(__int64 **)a1 != v53 )
      {
        do
        {
          v40 = (_QWORD *)*--v53;
          if ( *v53 )
          {
            v41 = (_QWORD *)v40[17];
            v42 = (_QWORD *)v40[16];
            if ( v41 != v42 )
            {
              do
              {
                v43 = (_QWORD *)v42[10];
                if ( v43 != v42 + 12 )
                  j_j___libc_free_0(v43, v42[12] + 1LL);
                v44 = (_QWORD *)v42[6];
                if ( v44 != v42 + 8 )
                  j_j___libc_free_0(v44, v42[8] + 1LL);
                v45 = (_QWORD *)v42[2];
                if ( v45 != v42 + 4 )
                  j_j___libc_free_0(v45, v42[4] + 1LL);
                v42 += 16;
              }
              while ( v41 != v42 );
              v42 = (_QWORD *)v40[16];
            }
            if ( v42 )
              j_j___libc_free_0(v42, v40[18] - (_QWORD)v42);
            v46 = (_QWORD *)v40[10];
            if ( v46 != v40 + 12 )
              j_j___libc_free_0(v46, v40[12] + 1LL);
            v47 = (_QWORD *)v40[6];
            if ( v47 != v40 + 8 )
              j_j___libc_free_0(v47, v40[8] + 1LL);
            v48 = (_QWORD *)v40[2];
            if ( v48 != v40 + 4 )
              j_j___libc_free_0(v48, v40[4] + 1LL);
            v35 = 152;
            j_j___libc_free_0(v40, 152);
          }
        }
        while ( v51 != v53 );
        v53 = *(__int64 **)a1;
      }
    }
    v49 = (int)v60;
    if ( (__int64 *)(a1 + 16) != v53 )
      _libc_free(v53, v35);
    ++*(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 12) = v49;
    *(_QWORD *)a1 = v50;
  }
  else
  {
    v24 = (__m128i **)(*(_QWORD *)a1 + 8 * v22);
    if ( v24 )
    {
      *v24 = v15;
      ++*(_DWORD *)(a1 + 8);
      goto LABEL_18;
    }
    *(_DWORD *)(a1 + 8) = v23 + 1;
  }
  if ( v15 )
  {
    v27 = (_QWORD *)v15[8].m128i_i64[1];
    v28 = (_QWORD *)v15[8].m128i_i64[0];
    if ( v27 != v28 )
    {
      do
      {
        v29 = (_QWORD *)v28[10];
        if ( v29 != v28 + 12 )
          j_j___libc_free_0(v29, v28[12] + 1LL);
        v30 = (_QWORD *)v28[6];
        if ( v30 != v28 + 8 )
          j_j___libc_free_0(v30, v28[8] + 1LL);
        v31 = (_QWORD *)v28[2];
        if ( v31 != v28 + 4 )
          j_j___libc_free_0(v31, v28[4] + 1LL);
        v28 += 16;
      }
      while ( v27 != v28 );
      v28 = (_QWORD *)v15[8].m128i_i64[0];
    }
    if ( v28 )
      j_j___libc_free_0(v28, v15[9].m128i_i64[0] - (_QWORD)v28);
    v32 = (__m128i *)v15[5].m128i_i64[0];
    if ( v32 != &v15[6] )
      j_j___libc_free_0(v32, v15[6].m128i_i64[0] + 1);
    v33 = (__m128i *)v15[3].m128i_i64[0];
    if ( v33 != &v15[4] )
      j_j___libc_free_0(v33, v15[4].m128i_i64[0] + 1);
    v34 = (__m128i *)v15[1].m128i_i64[0];
    if ( v34 != &v15[2] )
      j_j___libc_free_0(v34, v15[2].m128i_i64[0] + 1);
    j_j___libc_free_0(v15, 152);
  }
LABEL_18:
  if ( v54 != &v56 )
    j_j___libc_free_0(v54, v56.m128i_i64[0] + 1);
  return *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8);
}
