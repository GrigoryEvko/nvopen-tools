// Function: sub_1FF20F0
// Address: 0x1ff20f0
//
void __fastcall sub_1FF20F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  char **v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r13
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // r9
  int v28; // edi
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // rdx
  const __m128i *v32; // r14
  __m128i v33; // xmm1
  __int64 m128i_i64; // rsi
  __int64 v35; // rdi
  __int64 v36; // r13
  __int64 v37; // r12
  unsigned __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // rax
  const __m128i *v42; // r14
  __m128i v43; // xmm2
  __int64 v44; // rcx
  __int64 v45; // r13
  __int64 v46; // rbx
  unsigned __int64 v47; // rdi
  int v48; // edx
  unsigned __int64 v49; // [rsp-B8h] [rbp-B8h]
  __int64 v50; // [rsp-B0h] [rbp-B0h]
  __int64 v51; // [rsp-B0h] [rbp-B0h]
  __int64 v52; // [rsp-B0h] [rbp-B0h]
  __int64 v53; // [rsp-A8h] [rbp-A8h]
  unsigned __int64 v54; // [rsp-A8h] [rbp-A8h]
  unsigned __int64 v55; // [rsp-A8h] [rbp-A8h]
  const __m128i *v56; // [rsp-A0h] [rbp-A0h]
  __int64 v57; // [rsp-A0h] [rbp-A0h]
  char *v58[2]; // [rsp-88h] [rbp-88h] BYREF
  _BYTE v59[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a1 == a2 )
    return;
  v8 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 != a1 + 16 )
  {
    a4 = a2 + 16;
    if ( *(_QWORD *)a2 != a2 + 16 )
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
      v9 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8;
      v10 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(a1 + 8) = v9;
      v11 = *(_DWORD *)(a2 + 12);
      *(_DWORD *)(a2 + 8) = v10;
      v12 = *(_DWORD *)(a1 + 12);
      *(_DWORD *)(a1 + 12) = v11;
      *(_DWORD *)(a2 + 12) = v12;
      return;
    }
  }
  v13 = *(unsigned int *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 12) >= (unsigned int)v13 )
  {
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v14 )
      goto LABEL_7;
    goto LABEL_20;
  }
  sub_1FF1F30(a1, v13);
  v14 = *(unsigned int *)(a1 + 8);
  v15 = v14;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v14 )
  {
LABEL_20:
    sub_1FF1F30(a2, v15);
    v14 = *(unsigned int *)(a1 + 8);
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
LABEL_7:
  v16 = *(unsigned int *)(a2 + 8);
  v17 = v14;
  if ( v16 <= v14 )
    v17 = *(unsigned int *)(a2 + 8);
  v49 = v17;
  if ( v17 )
  {
    v18 = 0;
    v53 = 96 * v17;
    do
    {
      v24 = v18 + *(_QWORD *)a2;
      v25 = v18 + *(_QWORD *)a1;
      v26 = _mm_loadu_si128((const __m128i *)v25);
      v27 = v25 + 16;
      *(_QWORD *)v25 = *(_QWORD *)v24;
      *(_DWORD *)(v25 + 8) = *(_DWORD *)(v24 + 8);
      *(_QWORD *)v24 = v26.m128i_i64[0];
      v58[0] = v59;
      *(_DWORD *)(v24 + 8) = v26.m128i_i32[2];
      v28 = *(_DWORD *)(v25 + 24);
      v58[1] = (char *)0x1000000000LL;
      if ( v28 )
      {
        v50 = v25 + 16;
        sub_1FEB760((__int64)v58, (char **)(v25 + 16), v25, a4, v14, v27);
        v27 = v50;
      }
      v19 = (char **)(v24 + 16);
      sub_1FEB760(v27, v19, v25, a4, v14, v27);
      sub_1FEB760((__int64)v19, v58, v20, v21, v22, v23);
      if ( v58[0] != v59 )
        _libc_free((unsigned __int64)v58[0]);
      v18 += 96;
    }
    while ( v53 != v18 );
    v14 = *(unsigned int *)(a1 + 8);
    v16 = *(unsigned int *)(a2 + 8);
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
  if ( v16 < v14 )
  {
    v29 = *(_QWORD *)a2 + 96 * v16;
    v30 = *(_QWORD *)a1 + 96 * v14;
    v31 = 96 * v49;
    v32 = (const __m128i *)(96 * v49 + *(_QWORD *)a1);
    if ( v32 != (const __m128i *)v30 )
    {
      do
      {
        while ( 1 )
        {
          if ( v29 )
          {
            v33 = _mm_loadu_si128(v32);
            *(_DWORD *)(v29 + 24) = 0;
            *(_QWORD *)(v29 + 16) = v29 + 32;
            *(_DWORD *)(v29 + 28) = 16;
            *(__m128i *)v29 = v33;
            if ( v32[1].m128i_i32[2] )
              break;
          }
          v32 += 6;
          v29 += 96;
          if ( (const __m128i *)v30 == v32 )
            goto LABEL_27;
        }
        m128i_i64 = (__int64)v32[1].m128i_i64;
        v35 = v29 + 16;
        v51 = v31;
        v32 += 6;
        v54 = v14;
        v29 += 96;
        v56 = (const __m128i *)v30;
        sub_1FEB680(v35, m128i_i64, v31, v30, v14, a6);
        v30 = (__int64)v56;
        v31 = v51;
        v14 = v54;
      }
      while ( v56 != v32 );
LABEL_27:
      LODWORD(v15) = v14 + *(_DWORD *)(a2 + 8) - v16;
    }
    *(_DWORD *)(a2 + 8) = v15;
    v36 = *(_QWORD *)a1 + v31;
    v37 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    while ( v36 != v37 )
    {
      while ( 1 )
      {
        v37 -= 96;
        v38 = *(_QWORD *)(v37 + 16);
        if ( v38 == v37 + 32 )
          break;
        _libc_free(v38);
        if ( v36 == v37 )
          goto LABEL_32;
      }
    }
LABEL_32:
    *(_DWORD *)(a1 + 8) = v49;
  }
  else if ( v16 > v14 )
  {
    v39 = *(_QWORD *)a2 + 96 * v16;
    v40 = *(_QWORD *)a1 + 96 * v14;
    v41 = 96 * v49;
    v42 = (const __m128i *)(96 * v49 + *(_QWORD *)a2);
    if ( v42 == (const __m128i *)v39 )
    {
      v48 = v14;
    }
    else
    {
      do
      {
        if ( v40 )
        {
          v43 = _mm_loadu_si128(v42);
          *(_DWORD *)(v40 + 24) = 0;
          *(_QWORD *)(v40 + 16) = v40 + 32;
          *(_DWORD *)(v40 + 28) = 16;
          *(__m128i *)v40 = v43;
          v44 = v42[1].m128i_u32[2];
          if ( (_DWORD)v44 )
          {
            v52 = v41;
            v55 = v14;
            v57 = v39;
            sub_1FEB680(v40 + 16, (__int64)v42[1].m128i_i64, v39, v44, v14, a6);
            v41 = v52;
            v14 = v55;
            v39 = v57;
          }
        }
        v42 += 6;
        v40 += 96;
      }
      while ( (const __m128i *)v39 != v42 );
      v48 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v48 + v16 - v14;
    v45 = *(_QWORD *)a2 + v41;
    v46 = *(_QWORD *)a2 + 96LL * *(unsigned int *)(a2 + 8);
    while ( v45 != v46 )
    {
      while ( 1 )
      {
        v46 -= 96;
        v47 = *(_QWORD *)(v46 + 16);
        if ( v47 == v46 + 32 )
          break;
        _libc_free(v47);
        if ( v45 == v46 )
          goto LABEL_47;
      }
    }
LABEL_47:
    *(_DWORD *)(a2 + 8) = v49;
  }
}
