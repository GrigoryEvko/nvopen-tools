// Function: sub_332D790
// Address: 0x332d790
//
void __fastcall sub_332D790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v9; // rax
  int v10; // edx
  int v11; // eax
  int v12; // edx
  int v13; // eax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rax
  __int64 v19; // r15
  char **v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  __int64 v26; // rdx
  __m128i v27; // xmm0
  __int64 v28; // r9
  int v29; // edi
  __int64 v30; // r13
  __int64 v31; // rcx
  __int64 v32; // rdx
  const __m128i *v33; // r14
  __m128i v34; // xmm1
  __int64 m128i_i64; // rsi
  __int64 v36; // rdi
  __int64 v37; // r13
  __int64 v38; // r12
  unsigned __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rax
  const __m128i *v43; // r14
  __m128i v44; // xmm2
  __int64 v45; // rcx
  __int64 v46; // r13
  __int64 v47; // rbx
  unsigned __int64 v48; // rdi
  int v49; // edx
  unsigned __int64 v50; // [rsp-B8h] [rbp-B8h]
  __int64 v51; // [rsp-B0h] [rbp-B0h]
  __int64 v52; // [rsp-B0h] [rbp-B0h]
  __int64 v53; // [rsp-B0h] [rbp-B0h]
  __int64 v54; // [rsp-A8h] [rbp-A8h]
  const __m128i *v55; // [rsp-A8h] [rbp-A8h]
  __int64 v56; // [rsp-A8h] [rbp-A8h]
  unsigned __int64 v57; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v58; // [rsp-A0h] [rbp-A0h]
  char *v59[2]; // [rsp-88h] [rbp-88h] BYREF
  _BYTE v60[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a1 == a2 )
    return;
  v6 = a1 + 16;
  v9 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 != a1 + 16 )
  {
    v6 = *(_QWORD *)a2;
    a4 = a2 + 16;
    if ( *(_QWORD *)a2 != a2 + 16 )
    {
      *(_QWORD *)a1 = v6;
      v10 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v9;
      v11 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(a1 + 8) = v10;
      v12 = *(_DWORD *)(a2 + 12);
      *(_DWORD *)(a2 + 8) = v11;
      v13 = *(_DWORD *)(a1 + 12);
      *(_DWORD *)(a1 + 12) = v12;
      *(_DWORD *)(a2 + 12) = v13;
      return;
    }
  }
  v14 = *(unsigned int *)(a2 + 8);
  if ( v14 <= *(unsigned int *)(a1 + 12) )
  {
    v15 = *(unsigned int *)(a1 + 8);
    v16 = v15;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v15 )
      goto LABEL_7;
    goto LABEL_20;
  }
  sub_332D670(a1, v14, v6, a4, a5, a6);
  v15 = *(unsigned int *)(a1 + 8);
  v16 = v15;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v15 )
  {
LABEL_20:
    sub_332D670(a2, v16, v6, a4, v15, a6);
    v15 = *(unsigned int *)(a1 + 8);
    LODWORD(v16) = *(_DWORD *)(a1 + 8);
  }
LABEL_7:
  v17 = *(unsigned int *)(a2 + 8);
  v18 = v15;
  if ( v17 <= v15 )
    v18 = *(unsigned int *)(a2 + 8);
  v50 = v18;
  if ( v18 )
  {
    v19 = 0;
    v54 = 96 * v18;
    do
    {
      v25 = v19 + *(_QWORD *)a2;
      v26 = v19 + *(_QWORD *)a1;
      v27 = _mm_loadu_si128((const __m128i *)v26);
      v28 = v26 + 16;
      *(_QWORD *)v26 = *(_QWORD *)v25;
      *(_DWORD *)(v26 + 8) = *(_DWORD *)(v25 + 8);
      *(_QWORD *)v25 = v27.m128i_i64[0];
      v59[0] = v60;
      *(_DWORD *)(v25 + 8) = v27.m128i_i32[2];
      v29 = *(_DWORD *)(v26 + 24);
      v59[1] = (char *)0x1000000000LL;
      if ( v29 )
      {
        v51 = v26 + 16;
        sub_33256C0((__int64)v59, (char **)(v26 + 16), v26, a4, v15, v28);
        v28 = v51;
      }
      v20 = (char **)(v25 + 16);
      sub_33256C0(v28, v20, v26, a4, v15, v28);
      sub_33256C0((__int64)v20, v59, v21, v22, v23, v24);
      if ( v59[0] != v60 )
        _libc_free((unsigned __int64)v59[0]);
      v19 += 96;
    }
    while ( v54 != v19 );
    v15 = *(unsigned int *)(a1 + 8);
    v17 = *(unsigned int *)(a2 + 8);
    LODWORD(v16) = *(_DWORD *)(a1 + 8);
  }
  if ( v17 < v15 )
  {
    v30 = *(_QWORD *)a2 + 96 * v17;
    v31 = *(_QWORD *)a1 + 96 * v15;
    v32 = 96 * v50;
    v33 = (const __m128i *)(96 * v50 + *(_QWORD *)a1);
    if ( v33 != (const __m128i *)v31 )
    {
      do
      {
        while ( 1 )
        {
          if ( v30 )
          {
            v34 = _mm_loadu_si128(v33);
            *(_DWORD *)(v30 + 24) = 0;
            *(_QWORD *)(v30 + 16) = v30 + 32;
            *(_DWORD *)(v30 + 28) = 16;
            *(__m128i *)v30 = v34;
            if ( v33[1].m128i_i32[2] )
              break;
          }
          v33 += 6;
          v30 += 96;
          if ( (const __m128i *)v31 == v33 )
            goto LABEL_27;
        }
        m128i_i64 = (__int64)v33[1].m128i_i64;
        v36 = v30 + 16;
        v52 = v32;
        v33 += 6;
        v55 = (const __m128i *)v31;
        v30 += 96;
        v57 = v15;
        sub_33255E0(v36, m128i_i64, v32, v31, v15, a6);
        v31 = (__int64)v55;
        v32 = v52;
        v15 = v57;
      }
      while ( v55 != v33 );
LABEL_27:
      LODWORD(v16) = v15 + *(_DWORD *)(a2 + 8) - v17;
    }
    *(_DWORD *)(a2 + 8) = v16;
    v37 = *(_QWORD *)a1 + v32;
    v38 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    while ( v37 != v38 )
    {
      while ( 1 )
      {
        v38 -= 96;
        v39 = *(_QWORD *)(v38 + 16);
        if ( v39 == v38 + 32 )
          break;
        _libc_free(v39);
        if ( v37 == v38 )
          goto LABEL_32;
      }
    }
LABEL_32:
    *(_DWORD *)(a1 + 8) = v50;
  }
  else if ( v17 > v15 )
  {
    v40 = *(_QWORD *)a2 + 96 * v17;
    v41 = *(_QWORD *)a1 + 96 * v15;
    v42 = 96 * v50;
    v43 = (const __m128i *)(96 * v50 + *(_QWORD *)a2);
    if ( v43 == (const __m128i *)v40 )
    {
      v49 = v15;
    }
    else
    {
      do
      {
        if ( v41 )
        {
          v44 = _mm_loadu_si128(v43);
          *(_DWORD *)(v41 + 24) = 0;
          *(_QWORD *)(v41 + 16) = v41 + 32;
          *(_DWORD *)(v41 + 28) = 16;
          *(__m128i *)v41 = v44;
          v45 = v43[1].m128i_u32[2];
          if ( (_DWORD)v45 )
          {
            v53 = v42;
            v56 = v40;
            v58 = v15;
            sub_33255E0(v41 + 16, (__int64)v43[1].m128i_i64, v40, v45, v15, a6);
            v42 = v53;
            v40 = v56;
            v15 = v58;
          }
        }
        v43 += 6;
        v41 += 96;
      }
      while ( (const __m128i *)v40 != v43 );
      v49 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v49 + v17 - v15;
    v46 = *(_QWORD *)a2 + v42;
    v47 = *(_QWORD *)a2 + 96LL * *(unsigned int *)(a2 + 8);
    while ( v46 != v47 )
    {
      while ( 1 )
      {
        v47 -= 96;
        v48 = *(_QWORD *)(v47 + 16);
        if ( v48 == v47 + 32 )
          break;
        _libc_free(v48);
        if ( v46 == v47 )
          goto LABEL_47;
      }
    }
LABEL_47:
    *(_DWORD *)(a2 + 8) = v50;
  }
}
