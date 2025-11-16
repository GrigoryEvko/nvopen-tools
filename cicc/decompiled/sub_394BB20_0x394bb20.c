// Function: sub_394BB20
// Address: 0x394bb20
//
unsigned __int64 __fastcall sub_394BB20(unsigned __int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // rcx
  __m128i *v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  __int64 v12; // r8
  char *v13; // rax
  __int64 v14; // rsi
  __m128i v15; // xmm0
  __m128i v16; // xmm3
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // r15
  __int64 i; // r14
  int v21; // eax
  __m128i v22; // xmm2
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  void (__fastcall *v27)(unsigned __int64, unsigned __int64, __int64, __int64, __int64); // rax
  __m128i *v28; // rax
  __int64 v29; // rdx
  __int32 v30; // ecx
  __m128i v31; // xmm1
  __m128i v32; // xmm0
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rcx
  unsigned __int64 v36; // rdi
  unsigned __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h]
  unsigned __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v3 = a3;
  v5 = (__m128i *)a1[1];
  v42 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[1] - *a1) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x3333333333333333LL * ((__int64)(a1[1] - *a1) >> 3);
  v11 = &a2->m128i_i8[-v42];
  if ( v9 )
  {
    v38 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v41 = 0;
      v12 = 40;
      v44 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x333333333333333LL )
      v10 = 0x333333333333333LL;
    v38 = 40 * v10;
  }
  v40 = v3;
  v39 = sub_22077B0(v38);
  v11 = &a2->m128i_i8[-v42];
  v44 = v39;
  v3 = v40;
  v12 = v39 + 40;
  v41 = v39 + v38;
LABEL_7:
  v13 = &v11[v44];
  if ( &v11[v44] )
  {
    v14 = *((_QWORD *)v13 + 4);
    v15 = _mm_loadu_si128((const __m128i *)(v3 + 8));
    v16 = _mm_loadu_si128((const __m128i *)(v13 + 8));
    *(_DWORD *)v13 = *(_DWORD *)v3;
    v17 = *(_QWORD *)(v3 + 24);
    *(_QWORD *)(v3 + 24) = 0;
    *((_QWORD *)v13 + 3) = v17;
    v18 = *(_QWORD *)(v3 + 32);
    *(_QWORD *)(v3 + 32) = v14;
    *((_QWORD *)v13 + 4) = v18;
    *(__m128i *)(v3 + 8) = v16;
    *(__m128i *)(v13 + 8) = v15;
  }
  v19 = v42;
  if ( a2 != (__m128i *)v42 )
  {
    for ( i = v44; ; i += 40 )
    {
      if ( i )
      {
        v21 = *(_DWORD *)v19;
        v22 = _mm_loadu_si128((const __m128i *)(i + 8));
        *(_QWORD *)(i + 24) = 0;
        *(_DWORD *)i = v21;
        v23 = _mm_loadu_si128((const __m128i *)(v19 + 8));
        *(__m128i *)(v19 + 8) = v22;
        *(__m128i *)(i + 8) = v23;
        v24 = *(_QWORD *)(v19 + 24);
        *(_QWORD *)(v19 + 24) = 0;
        v25 = *(_QWORD *)(i + 32);
        *(_QWORD *)(i + 24) = v24;
        v26 = *(_QWORD *)(v19 + 32);
        *(_QWORD *)(v19 + 32) = v25;
        *(_QWORD *)(i + 32) = v26;
      }
      v27 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64, __int64, __int64))(v19 + 24);
      if ( v27 )
        v27(v19 + 8, v19 + 8, 3, v3, v12);
      v19 += 40LL;
      if ( (__m128i *)v19 == a2 )
        break;
    }
    v12 = i + 80;
  }
  if ( a2 != v5 )
  {
    v28 = a2;
    v29 = v12;
    do
    {
      v30 = v28->m128i_i32[0];
      v31 = _mm_loadu_si128((const __m128i *)(v29 + 8));
      v28 = (__m128i *)((char *)v28 + 40);
      v29 += 40;
      v32 = _mm_loadu_si128(v28 - 2);
      v33 = *(_QWORD *)(v29 - 8);
      *(_DWORD *)(v29 - 40) = v30;
      v34 = v28[-1].m128i_i64[0];
      v28[-2] = v31;
      *(_QWORD *)(v29 - 16) = v34;
      v35 = v28[-1].m128i_i64[1];
      *(__m128i *)(v29 - 32) = v32;
      v28[-1].m128i_i64[0] = 0;
      v28[-1].m128i_i64[1] = v33;
      *(_QWORD *)(v29 - 8) = v35;
    }
    while ( v28 != v5 );
    v12 += 8 * ((unsigned __int64)((char *)v28 - (char *)a2 - 40) >> 3) + 40;
  }
  v36 = v42;
  if ( v42 )
  {
    v43 = v12;
    j_j___libc_free_0(v36);
    v12 = v43;
  }
  a1[1] = v12;
  *a1 = v44;
  a1[2] = v41;
  return v41;
}
