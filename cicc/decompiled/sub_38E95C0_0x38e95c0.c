// Function: sub_38E95C0
// Address: 0x38e95c0
//
unsigned __int64 __fastcall sub_38E95C0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r15
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // rdi
  __m128i v15; // xmm2
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rsi
  unsigned int v20; // esi
  const void **v21; // rsi
  __int64 v22; // rax
  __m128i v23; // xmm1
  unsigned int v24; // eax
  const void **v25; // rsi
  __int64 v26; // rdi
  unsigned __int64 i; // r15
  unsigned __int64 v28; // rdi
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-58h]
  unsigned __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  unsigned __int64 v37; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - v5) >> 3);
  v9 = a2;
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x3333333333333333LL * ((__int64)(v4 - v5) >> 3);
  v12 = a2 - v5;
  if ( v10 )
  {
    v30 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v33 = 0;
      v13 = 40;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x333333333333333LL )
      v11 = 0x333333333333333LL;
    v30 = 40 * v11;
  }
  v32 = a3;
  v31 = sub_22077B0(v30);
  v12 = a2 - v5;
  v8 = a2;
  v37 = v31;
  a3 = v32;
  v33 = v31 + v30;
  v13 = v31 + 40;
LABEL_7:
  v14 = v37 + v12;
  if ( v37 + v12 )
  {
    v15 = _mm_loadu_si128((const __m128i *)(a3 + 8));
    *(_DWORD *)v14 = *(_DWORD *)a3;
    v16 = *(_DWORD *)(a3 + 32);
    *(__m128i *)(v14 + 8) = v15;
    *(_DWORD *)(v14 + 32) = v16;
    if ( v16 > 0x40 )
    {
      v36 = v8;
      sub_16A4FD0(v14 + 24, (const void **)(a3 + 24));
      v8 = v36;
    }
    else
    {
      *(_QWORD *)(v14 + 24) = *(_QWORD *)(a3 + 24);
    }
  }
  if ( v8 != v5 )
  {
    v17 = v37;
    v18 = v5;
    while ( 1 )
    {
      if ( !v17 )
        goto LABEL_13;
      *(_DWORD *)v17 = *(_DWORD *)v18;
      *(__m128i *)(v17 + 8) = _mm_loadu_si128((const __m128i *)(v18 + 8));
      v20 = *(_DWORD *)(v18 + 32);
      *(_DWORD *)(v17 + 32) = v20;
      if ( v20 <= 0x40 )
        break;
      v21 = (const void **)(v18 + 24);
      v34 = v8;
      v18 += 40LL;
      v35 = v17;
      sub_16A4FD0(v17 + 24, v21);
      v17 = v35;
      v8 = v34;
      v19 = v35 + 40;
      if ( v34 == v18 )
      {
LABEL_18:
        v13 = v17 + 80;
        goto LABEL_19;
      }
LABEL_14:
      v17 = v19;
    }
    *(_QWORD *)(v17 + 24) = *(_QWORD *)(v18 + 24);
LABEL_13:
    v18 += 40LL;
    v19 = v17 + 40;
    if ( v8 == v18 )
      goto LABEL_18;
    goto LABEL_14;
  }
LABEL_19:
  if ( v8 != v4 )
  {
    do
    {
      while ( 1 )
      {
        v23 = _mm_loadu_si128((const __m128i *)(v9 + 8));
        *(_DWORD *)v13 = *(_DWORD *)v9;
        v24 = *(_DWORD *)(v9 + 32);
        *(__m128i *)(v13 + 8) = v23;
        *(_DWORD *)(v13 + 32) = v24;
        if ( v24 > 0x40 )
          break;
        v22 = *(_QWORD *)(v9 + 24);
        v9 += 40;
        v13 += 40;
        *(_QWORD *)(v13 - 16) = v22;
        if ( v4 == v9 )
          goto LABEL_24;
      }
      v25 = (const void **)(v9 + 24);
      v26 = v13 + 24;
      v9 += 40;
      v13 += 40;
      sub_16A4FD0(v26, v25);
    }
    while ( v4 != v9 );
  }
LABEL_24:
  for ( i = v5; i != v4; i += 40LL )
  {
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v28 = *(_QWORD *)(i + 24);
      if ( v28 )
        j_j___libc_free_0_0(v28);
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  a1[1] = v13;
  *a1 = v37;
  a1[2] = v33;
  return v33;
}
