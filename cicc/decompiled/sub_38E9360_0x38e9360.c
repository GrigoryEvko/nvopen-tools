// Function: sub_38E9360
// Address: 0x38e9360
//
unsigned __int64 __fastcall sub_38E9360(unsigned __int64 *a1, __int64 a2, _DWORD *a3, _QWORD *a4)
{
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // rbx
  __int64 v15; // r10
  __int64 v16; // rdi
  __int64 v17; // rax
  int v18; // esi
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rsi
  unsigned int v22; // esi
  const void **v23; // rsi
  __int64 v24; // rax
  __m128i v25; // xmm1
  unsigned int v26; // eax
  const void **v27; // rsi
  __int64 v28; // rdi
  unsigned __int64 i; // r14
  unsigned __int64 v30; // rdi
  unsigned __int64 v32; // rbx
  __int64 v33; // rax
  _QWORD *v34; // [rsp+0h] [rbp-60h]
  _DWORD *v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v5 - *a1) >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * ((__int64)(v5 - v6) >> 3);
  v13 = a2 - v6;
  if ( v11 )
  {
    v32 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v36 = 0;
      v14 = 40;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v32 = 40 * v12;
  }
  v34 = a4;
  v35 = a3;
  v33 = sub_22077B0(v32);
  v13 = a2 - v6;
  v9 = a2;
  a3 = v35;
  a4 = v34;
  v39 = v33;
  v36 = v33 + v32;
  v14 = v33 + 40;
LABEL_7:
  v15 = v39 + v13;
  if ( v15 )
  {
    v16 = *a4;
    v17 = a4[1];
    *(_DWORD *)(v15 + 32) = 64;
    v18 = *a3;
    *(_QWORD *)(v15 + 24) = 0;
    *(_QWORD *)(v15 + 8) = v16;
    *(_DWORD *)v15 = v18;
    *(_QWORD *)(v15 + 16) = v17;
  }
  if ( v9 != v6 )
  {
    v19 = v39;
    v20 = v6;
    while ( 1 )
    {
      if ( !v19 )
        goto LABEL_12;
      *(_DWORD *)v19 = *(_DWORD *)v20;
      *(__m128i *)(v19 + 8) = _mm_loadu_si128((const __m128i *)(v20 + 8));
      v22 = *(_DWORD *)(v20 + 32);
      *(_DWORD *)(v19 + 32) = v22;
      if ( v22 <= 0x40 )
        break;
      v23 = (const void **)(v20 + 24);
      v37 = v9;
      v20 += 40LL;
      v38 = v19;
      sub_16A4FD0(v19 + 24, v23);
      v19 = v38;
      v9 = v37;
      v21 = v38 + 40;
      if ( v37 == v20 )
      {
LABEL_17:
        v14 = v19 + 80;
        goto LABEL_18;
      }
LABEL_13:
      v19 = v21;
    }
    *(_QWORD *)(v19 + 24) = *(_QWORD *)(v20 + 24);
LABEL_12:
    v20 += 40LL;
    v21 = v19 + 40;
    if ( v9 == v20 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( v9 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v25 = _mm_loadu_si128((const __m128i *)(v10 + 8));
        *(_DWORD *)v14 = *(_DWORD *)v10;
        v26 = *(_DWORD *)(v10 + 32);
        *(__m128i *)(v14 + 8) = v25;
        *(_DWORD *)(v14 + 32) = v26;
        if ( v26 > 0x40 )
          break;
        v24 = *(_QWORD *)(v10 + 24);
        v10 += 40;
        v14 += 40;
        *(_QWORD *)(v14 - 16) = v24;
        if ( v5 == v10 )
          goto LABEL_23;
      }
      v27 = (const void **)(v10 + 24);
      v28 = v14 + 24;
      v10 += 40;
      v14 += 40;
      sub_16A4FD0(v28, v27);
    }
    while ( v5 != v10 );
  }
LABEL_23:
  for ( i = v6; i != v5; i += 40LL )
  {
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v30 = *(_QWORD *)(i + 24);
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = v14;
  *a1 = v39;
  a1[2] = v36;
  return v36;
}
