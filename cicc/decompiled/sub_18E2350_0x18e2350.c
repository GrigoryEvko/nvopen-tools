// Function: sub_18E2350
// Address: 0x18e2350
//
__int64 *__fastcall sub_18E2350(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 *v7; // rsi
  __m128i *v8; // rbx
  char v9; // dl
  const __m128i *v10; // rdi
  __int64 v11; // r12
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __m128i *v14; // r15
  const __m128i *v15; // rdx
  __m128i *v16; // rax
  __m128i *v17; // rbx
  __int64 i; // r15
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 *v22; // rcx
  signed __int64 v23; // [rsp+8h] [rbp-68h]
  unsigned __int64 v24; // [rsp+8h] [rbp-68h]
  __m128i v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  const __m128i *v27; // [rsp+28h] [rbp-48h] BYREF
  const __m128i *v28; // [rsp+30h] [rbp-40h]
  __m128i *v29; // [rsp+38h] [rbp-38h]

  v26 = a3;
  v5 = *a2;
  v27 = 0;
  v6 = *(__int64 **)(a3 + 8);
  v28 = 0;
  v29 = 0;
  if ( *(__int64 **)(a3 + 16) == v6 )
  {
    v21 = *(unsigned int *)(a3 + 28);
    v7 = &v6[v21];
    if ( v6 != v7 )
    {
      v22 = 0;
      do
      {
        if ( v5 == *v6 )
        {
          v8 = 0;
          goto LABEL_3;
        }
        if ( *v6 == -2 )
          v22 = v6;
        ++v6;
      }
      while ( v7 != v6 );
      if ( !v22 )
        goto LABEL_33;
      *v22 = v5;
      v8 = (__m128i *)v28;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      goto LABEL_15;
    }
LABEL_33:
    if ( (unsigned int)v21 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v21 + 1;
      *v7 = v5;
      v8 = (__m128i *)v28;
      ++*(_QWORD *)a3;
      goto LABEL_15;
    }
  }
  v7 = (__int64 *)v5;
  sub_16CCBA0(a3, v5);
  v8 = (__m128i *)v28;
  if ( !v9 )
  {
LABEL_3:
    v10 = v27;
    v11 = v26;
    v12 = (char *)v8 - (char *)v27;
    goto LABEL_22;
  }
LABEL_15:
  for ( i = *(_QWORD *)(v5 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
      break;
  }
  v25.m128i_i64[0] = v5;
  v25.m128i_i64[1] = i;
  if ( v8 == v29 )
  {
    v7 = (__int64 *)v8;
    sub_18E2000(&v27, v8, &v25);
  }
  else
  {
    if ( v8 )
    {
      *v8 = _mm_loadu_si128(&v25);
      v8 = (__m128i *)v28;
    }
    v28 = v8 + 1;
  }
  sub_18E2180((__int64)&v26);
  v8 = (__m128i *)v28;
  v10 = v27;
  v11 = v26;
  v12 = (char *)v28 - (char *)v27;
  if ( v28 != v27 )
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(v27, v7, v20);
    v23 = (char *)v28 - (char *)v27;
    v13 = sub_22077B0(v12);
    v8 = (__m128i *)v28;
    v10 = v27;
    v14 = (__m128i *)v13;
    v12 = v13 + v23;
    if ( v28 == v27 )
      goto LABEL_23;
    goto LABEL_6;
  }
LABEL_22:
  v14 = 0;
  if ( v8 == v10 )
  {
LABEL_23:
    v17 = v14;
    goto LABEL_10;
  }
LABEL_6:
  v15 = v10;
  v16 = v14;
  v17 = (__m128i *)((char *)v14 + (char *)v8 - (char *)v10);
  do
  {
    if ( v16 )
      *v16 = _mm_loadu_si128(v15);
    ++v16;
    ++v15;
  }
  while ( v16 != v17 );
LABEL_10:
  if ( v10 )
  {
    v24 = v12;
    j_j___libc_free_0(v10, (char *)v29 - (char *)v10);
    v12 = v24;
  }
  *a1 = v11;
  a1[1] = (__int64)v14;
  a1[2] = (__int64)v17;
  a1[4] = a3;
  a1[3] = v12;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
