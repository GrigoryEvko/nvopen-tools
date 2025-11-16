// Function: sub_E71020
// Address: 0xe71020
//
__int64 __fastcall sub_E71020(__int64 a1, _QWORD *a2, size_t a3, unsigned int a4, int a5, unsigned int a6)
{
  __m128i v10; // xmm0
  __int64 result; // rax
  int v12; // ecx
  unsigned int v13; // esi
  int v14; // edx
  unsigned int v15; // ecx
  __int64 v16; // rdi
  __m128i *v17; // r10
  int v18; // eax
  __int64 v19; // r15
  int v20; // eax
  int v21; // r11d
  int v22; // ecx
  unsigned int i; // r9d
  __m128i *v24; // r8
  const void *v25; // rsi
  unsigned int v26; // r9d
  int v27; // eax
  int v28; // eax
  __m128i *v29; // [rsp+0h] [rbp-C0h]
  __m128i *v30; // [rsp+8h] [rbp-B8h]
  int v31; // [rsp+14h] [rbp-ACh]
  unsigned int v32; // [rsp+18h] [rbp-A8h]
  int v33; // [rsp+1Ch] [rbp-A4h]
  __int64 v34; // [rsp+28h] [rbp-98h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  int v36; // [rsp+28h] [rbp-98h]
  size_t n[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v38; // [rsp+40h] [rbp-80h] BYREF
  __int64 v39; // [rsp+48h] [rbp-78h] BYREF
  __m128i v40; // [rsp+58h] [rbp-68h]
  unsigned __int64 v41; // [rsp+70h] [rbp-50h] BYREF
  __m128i v42; // [rsp+78h] [rbp-48h] BYREF
  int v43; // [rsp+88h] [rbp-38h]

  n[0] = (size_t)a2;
  n[1] = a3;
  if ( a5 != -1 )
  {
    if ( (a4 & 0x10) != 0 )
      goto LABEL_3;
    result = sub_E65440(a1, a2, a3);
    if ( (_BYTE)result )
      goto LABEL_3;
    return result;
  }
  v15 = *(_DWORD *)(a1 + 2464);
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 2440);
    v16 = a1 + 2440;
    v41 = 0;
    goto LABEL_14;
  }
  v19 = *(_QWORD *)(a1 + 2448);
  v36 = *(_DWORD *)(a1 + 2464);
  v20 = sub_C94890(a2, a3);
  v17 = 0;
  v21 = 1;
  v22 = v36 - 1;
  for ( i = (v36 - 1) & v20; ; i = v22 & v26 )
  {
    v24 = (__m128i *)(v19 + 16LL * i);
    v25 = (const void *)v24->m128i_i64[0];
    if ( v24->m128i_i64[0] == -1 )
      break;
    if ( v25 == (const void *)-2LL )
    {
      if ( n[0] == -2 )
        goto LABEL_3;
    }
    else
    {
      if ( n[1] != v24->m128i_i64[1] )
        goto LABEL_26;
      if ( !n[1] )
        goto LABEL_3;
      v29 = (__m128i *)(v19 + 16LL * i);
      v30 = v17;
      v31 = v21;
      v32 = i;
      v33 = v22;
      v28 = memcmp((const void *)n[0], v25, n[1]);
      v22 = v33;
      i = v32;
      v21 = v31;
      v17 = v30;
      v24 = v29;
      if ( !v28 )
        goto LABEL_3;
    }
    if ( v25 == (const void *)-2LL && !v17 )
      v17 = v24;
LABEL_26:
    v26 = v21 + i;
    ++v21;
  }
  if ( n[0] == -1 )
    goto LABEL_3;
  v27 = *(_DWORD *)(a1 + 2456);
  v15 = *(_DWORD *)(a1 + 2464);
  v16 = a1 + 2440;
  if ( !v17 )
    v17 = (__m128i *)(v19 + 16LL * i);
  ++*(_QWORD *)(a1 + 2440);
  v18 = v27 + 1;
  v41 = (unsigned __int64)v17;
  if ( 4 * v18 >= 3 * v15 )
  {
LABEL_14:
    v35 = v16;
    sub_BA8070(v16, 2 * v15);
    goto LABEL_15;
  }
  if ( v15 - (v18 + *(_DWORD *)(a1 + 2460)) <= v15 >> 3 )
  {
    v35 = a1 + 2440;
    sub_BA8070(v16, v15);
LABEL_15:
    sub_B9B010(v35, n, &v41);
    v17 = (__m128i *)v41;
    v18 = *(_DWORD *)(a1 + 2456) + 1;
  }
  *(_DWORD *)(a1 + 2456) = v18;
  if ( v17->m128i_i64[0] != -1 )
    --*(_DWORD *)(a1 + 2460);
  *v17 = _mm_load_si128((const __m128i *)n);
LABEL_3:
  v10 = _mm_loadu_si128((const __m128i *)n);
  v41 = __PAIR64__(a4, a6);
  v34 = a1 + 2408;
  v43 = a5;
  v40 = v10;
  v42 = v10;
  result = sub_E6EBC0(a1 + 2408, (__int64)&v41, &v38);
  if ( !(_BYTE)result )
  {
    v12 = *(_DWORD *)(a1 + 2424);
    v13 = *(_DWORD *)(a1 + 2432);
    result = v38;
    ++*(_QWORD *)(a1 + 2408);
    v14 = v12 + 1;
    v39 = result;
    if ( 4 * (v12 + 1) >= 3 * v13 )
    {
      v13 *= 2;
    }
    else
    {
      if ( v13 - *(_DWORD *)(a1 + 2428) - v14 > v13 >> 3 )
        goto LABEL_7;
      v34 = a1 + 2408;
    }
    sub_E70E30(a1 + 2408, v13);
    sub_E6EBC0(v34, (__int64)&v41, &v39);
    v14 = *(_DWORD *)(a1 + 2424) + 1;
    result = v39;
LABEL_7:
    *(_DWORD *)(a1 + 2424) = v14;
    if ( *(_QWORD *)(result + 8) != -1 || *(_DWORD *)(result + 4) != -1 || *(_DWORD *)result != -1 )
      --*(_DWORD *)(a1 + 2428);
    *(__m128i *)(result + 8) = _mm_loadu_si128(&v42);
    *(_QWORD *)result = v41;
    *(_DWORD *)(result + 24) = v43;
  }
  return result;
}
