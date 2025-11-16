// Function: sub_29C5520
// Address: 0x29c5520
//
__int64 __fastcall sub_29C5520(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // r13d
  __int64 v5; // r9
  __int64 v6; // r15
  int v7; // eax
  __int64 v8; // r14
  unsigned int v9; // r13d
  int v10; // eax
  size_t v11; // rdx
  unsigned int i; // ecx
  __int64 v13; // r8
  const void *v14; // rsi
  int v15; // eax
  __int64 v16; // rax
  int v18; // eax
  __int64 v19; // rdx
  __m128i v20; // xmm2
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r8
  __int64 v23; // rcx
  __m128i *v24; // rax
  __m128i *v25; // rdx
  int v26; // ecx
  unsigned __int64 v27; // r12
  __int64 v28; // rdi
  const void *v29; // rsi
  __int64 v30; // [rsp+8h] [rbp-98h]
  size_t v31; // [rsp+10h] [rbp-90h]
  int v32; // [rsp+28h] [rbp-78h]
  unsigned int v33; // [rsp+2Ch] [rbp-74h]
  size_t n[2]; // [rsp+30h] [rbp-70h] BYREF
  int v35; // [rsp+40h] [rbp-60h]
  __m128i v36; // [rsp+50h] [rbp-50h] BYREF
  __int64 v37; // [rsp+60h] [rbp-40h]
  __int64 v38; // [rsp+68h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  v35 = 0;
  v36 = _mm_loadu_si128(a2);
  *(__m128i *)n = v36;
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v36.m128i_i64[0] = 0;
LABEL_3:
    sub_1253750(a1, 2 * v4);
    goto LABEL_4;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v4 - 1;
  v6 = 0;
  v10 = sub_C94890((_QWORD *)n[0], n[1]);
  v11 = n[1];
  v5 = 1;
  for ( i = v9 & v10; ; i = v9 & v26 )
  {
    v13 = v8 + 24LL * i;
    v14 = *(const void **)v13;
    if ( *(_QWORD *)v13 == -1 )
      break;
    if ( v14 == (const void *)-2LL )
    {
      if ( n[0] == -2 )
        goto LABEL_11;
    }
    else
    {
      if ( v11 != *(_QWORD *)(v13 + 8) )
        goto LABEL_27;
      v32 = v5;
      v33 = i;
      if ( !v11 )
        goto LABEL_11;
      v30 = v8 + 24LL * i;
      v31 = v11;
      v15 = memcmp((const void *)n[0], v14, v11);
      i = v33;
      LODWORD(v5) = v32;
      v11 = v31;
      v13 = v30;
      if ( !v15 )
        goto LABEL_11;
    }
    if ( !v6 && v14 == (const void *)-2LL )
      v6 = v13;
LABEL_27:
    v26 = v5 + i;
    v5 = (unsigned int)(v5 + 1);
  }
  if ( n[0] == -1 )
  {
LABEL_11:
    v16 = *(unsigned int *)(v13 + 16);
    return *(_QWORD *)(a1 + 32) + 32 * v16 + 16;
  }
  v18 = *(_DWORD *)(a1 + 16);
  v4 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
    v6 = v8 + 24LL * i;
  ++*(_QWORD *)a1;
  v7 = v18 + 1;
  v36.m128i_i64[0] = v6;
  if ( 4 * v7 >= 3 * v4 )
    goto LABEL_3;
  if ( v4 - (v7 + *(_DWORD *)(a1 + 20)) > v4 >> 3 )
    goto LABEL_18;
  sub_1253750(a1, v4);
LABEL_4:
  sub_262D160(a1, (__int64)n, &v36);
  v6 = v36.m128i_i64[0];
  v7 = *(_DWORD *)(a1 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 16) = v7;
  if ( *(_QWORD *)v6 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v6 = _mm_loadu_si128((const __m128i *)n);
  *(_DWORD *)(v6 + 16) = v35;
  v19 = *(unsigned int *)(a1 + 40);
  v20 = _mm_loadu_si128(a2);
  v21 = *(unsigned int *)(a1 + 44);
  v37 = 0;
  v22 = v19 + 1;
  v38 = 0;
  v36 = v20;
  if ( v19 + 1 > v21 )
  {
    v27 = *(_QWORD *)(a1 + 32);
    v28 = a1 + 32;
    v29 = (const void *)(a1 + 48);
    if ( v27 > (unsigned __int64)&v36 || (unsigned __int64)&v36 >= v27 + 32 * v19 )
    {
      sub_C8D5F0(v28, v29, v22, 0x20u, v22, v5);
      v23 = *(_QWORD *)(a1 + 32);
      v19 = *(unsigned int *)(a1 + 40);
      v24 = &v36;
    }
    else
    {
      sub_C8D5F0(v28, v29, v22, 0x20u, v22, v5);
      v23 = *(_QWORD *)(a1 + 32);
      v19 = *(unsigned int *)(a1 + 40);
      v24 = (__m128i *)((char *)&v36 + v23 - v27);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 32);
    v24 = &v36;
  }
  v25 = (__m128i *)(v23 + 32 * v19);
  *v25 = _mm_loadu_si128(v24);
  v25[1] = _mm_loadu_si128(v24 + 1);
  v16 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v16 + 1;
  *(_DWORD *)(v6 + 16) = v16;
  return *(_QWORD *)(a1 + 32) + 32 * v16 + 16;
}
