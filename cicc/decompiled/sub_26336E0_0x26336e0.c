// Function: sub_26336E0
// Address: 0x26336e0
//
__int64 __fastcall sub_26336E0(__int64 a1, const __m128i *a2, const __m128i *a3)
{
  unsigned int v6; // r12d
  int v7; // esi
  __int64 v8; // rcx
  int v9; // eax
  __int64 v10; // r15
  unsigned int v11; // r12d
  int v12; // eax
  void *v13; // rdx
  int v14; // r10d
  unsigned int i; // r8d
  __int64 v16; // r9
  const void *v17; // rsi
  int v18; // eax
  int v20; // eax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __m128i v26; // xmm3
  __int64 v27; // rax
  unsigned int v28; // r8d
  __int64 v29; // r8
  unsigned __int64 v30; // rsi
  const __m128i *v31; // rdi
  __m128i v32; // xmm5
  __m128i *v33; // rdx
  __int64 v34; // r9
  char *v35; // r12
  __int64 v36; // [rsp+0h] [rbp-B0h]
  void *v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+20h] [rbp-90h]
  int v39; // [rsp+28h] [rbp-88h]
  unsigned int v40; // [rsp+2Ch] [rbp-84h]
  __int64 v41; // [rsp+38h] [rbp-78h] BYREF
  __m128i v42; // [rsp+40h] [rbp-70h]
  void *s1[2]; // [rsp+60h] [rbp-50h] BYREF
  __m128i v44; // [rsp+70h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 24);
  v44.m128i_i32[0] = 0;
  v42 = _mm_loadu_si128(a2);
  *(__m128i *)s1 = v42;
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    v41 = 0;
LABEL_3:
    v7 = 2 * v6;
    goto LABEL_4;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = v6 - 1;
  v12 = sub_C94890((_QWORD *)s1[0], (__int64)s1[1]);
  v13 = s1[1];
  v8 = 0;
  v14 = 1;
  for ( i = v11 & v12; ; i = v11 & v28 )
  {
    v16 = v10 + 24LL * i;
    v17 = *(const void **)v16;
    if ( *(_QWORD *)v16 == -1 )
      break;
    if ( v17 == (const void *)-2LL )
    {
      if ( s1[0] == (void *)-2LL )
        return *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(v16 + 16);
    }
    else
    {
      if ( *(void **)(v16 + 8) != v13 )
        goto LABEL_28;
      v38 = v8;
      v39 = v14;
      v40 = i;
      if ( !v13 )
        return *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(v16 + 16);
      v36 = v10 + 24LL * i;
      v37 = v13;
      v18 = memcmp(s1[0], v17, (size_t)v13);
      v13 = v37;
      v16 = v36;
      i = v40;
      v14 = v39;
      v8 = v38;
      if ( !v18 )
        return *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(v16 + 16);
    }
    if ( v17 == (const void *)-2LL && !v8 )
      v8 = v16;
LABEL_28:
    v28 = v14 + i;
    ++v14;
  }
  if ( s1[0] == (void *)-1LL )
    return *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(v16 + 16);
  v20 = *(_DWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
    v8 = v10 + 24LL * i;
  ++*(_QWORD *)a1;
  v9 = v20 + 1;
  v41 = v8;
  if ( 4 * v9 >= 3 * v6 )
    goto LABEL_3;
  if ( v6 - (v9 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
    goto LABEL_17;
  v7 = v6;
LABEL_4:
  sub_1253750(a1, v7);
  sub_262D160(a1, (__int64)s1, &v41);
  v8 = v41;
  v9 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 16) = v9;
  if ( *(_QWORD *)v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)s1);
  *(_DWORD *)(v8 + 16) = v44.m128i_i32[0];
  *(_DWORD *)(v8 + 16) = *(_DWORD *)(a1 + 40);
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *(unsigned int *)(a1 + 44);
  v23 = *(_DWORD *)(a1 + 40);
  if ( v21 >= v22 )
  {
    v29 = v21 + 1;
    v30 = *(_QWORD *)(a1 + 32);
    v31 = (const __m128i *)s1;
    v32 = _mm_loadu_si128(a3);
    *(__m128i *)s1 = _mm_loadu_si128(a2);
    v44 = v32;
    if ( v22 < v21 + 1 )
    {
      v34 = a1 + 32;
      if ( v30 > (unsigned __int64)s1 || (unsigned __int64)s1 >= v30 + 32 * v21 )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v21 + 1, 0x20u, v29, v34);
        v30 = *(_QWORD *)(a1 + 32);
        v21 = *(unsigned int *)(a1 + 40);
        v31 = (const __m128i *)s1;
      }
      else
      {
        v35 = (char *)s1 - v30;
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v21 + 1, 0x20u, v29, v34);
        v30 = *(_QWORD *)(a1 + 32);
        v21 = *(unsigned int *)(a1 + 40);
        v31 = (const __m128i *)&v35[v30];
      }
    }
    v33 = (__m128i *)(v30 + 32 * v21);
    *v33 = _mm_loadu_si128(v31);
    v33[1] = _mm_loadu_si128(v31 + 1);
    v24 = *(_QWORD *)(a1 + 32);
    v27 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v27;
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = (__m128i *)(v24 + 32 * v21);
    if ( v25 )
    {
      v26 = _mm_loadu_si128(a3);
      *v25 = _mm_loadu_si128(a2);
      v25[1] = v26;
      v23 = *(_DWORD *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
    }
    v27 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 40) = v27;
  }
  return v24 + 32 * v27 - 32;
}
