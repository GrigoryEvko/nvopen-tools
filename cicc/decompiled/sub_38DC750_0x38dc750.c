// Function: sub_38DC750
// Address: 0x38dc750
//
__m128i *__fastcall sub_38DC750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // r15
  unsigned __int64 *v13; // r14
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *i; // rdx
  int v21; // edx
  __int64 v22; // rax
  __m128i v23; // xmm1
  __m128i *result; // rax
  unsigned int v25; // ecx
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // ebx
  unsigned __int64 v32; // r12
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *j; // rdx
  _QWORD *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-58h]
  __m128i v38; // [rsp+10h] [rbp-50h] BYREF
  __m128i v39[4]; // [rsp+20h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a1 + 32);
  v37 = *(_QWORD *)(a1 + 24);
  if ( v37 != v7 )
  {
    v8 = *(_QWORD *)(a1 + 24);
    do
    {
      v9 = *(_QWORD *)(v8 + 40);
      v10 = *(_QWORD *)(v8 + 32);
      if ( v9 != v10 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 24);
          if ( v11 )
            j_j___libc_free_0(v11);
          v10 += 48LL;
        }
        while ( v9 != v10 );
        v10 = *(_QWORD *)(v8 + 32);
      }
      if ( v10 )
        j_j___libc_free_0(v10);
      v8 += 80;
    }
    while ( v7 != v8 );
    *(_QWORD *)(a1 + 32) = v37;
  }
  v12 = *(unsigned __int64 **)(a1 + 48);
  v13 = *(unsigned __int64 **)(a1 + 56);
  *(_QWORD *)(a1 + 72) = 0;
  if ( v12 != v13 )
  {
    v14 = v12;
    do
    {
      v15 = *v14;
      if ( *v14 )
      {
        v16 = *(_QWORD *)(v15 + 72);
        if ( v16 )
          j_j___libc_free_0(v16);
        j_j___libc_free_0(v15);
      }
      ++v14;
    }
    while ( v13 != v14 );
    *(_QWORD *)(a1 + 56) = v12;
  }
  v17 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 80);
  if ( !v17 )
  {
    if ( !*(_DWORD *)(a1 + 100) )
      goto LABEL_26;
    v18 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)v18 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 88));
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
      *(_DWORD *)(a1 + 104) = 0;
      goto LABEL_26;
    }
    goto LABEL_23;
  }
  v25 = 4 * v17;
  v18 = *(unsigned int *)(a1 + 104);
  if ( (unsigned int)(4 * v17) < 0x40 )
    v25 = 64;
  if ( v25 >= (unsigned int)v18 )
  {
LABEL_23:
    v19 = *(_QWORD **)(a1 + 88);
    for ( i = &v19[2 * v18]; i != v19; v19 += 2 )
      *v19 = -8;
    *(_QWORD *)(a1 + 96) = 0;
    goto LABEL_26;
  }
  v26 = *(_QWORD **)(a1 + 88);
  v27 = v17 - 1;
  if ( !v27 )
  {
    v32 = 2048;
    v31 = 128;
LABEL_37:
    j___libc_free_0((unsigned __int64)v26);
    *(_DWORD *)(a1 + 104) = v31;
    v33 = (_QWORD *)sub_22077B0(v32);
    v34 = *(unsigned int *)(a1 + 104);
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 88) = v33;
    for ( j = &v33[2 * v34]; j != v33; v33 += 2 )
    {
      if ( v33 )
        *v33 = -8;
    }
    goto LABEL_26;
  }
  _BitScanReverse(&v27, v27);
  v28 = (unsigned int)(1 << (33 - (v27 ^ 0x1F)));
  if ( (int)v28 < 64 )
    v28 = 64;
  if ( (_DWORD)v28 != (_DWORD)v18 )
  {
    v29 = (4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1);
    v30 = ((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2) | ((((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2)) >> 8);
    v31 = (v30 | (v30 >> 16)) + 1;
    v32 = 16 * ((v30 | (v30 >> 16)) + 1);
    goto LABEL_37;
  }
  *(_QWORD *)(a1 + 96) = 0;
  v36 = &v26[2 * v28];
  do
  {
    if ( v26 )
      *v26 = -8;
    v26 += 2;
  }
  while ( v36 != v26 );
LABEL_26:
  v21 = *(_DWORD *)(a1 + 124);
  *(_DWORD *)(a1 + 120) = 0;
  v22 = 0;
  v38 = 0u;
  v39[0] = 0u;
  if ( !v21 )
  {
    sub_16CD150(a1 + 112, (const void *)(a1 + 128), 0, 32, a5, a6);
    v22 = 32LL * *(unsigned int *)(a1 + 120);
  }
  v23 = _mm_loadu_si128(v39);
  result = (__m128i *)(*(_QWORD *)(a1 + 112) + v22);
  *result = _mm_loadu_si128(&v38);
  result[1] = v23;
  ++*(_DWORD *)(a1 + 120);
  return result;
}
