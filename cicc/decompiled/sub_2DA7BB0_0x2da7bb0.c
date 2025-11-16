// Function: sub_2DA7BB0
// Address: 0x2da7bb0
//
__int64 __fastcall sub_2DA7BB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 v12; // r15
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rcx
  const __m128i *v25; // rdx
  __m128i *v26; // rax
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 v31; // rsi
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  const void *v34; // rsi
  int v35; // eax
  int v36; // eax
  __int64 v37; // rsi
  unsigned int v38; // r14d
  __int64 v39; // rdi
  __int64 v40; // rcx
  _QWORD v41[10]; // [rsp+0h] [rbp-50h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v10 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_3:
    v16 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 32) + 24 * v16 + 8;
  }
  while ( v15 != -4096 )
  {
    if ( !v12 && v15 == -8192 )
      v12 = v14;
    a6 = (unsigned int)(v11 + 1);
    v13 = (v9 - 1) & (v11 + v13);
    v14 = v10 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_21:
    sub_9BAAD0(a1, 2 * v9);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v30 = (v27 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v29 + 16LL * v30;
      v31 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v31 != -4096 )
        {
          if ( !v15 && v31 == -8192 )
            v15 = v12;
          v30 = v28 & (a6 + v30);
          v12 = v29 + 16LL * v30;
          v31 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
    goto LABEL_48;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_9BAAD0(a1, v9);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v38 = v36 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v39 = 0;
      v12 = v37 + 16LL * v38;
      v40 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v40 != -4096 )
        {
          if ( !v39 && v40 == -8192 )
            v39 = v12;
          a6 = (unsigned int)(v15 + 1);
          v38 = v36 & (v15 + v38);
          v12 = v37 + 16LL * v38;
          v40 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v39 )
          v12 = v39;
      }
      goto LABEL_15;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *a2;
  v21 = *(unsigned int *)(a1 + 44);
  v41[1] = 0;
  v41[0] = v20;
  v22 = *(unsigned int *)(a1 + 40);
  v41[2] = 0;
  v23 = v22 + 1;
  if ( v22 + 1 > v21 )
  {
    v32 = *(_QWORD *)(a1 + 32);
    v33 = a1 + 32;
    v34 = (const void *)(a1 + 48);
    if ( v32 > (unsigned __int64)v41 || (unsigned __int64)v41 >= v32 + 24 * v22 )
    {
      sub_C8D5F0(v33, v34, v23, 0x18u, v15, a6);
      v24 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v25 = (const __m128i *)v41;
    }
    else
    {
      sub_C8D5F0(v33, v34, v23, 0x18u, v15, a6);
      v24 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v25 = (const __m128i *)((char *)v41 + v24 - v32);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = (const __m128i *)v41;
  }
  v26 = (__m128i *)(v24 + 24 * v22);
  *v26 = _mm_loadu_si128(v25);
  v26[1].m128i_i64[0] = v25[1].m128i_i64[0];
  v16 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v16 + 1;
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 24 * v16 + 8;
}
