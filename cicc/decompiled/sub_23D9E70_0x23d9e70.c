// Function: sub_23D9E70
// Address: 0x23d9e70
//
__int64 __fastcall sub_23D9E70(__int64 a1, __int64 *a2, const __m128i *a3)
{
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r15d
  __int64 v9; // r9
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r11
  int v14; // eax
  int v15; // ecx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rsi
  __int64 v20; // rcx
  __m128i v21; // xmm0
  __int64 v22; // rax
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // r8
  int v28; // r11d
  __int64 v29; // r10
  __m128i v30; // xmm1
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // rdx
  const __m128i *v33; // rax
  __m128i *v34; // rdx
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // r8
  unsigned int v39; // r14d
  int v40; // r10d
  __int64 v41; // rsi
  __int64 v42; // rdi
  const void *v43; // rsi
  char *v44; // r12
  const __m128i *v45; // [rsp+8h] [rbp-58h]
  const __m128i *v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+10h] [rbp-50h] BYREF
  __m128i v48; // [rsp+18h] [rbp-48h]

  v5 = *a2;
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = v7 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( v5 == *(_QWORD *)v11 )
    return *(_QWORD *)(a1 + 32) + 24LL * *(unsigned int *)(v11 + 8);
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v6 - 1) & (v8 + v10);
    v11 = v7 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v5 == *(_QWORD *)v11 )
      return *(_QWORD *)(a1 + 32) + 24LL * *(unsigned int *)(v11 + 8);
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v6 )
  {
LABEL_22:
    v45 = a3;
    sub_9BAAD0(a1, 2 * v6);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      a3 = v45;
      v26 = (v23 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v25 + 16LL * v26;
      v27 = *(_QWORD *)v9;
      if ( v5 != *(_QWORD *)v9 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -4096 )
        {
          if ( !v29 && v27 == -8192 )
            v29 = v9;
          v26 = v24 & (v28 + v26);
          v9 = v25 + 16LL * v26;
          v27 = *(_QWORD *)v9;
          if ( v5 == *(_QWORD *)v9 )
            goto LABEL_14;
          ++v28;
        }
        if ( v29 )
          v9 = v29;
      }
      goto LABEL_14;
    }
    goto LABEL_51;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v15 <= v6 >> 3 )
  {
    v46 = a3;
    sub_9BAAD0(a1, v6);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 8);
      v38 = 0;
      v39 = v36 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      a3 = v46;
      v40 = 1;
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v37 + 16LL * v39;
      v41 = *(_QWORD *)v9;
      if ( v5 != *(_QWORD *)v9 )
      {
        while ( v41 != -4096 )
        {
          if ( !v38 && v41 == -8192 )
            v38 = v9;
          v39 = v36 & (v40 + v39);
          v9 = v37 + 16LL * v39;
          v41 = *(_QWORD *)v9;
          if ( v5 == *(_QWORD *)v9 )
            goto LABEL_14;
          ++v40;
        }
        if ( v38 )
          v9 = v38;
      }
      goto LABEL_14;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)(v9 + 8) = 0;
  *(_QWORD *)v9 = v5;
  *(_DWORD *)(v9 + 8) = *(_DWORD *)(a1 + 40);
  v16 = *(unsigned int *)(a1 + 40);
  v17 = *(unsigned int *)(a1 + 44);
  v18 = *(_DWORD *)(a1 + 40);
  if ( v16 >= v17 )
  {
    v30 = _mm_loadu_si128(a3);
    v31 = v16 + 1;
    v32 = *(_QWORD *)(a1 + 32);
    v47 = *a2;
    v33 = (const __m128i *)&v47;
    v48 = v30;
    if ( v17 < v16 + 1 )
    {
      v42 = a1 + 32;
      v43 = (const void *)(a1 + 48);
      if ( v32 > (unsigned __int64)&v47 || (unsigned __int64)&v47 >= v32 + 24 * v16 )
      {
        sub_C8D5F0(v42, v43, v31, 0x18u, v31, v9);
        v32 = *(_QWORD *)(a1 + 32);
        v16 = *(unsigned int *)(a1 + 40);
        v33 = (const __m128i *)&v47;
      }
      else
      {
        v44 = (char *)&v47 - v32;
        sub_C8D5F0(v42, v43, v31, 0x18u, v31, v9);
        v32 = *(_QWORD *)(a1 + 32);
        v16 = *(unsigned int *)(a1 + 40);
        v33 = (const __m128i *)&v44[v32];
      }
    }
    v34 = (__m128i *)(v32 + 24 * v16);
    *v34 = _mm_loadu_si128(v33);
    v34[1].m128i_i64[0] = v33[1].m128i_i64[0];
    v19 = *(_QWORD *)(a1 + 32);
    v22 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = v19 + 24 * v16;
    if ( v20 )
    {
      v21 = _mm_loadu_si128(a3);
      *(_QWORD *)v20 = *a2;
      *(__m128i *)(v20 + 8) = v21;
      v18 = *(_DWORD *)(a1 + 40);
      v19 = *(_QWORD *)(a1 + 32);
    }
    v22 = (unsigned int)(v18 + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  return v19 + 24 * v22 - 24;
}
