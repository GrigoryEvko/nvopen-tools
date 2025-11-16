// Function: sub_14318B0
// Address: 0x14318b0
//
__int64 __fastcall sub_14318B0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v4; // r15
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  unsigned __int64 v8; // r14
  __int64 v9; // r13
  unsigned int v10; // edx
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  __int64 v13; // rax
  int v15; // eax
  int v16; // edx
  __m128i *v17; // rsi
  __m128i *v18; // rsi
  int v19; // eax
  int v20; // ecx
  __int64 v21; // r8
  unsigned int v22; // edi
  unsigned __int64 v23; // rax
  int v24; // r10d
  __int64 v25; // r9
  int v26; // eax
  int v27; // ecx
  __int64 v28; // rdi
  int v29; // r9d
  __int64 v30; // r8
  unsigned int v31; // esi
  unsigned __int64 v32; // rax
  __m128i v33[4]; // [rsp+0h] [rbp-40h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = 0;
  v10 = v4 & 0xFFFFFFF8 & (v5 - 1);
  v11 = v6 + 16LL * v10;
  v12 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == v12 )
  {
LABEL_3:
    v13 = *(unsigned int *)(v11 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v13 + 8;
  }
  while ( v12 != -8 )
  {
    if ( v12 == -16 && !v9 )
      v9 = v11;
    v10 = (v5 - 1) & (v7 + v10);
    v11 = v6 + 16LL * v10;
    v12 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 == v12 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v9 )
    v9 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_24:
    sub_14316D0(a1, 2 * v5);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = v4 & 0xFFFFFFF8 & (v19 - 1);
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v21 + 16LL * v22;
      v23 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != v23 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v9;
          v22 = v20 & (v24 + v22);
          v9 = v21 + 16LL * v22;
          v23 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == v23 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v9 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_14316D0(a1, v5);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = 1;
      v30 = 0;
      v31 = v8 & (v26 - 1);
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v28 + 16LL * v31;
      v32 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 != v32 )
      {
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v30 )
            v30 = v9;
          v31 = v27 & (v29 + v31);
          v9 = v28 + 16LL * v31;
          v32 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v8 == v32 )
            goto LABEL_15;
          ++v29;
        }
        if ( v30 )
          v9 = v30;
      }
      goto LABEL_15;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( (*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  *(_DWORD *)(v9 + 8) = 0;
  v17 = *(__m128i **)(a1 + 40);
  v33[0] = (__m128i)*a2;
  if ( v17 == *(__m128i **)(a1 + 48) )
  {
    sub_142DD90((const __m128i **)(a1 + 32), v17, v33);
    v18 = *(__m128i **)(a1 + 40);
  }
  else
  {
    if ( v17 )
    {
      *v17 = _mm_loadu_si128(v33);
      v17 = *(__m128i **)(a1 + 40);
    }
    v18 = v17 + 1;
    *(_QWORD *)(a1 + 40) = v18;
  }
  v13 = (unsigned int)(((__int64)v18->m128i_i64 - *(_QWORD *)(a1 + 32)) >> 4) - 1;
  *(_DWORD *)(v9 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 16 * v13 + 8;
}
