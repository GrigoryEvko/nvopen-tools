// Function: sub_2F8F1B0
// Address: 0x2f8f1b0
//
__int64 __fastcall sub_2F8F1B0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  const __m128i *v7; // rbx
  __int64 v8; // rcx
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // rcx
  __int64 v12; // rdx
  unsigned int v13; // edi
  __int64 v14; // r9
  unsigned int v15; // r10d
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v25; // r15
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  __m128i si128; // xmm0
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __m128i v37[4]; // [rsp+0h] [rbp-40h] BYREF

  v7 = (const __m128i *)a2;
  v8 = *(unsigned int *)(a1 + 48);
  v9 = *(__int64 **)(a1 + 40);
  v10 = *(_QWORD *)a2;
  v11 = &v9[2 * v8];
  if ( v9 == v11 )
  {
LABEL_16:
    v25 = v7->m128i_i64[1];
    v26 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v10 & 6) == 0 )
    {
      ++*(_DWORD *)(a1 + 208);
      ++*(_DWORD *)(v26 + 212);
    }
    if ( (*(_BYTE *)(v26 + 249) & 4) == 0 )
    {
      if ( ((v7->m128i_i8[0] ^ 6) & 6) != 0 || v7->m128i_i32[2] <= 3u )
        ++*(_DWORD *)(a1 + 216);
      else
        ++*(_DWORD *)(a1 + 224);
    }
    if ( (*(_BYTE *)(a1 + 249) & 4) == 0 )
    {
      if ( ((v7->m128i_i8[0] ^ 6) & 6) != 0 || v7->m128i_i32[2] <= 3u )
        ++*(_DWORD *)(v26 + 220);
      else
        ++*(_DWORD *)(v26 + 228);
    }
    v27 = *(unsigned int *)(a1 + 48);
    si128 = _mm_loadu_si128(v7);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
    {
      a2 = a1 + 56;
      v37[0] = si128;
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v27 + 1, 0x10u, a5, a6);
      v27 = *(unsigned int *)(a1 + 48);
      si128 = _mm_load_si128(v37);
    }
    *(__m128i *)(*(_QWORD *)(a1 + 40) + 16 * v27) = si128;
    ++*(_DWORD *)(a1 + 48);
    v29 = *(unsigned int *)(v26 + 128);
    v30 = *(unsigned int *)(v26 + 132);
    v31 = v29 + 1;
    if ( v29 + 1 > v30 )
    {
      a2 = v26 + 136;
      sub_C8D5F0(v26 + 120, (const void *)(v26 + 136), v31, 0x10u, a5, a6);
      v29 = *(unsigned int *)(v26 + 128);
    }
    v32 = (__int64 *)(*(_QWORD *)(v26 + 120) + 16 * v29);
    *v32 = a1 | v10 & 7;
    v32[1] = v25;
    ++*(_DWORD *)(v26 + 128);
    sub_2F8EFB0(a1, a2, v31, v30, a5, a6);
    sub_2F8F0B0(v26, a2, v33, v34, v35, v36);
    return 1;
  }
  else
  {
    a2 = a3;
    a6 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v12 = *v9;
      if ( !(_BYTE)a2 )
      {
        a5 = v12 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) == a6 )
          return 0;
      }
      if ( v12 == v10 )
      {
        a5 = *((unsigned int *)v9 + 2);
        if ( v7->m128i_i32[2] == (_DWORD)a5 )
          break;
      }
      v9 += 2;
      if ( v11 == v9 )
        goto LABEL_16;
    }
    v13 = *((_DWORD *)v9 + 3);
    v14 = v7->m128i_u32[3];
    v15 = 0;
    if ( v13 < (unsigned int)v14 )
    {
      v16 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      v17 = a1 | *v9 & 7;
      v18 = *(_QWORD *)(v16 + 120);
      v19 = v18 + 16LL * *(unsigned int *)(v16 + 128);
      if ( v19 != v18 )
      {
        while ( __PAIR128__(__PAIR64__(v13, a5), v17) != *(_OWORD *)v18 )
        {
          v18 += 16;
          if ( v19 == v18 )
            goto LABEL_14;
        }
        *(_DWORD *)(v18 + 12) = v14;
      }
LABEL_14:
      *((_DWORD *)v9 + 3) = v14;
      sub_2F8EFB0(a1, v19, v17, v18, a5, v14);
      sub_2F8F0B0(v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, v19, v20, v21, v22, v23);
      return 0;
    }
  }
  return v15;
}
