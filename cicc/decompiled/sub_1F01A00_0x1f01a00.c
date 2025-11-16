// Function: sub_1F01A00
// Address: 0x1f01a00
//
__int64 __fastcall sub_1F01A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned __int64 a6)
{
  const __m128i *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdi
  unsigned __int32 v11; // edi
  unsigned __int32 v12; // r9d
  unsigned int v13; // r14d
  __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d

  v7 = (const __m128i *)a2;
  v8 = *(_QWORD *)(a1 + 32);
  v9 = v8 + 16LL * *(unsigned int *)(a1 + 40);
  if ( v8 == v9 )
  {
LABEL_18:
    v17 = v7->m128i_u64[1];
    v18 = v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    v19 = a1 | v7->m128i_i64[0] & 7;
    if ( (v7->m128i_i64[0] & 6) == 0 )
    {
      ++*(_DWORD *)(a1 + 200);
      ++*(_DWORD *)(v18 + 204);
    }
    if ( (*(_BYTE *)(v18 + 229) & 4) == 0 )
    {
      if ( ((v7->m128i_i8[0] ^ 6) & 6) != 0 || v7->m128i_i32[2] <= 3u )
        ++*(_DWORD *)(a1 + 208);
      else
        ++*(_DWORD *)(a1 + 216);
    }
    if ( (*(_BYTE *)(a1 + 229) & 4) == 0 )
    {
      if ( ((v7->m128i_i8[0] ^ 6) & 6) != 0 || v7->m128i_i32[2] <= 3u )
        ++*(_DWORD *)(v18 + 212);
      else
        ++*(_DWORD *)(v18 + 220);
    }
    v20 = *(unsigned int *)(a1 + 40);
    if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 44) )
    {
      a2 = a1 + 48;
      sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 16, a5, a6);
      v20 = *(unsigned int *)(a1 + 40);
    }
    *(__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v20) = _mm_loadu_si128(v7);
    ++*(_DWORD *)(a1 + 40);
    v21 = *(unsigned int *)(v18 + 120);
    if ( (unsigned int)v21 >= *(_DWORD *)(v18 + 124) )
    {
      a2 = v18 + 128;
      sub_16CD150(v18 + 112, (const void *)(v18 + 128), 0, 16, a5, a6);
      v21 = *(unsigned int *)(v18 + 120);
    }
    v22 = (__int64 *)(*(_QWORD *)(v18 + 112) + 16 * v21);
    *v22 = v19;
    v13 = 1;
    v22[1] = v17;
    ++*(_DWORD *)(v18 + 120);
    if ( HIDWORD(v17) )
    {
      sub_1F01800(a1, a2, a3, v9, a5, a6);
      sub_1F01900(v18, a2, v23, v24, v25, v26);
    }
  }
  else
  {
    v10 = *(_QWORD *)a2;
    a2 = (unsigned int)a3;
    a6 = v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      a3 = *(_QWORD *)v8;
      if ( !(_BYTE)a2 )
      {
        a5 = a3 & 0xFFFFFFF8;
        if ( a6 == (a3 & 0xFFFFFFFFFFFFFFF8LL) )
          return 0;
      }
      if ( a3 == v10 )
      {
        a5 = *(_DWORD *)(v8 + 8);
        if ( a5 == v7->m128i_i32[2] )
          break;
      }
      v8 += 16;
      if ( v9 == v8 )
        goto LABEL_18;
    }
    v11 = *(_DWORD *)(v8 + 12);
    v12 = v7->m128i_u32[3];
    v13 = 0;
    if ( v12 > v11 )
    {
      v14 = *(_QWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 112);
      v15 = v14 + 16LL * *(unsigned int *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 120);
      if ( v14 != v15 )
      {
        while ( (a1 | *(_QWORD *)v8 & 7LL) != *(_QWORD *)v14
             || a5 != *(_DWORD *)(v14 + 8)
             || v11 != *(_DWORD *)(v14 + 12) )
        {
          v14 += 16;
          if ( v15 == v14 )
            goto LABEL_16;
        }
        *(_DWORD *)(v14 + 12) = v12;
      }
LABEL_16:
      *(_DWORD *)(v8 + 12) = v12;
      return 0;
    }
  }
  return v13;
}
