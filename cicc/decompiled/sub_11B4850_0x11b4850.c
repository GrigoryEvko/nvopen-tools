// Function: sub_11B4850
// Address: 0x11b4850
//
unsigned __int8 *__fastcall sub_11B4850(const __m128i *a1, __int64 a2)
{
  _DWORD *v4; // rdx
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __m128i v7; // xmm3
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v14; // rax
  const void *v15; // rsi
  __int64 v16; // rcx
  int v17; // r15d
  __int64 v18; // r14
  size_t v19; // rdx
  __int64 v20; // rbx
  _QWORD *v21; // rdi
  int v22; // eax
  size_t v23; // [rsp+0h] [rbp-90h]
  _OWORD v24[2]; // [rsp+10h] [rbp-80h] BYREF
  __m128i v25; // [rsp+30h] [rbp-60h]
  __m128i v26; // [rsp+40h] [rbp-50h]
  __int64 v27; // [rsp+50h] [rbp-40h]

  v4 = *(_DWORD **)(a2 + 72);
  v5 = _mm_loadu_si128(a1 + 6);
  v6 = _mm_loadu_si128(a1 + 7);
  v7 = _mm_loadu_si128(a1 + 9);
  v8 = a1[10].m128i_i64[0];
  v25 = _mm_loadu_si128(a1 + 8);
  v9 = *(unsigned int *)(a2 + 80);
  v10 = *(_QWORD *)(a2 - 64);
  v25.m128i_i64[1] = a2;
  v11 = *(_QWORD *)(a2 - 32);
  v27 = v8;
  v24[0] = v5;
  v24[1] = v6;
  v26 = v7;
  v12 = sub_10047F0(v10, v11, v4, v9, (__int64)v24);
  if ( v12 )
    return sub_F162A0((__int64)a1, a2, v12);
  v14 = *(_QWORD *)(a2 + 16);
  if ( v14 && !*(_QWORD *)(v14 + 8) )
  {
    v15 = *(const void **)(a2 + 72);
    v16 = a2;
    v17 = 0;
    v18 = *(unsigned int *)(a2 + 80);
    v19 = 4 * v18;
    while ( 1 )
    {
      v20 = *(_QWORD *)(v14 + 24);
      if ( *(_BYTE *)v20 != 94 )
        break;
      v21 = (*(_BYTE *)(v20 + 7) & 0x40) != 0
          ? *(_QWORD **)(v20 - 8)
          : (_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF));
      if ( *v21 != v16 )
        break;
      if ( (_DWORD)v18 == *(_DWORD *)(v20 + 80) )
      {
        if ( !v19 || (v23 = v19, v22 = memcmp(*(const void **)(v20 + 72), v15, v19), v19 = v23, !v22) )
        {
          v12 = *(_QWORD *)(a2 - 64);
          return sub_F162A0((__int64)a1, a2, v12);
        }
      }
      v14 = *(_QWORD *)(v20 + 16);
      if ( !v14 )
        return sub_11B39B0((__int64)a1, a2);
      ++v17;
      if ( *(_QWORD *)(v14 + 8) || v17 == 10 )
        return sub_11B39B0((__int64)a1, a2);
      v16 = v20;
    }
  }
  return sub_11B39B0((__int64)a1, a2);
}
