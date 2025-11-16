// Function: sub_36FCF50
// Address: 0x36fcf50
//
__int64 __fastcall sub_36FCF50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  __m128i v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  const char *v9; // r14
  size_t v10; // rdx
  size_t v11; // r12
  int v12; // eax
  int v13; // eax
  __int64 **v14; // rax
  __int64 *v15; // rax
  __m128i *v16; // rsi
  __m128i v18[3]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_36FCA50(a1, a2);
  v5.m128i_i64[0] = sub_36FCA50(a1, a3);
  ++*(_DWORD *)(v5.m128i_i64[0] + 80);
  v8 = v5.m128i_i64[0];
  if ( *(_BYTE *)(v4 + 88) || *(_BYTE *)(v5.m128i_i64[0] + 88) )
  {
    v5.m128i_i64[0] = *(unsigned int *)(v4 + 8);
    if ( v5.m128i_i64[0] + 1 > (unsigned __int64)*(unsigned int *)(v4 + 12) )
    {
      sub_C8D5F0(v4, (const void *)(v4 + 16), v5.m128i_i64[0] + 1, 8u, v6, v7);
      v5.m128i_i64[0] = *(unsigned int *)(v4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v4 + 8 * v5.m128i_i64[0]) = v8;
    ++*(_DWORD *)(v4 + 8);
    if ( !*(_BYTE *)(v4 + 88) )
    {
      v9 = sub_BD5D20(a2);
      v11 = v10;
      v12 = sub_C92610();
      v13 = sub_C92860((__int64 *)a1, v9, v11, v12);
      if ( v13 == -1 )
        v14 = (__int64 **)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      else
        v14 = (__int64 **)(*(_QWORD *)a1 + 8LL * v13);
      v15 = *v14;
      v16 = *(__m128i **)(a1 + 32);
      v5.m128i_i64[1] = *v15;
      v5.m128i_i64[0] = (__int64)(v15 + 2);
      v18[0] = v5;
      if ( v16 == *(__m128i **)(a1 + 40) )
      {
        v5.m128i_i64[0] = sub_C677B0((const __m128i **)(a1 + 24), v16, v18);
      }
      else
      {
        if ( v16 )
        {
          *v16 = _mm_loadu_si128(v18);
          v16 = *(__m128i **)(a1 + 32);
        }
        *(_QWORD *)(a1 + 32) = v16 + 1;
      }
    }
  }
  else
  {
    ++*(_DWORD *)(v5.m128i_i64[0] + 84);
  }
  return v5.m128i_i64[0];
}
