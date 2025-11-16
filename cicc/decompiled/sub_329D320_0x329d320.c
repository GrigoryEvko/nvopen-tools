// Function: sub_329D320
// Address: 0x329d320
//
bool __fastcall sub_329D320(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v10 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v10 - 2);
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(a4 + 16);
  v9 = *(_QWORD *)(v7 + 40);
  if ( v8 )
  {
    if ( v9 != v8 || *(_DWORD *)(v7 + 48) != *(_DWORD *)(a4 + 24) )
      return 0;
  }
  else if ( !v9 )
  {
    return 0;
  }
  result = 1;
  if ( *(_BYTE *)(a4 + 36) )
    return (*(_DWORD *)(a4 + 32) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 32);
  return result;
}
