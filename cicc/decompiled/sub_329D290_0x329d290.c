// Function: sub_329D290
// Address: 0x329d290
//
char __fastcall sub_329D290(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char result; // al
  __m128i v6; // xmm0
  __int64 v7; // rdx

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  result = sub_32657E0(a4 + 8, **(_QWORD **)(a1 + 40));
  if ( !result )
    return 0;
  v6 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
  v7 = *(_QWORD *)(a4 + 24);
  *(_QWORD *)v7 = v6.m128i_i64[0];
  *(_DWORD *)(v7 + 8) = v6.m128i_i32[2];
  if ( *(_BYTE *)(a4 + 36) )
    return (*(_DWORD *)(a4 + 32) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 32);
  return result;
}
