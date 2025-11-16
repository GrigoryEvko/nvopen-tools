// Function: sub_329EC40
// Address: 0x329ec40
//
char __fastcall sub_329EC40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char result; // al
  __int64 v6; // rax
  __m128i v7; // [rsp-28h] [rbp-28h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(_QWORD *)(a4 + 8);
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)v6 = v7.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = v7.m128i_i32[2];
  result = sub_32657E0(a4 + 16, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL));
  if ( !result )
    return 0;
  if ( *(_BYTE *)(a4 + 36) )
    return (*(_DWORD *)(a4 + 32) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 32);
  return result;
}
