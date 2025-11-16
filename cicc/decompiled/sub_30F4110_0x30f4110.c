// Function: sub_30F4110
// Address: 0x30f4110
//
__int64 __fastcall sub_30F4110(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 result; // rax
  __int64 v7; // rdx

  v4 = *(int *)(a1 + 32);
  if ( !*(_DWORD *)(a1 + 32) )
    return 0xFFFFFFFFLL;
  v5 = *(_QWORD *)(a1 + 24);
  result = 0;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *(_WORD *)(v7 + 24) == 8 && a2 == *(_QWORD *)(v7 + 48) )
      break;
    if ( v4 == ++result )
      return 0xFFFFFFFFLL;
  }
  return result;
}
