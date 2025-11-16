// Function: sub_1456260
// Address: 0x1456260
//
bool __fastcall sub_1456260(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  __int64 v3; // rdi
  unsigned int v4; // esi
  __int64 v5; // rax

  result = 0;
  if ( *(_WORD *)(a1 + 24) == 5 )
  {
    v2 = **(_QWORD **)(a1 + 32);
    if ( !*(_WORD *)(v2 + 24) )
    {
      v3 = *(_QWORD *)(v2 + 32);
      v4 = *(_DWORD *)(v3 + 32);
      v5 = *(_QWORD *)(v3 + 24);
      if ( v4 > 0x40 )
        v5 = *(_QWORD *)(v5 + 8LL * ((v4 - 1) >> 6));
      return (v5 & (1LL << ((unsigned __int8)v4 - 1))) != 0;
    }
  }
  return result;
}
