// Function: sub_1456170
// Address: 0x1456170
//
bool __fastcall sub_1456170(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = 0;
  if ( !*(_WORD *)(a1 + 24) )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) == *(_QWORD *)(v2 + 24);
    else
      return v3 == (unsigned int)sub_16A58F0(v2 + 24);
  }
  return result;
}
