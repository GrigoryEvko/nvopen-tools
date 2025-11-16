// Function: sub_1B44350
// Address: 0x1b44350
//
bool __fastcall sub_1B44350(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  if ( !a1 )
    BUG();
  result = 0;
  if ( *(_BYTE *)(a1 - 8) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 48);
    if ( !*(_BYTE *)(v2 + 16) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      return (unsigned int)(*(_DWORD *)(v2 + 36) - 35) <= 3;
  }
  return result;
}
