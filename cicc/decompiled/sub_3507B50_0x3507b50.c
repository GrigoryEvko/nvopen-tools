// Function: sub_3507B50
// Address: 0x3507b50
//
bool __fastcall sub_3507B50(__int64 a1)
{
  bool result; // al

  result = 1;
  if ( *(_BYTE *)a1 != 12 )
  {
    result = 0;
    if ( !*(_BYTE *)a1 && (*(_BYTE *)(a1 + 4) & 8) == 0 )
      return (unsigned int)(*(_DWORD *)(a1 + 8) - 1) <= 0x3FFFFFFE;
  }
  return result;
}
