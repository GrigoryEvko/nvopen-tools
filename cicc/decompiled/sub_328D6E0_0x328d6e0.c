// Function: sub_328D6E0
// Address: 0x328d6e0
//
bool __fastcall sub_328D6E0(__int64 a1, unsigned int a2, unsigned __int16 a3)
{
  bool result; // al

  if ( a3 == 1 )
    return a2 <= 0x1F3 && *(_BYTE *)(a2 + 500LL * a3 + a1 + 6414) == 0;
  result = 0;
  if ( a3 )
  {
    if ( *(_QWORD *)(a1 + 8LL * a3 + 112) )
      return a2 <= 0x1F3 && *(_BYTE *)(a2 + 500LL * a3 + a1 + 6414) == 0;
  }
  return result;
}
