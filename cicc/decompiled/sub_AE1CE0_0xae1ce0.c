// Function: sub_AE1CE0
// Address: 0xae1ce0
//
bool __fastcall sub_AE1CE0(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( *(_DWORD *)a1 == *(_DWORD *)a2 && *(_BYTE *)(a1 + 4) == *(_BYTE *)(a2 + 4) )
    return *(_BYTE *)(a1 + 5) == *(_BYTE *)(a2 + 5);
  return result;
}
