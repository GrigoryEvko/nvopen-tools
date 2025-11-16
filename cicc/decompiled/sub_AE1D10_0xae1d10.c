// Function: sub_AE1D10
// Address: 0xae1d10
//
bool __fastcall sub_AE1D10(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( *(_DWORD *)a1 == *(_DWORD *)a2
    && *(_DWORD *)(a1 + 4) == *(_DWORD *)(a2 + 4)
    && *(_BYTE *)(a2 + 8) == *(_BYTE *)(a1 + 8)
    && *(_BYTE *)(a2 + 9) == *(_BYTE *)(a1 + 9)
    && *(_DWORD *)(a1 + 12) == *(_DWORD *)(a2 + 12) )
  {
    return *(_BYTE *)(a1 + 16) == *(_BYTE *)(a2 + 16);
  }
  return result;
}
