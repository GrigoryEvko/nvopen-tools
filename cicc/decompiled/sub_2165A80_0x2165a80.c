// Function: sub_2165A80
// Address: 0x2165a80
//
__int64 __fastcall sub_2165A80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d

  v3 = 0;
  if ( *(_BYTE *)(a2 + 8) == 11 && *(_BYTE *)(a3 + 8) == 11 && (unsigned int)sub_1643030(a2) == 64 )
    LOBYTE(v3) = (unsigned int)sub_1643030(a3) == 32;
  return v3;
}
