// Function: sub_35DDE70
// Address: 0x35dde70
//
_BOOL8 __fastcall sub_35DDE70(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 48);
  return (v1 & 2) != 0 && *(_DWORD *)(a1 + 108) != 3 || (v1 & 4) != 0;
}
