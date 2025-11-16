// Function: sub_6DEBF0
// Address: 0x6debf0
//
void __fastcall sub_6DEBF0(__int64 a1, __int64 a2)
{
  if ( (*(_BYTE *)(a1 - 8) & 1) == 0 || (*(_BYTE *)(a1 + 26) & 8) != 0 )
    *(_DWORD *)(a2 + 76) = 1;
}
