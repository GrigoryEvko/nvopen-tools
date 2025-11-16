// Function: sub_222DCB0
// Address: 0x222dcb0
//
void __fastcall sub_222DCB0(__int64 a1, int a2)
{
  *(_DWORD *)(a1 + 32) |= a2;
  if ( (*(_DWORD *)(a1 + 28) & a2) != 0 )
    sub_22534D0(a1);
}
