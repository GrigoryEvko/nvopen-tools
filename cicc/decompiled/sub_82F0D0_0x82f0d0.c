// Function: sub_82F0D0
// Address: 0x82f0d0
//
void __fastcall sub_82F0D0(__int64 a1, _DWORD *a2)
{
  unsigned int v2; // r13d

  v2 = *(_DWORD *)(a1 + 80);
  if ( v2 )
  {
    if ( sub_6E53E0(5, v2, a2) )
      sub_684B30(v2, a2);
  }
  else if ( *(_BYTE *)(a1 + 14) )
  {
    if ( sub_6E53E0(5, 0x20Cu, a2) )
      sub_684B30(0x20Cu, a2);
  }
}
