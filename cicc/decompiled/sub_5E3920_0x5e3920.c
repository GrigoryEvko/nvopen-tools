// Function: sub_5E3920
// Address: 0x5e3920
//
void __fastcall sub_5E3920(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx

  sub_5D2ED0(a1, a2, a3);
  if ( *(_BYTE *)(a1 + 140) == 2 && (**(_BYTE **)(a1 + 176) & 1) != 0 )
  {
    v4 = *(_QWORD *)(a1 + 168);
    if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
      v4 = *(_QWORD *)(v4 + 96);
    while ( v4 )
    {
      sub_5D2ED0(v4, a2, a3);
      v4 = *(_QWORD *)(v4 + 120);
    }
  }
}
