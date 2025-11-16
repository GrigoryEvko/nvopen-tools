// Function: sub_2534DD0
// Address: 0x2534dd0
//
__int64 __fastcall sub_2534DD0(__int64 a1)
{
  char v1; // al
  bool v2; // zf

  v1 = *(_BYTE *)(a1 + 96);
  v2 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 97) = v1;
  if ( v2 )
    *(_BYTE *)(a1 + 112) = 1;
  return 0;
}
