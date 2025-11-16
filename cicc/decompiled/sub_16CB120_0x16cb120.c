// Function: sub_16CB120
// Address: 0x16cb120
//
__int64 __fastcall sub_16CB120(__int64 a1)
{
  sub_16CB090(a1, 128);
  while ( *(_BYTE *)(a1 + 88) != 56 )
    sub_16CB090(a1, 0);
  sub_16CB090(a1, 0);
  sub_16CB090(a1, 0);
  sub_16CB090(a1, 0);
  sub_16CB090(a1, *(_DWORD *)(a1 + 84) >> 29);
  sub_16CB090(a1, *(_DWORD *)(a1 + 84) >> 21);
  sub_16CB090(a1, *(_DWORD *)(a1 + 84) >> 13);
  sub_16CB090(a1, *(_DWORD *)(a1 + 84) >> 5);
  return sub_16CB090(a1, 8 * *(_BYTE *)(a1 + 84));
}
