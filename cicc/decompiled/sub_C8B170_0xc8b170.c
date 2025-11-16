// Function: sub_C8B170
// Address: 0xc8b170
//
__int64 __fastcall sub_C8B170(__int64 a1)
{
  sub_C8B020(a1, 128);
  while ( *(_BYTE *)(a1 + 88) != 56 )
    sub_C8B020(a1, 0);
  sub_C8B020(a1, 0);
  sub_C8B020(a1, 0);
  sub_C8B020(a1, 0);
  sub_C8B020(a1, *(_DWORD *)(a1 + 84) >> 29);
  sub_C8B020(a1, *(_DWORD *)(a1 + 84) >> 21);
  sub_C8B020(a1, *(_DWORD *)(a1 + 84) >> 13);
  sub_C8B020(a1, *(_DWORD *)(a1 + 84) >> 5);
  return sub_C8B020(a1, 8 * *(_BYTE *)(a1 + 84));
}
