// Function: sub_266DDF0
// Address: 0x266ddf0
//
__int64 __fastcall sub_266DDF0(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = 0;
  if ( v1 )
    *(_BYTE *)(a1 + 112) = 1;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
