// Function: sub_8D1B30
// Address: 0x8d1b30
//
__int64 __fastcall sub_8D1B30(__int64 a1)
{
  _QWORD *v2; // rax

  if ( *(_BYTE *)(a1 + 140) != 8 || (*(_BYTE *)(a1 + 169) & 0x10) == 0 )
    return 0;
  v2 = sub_72D900(a1);
  sub_7D9F20((__int64)v2);
  return 0;
}
