// Function: sub_B53070
// Address: 0xb53070
//
__int64 __fastcall sub_B53070(__int64 a1)
{
  *(_WORD *)(a1 + 2) = sub_B52F50(*(_WORD *)(a1 + 2) & 0x3F) | *(_WORD *)(a1 + 2) & 0xFFC0;
  return sub_BD28A0(a1 - 64, a1 - 32);
}
