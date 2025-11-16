// Function: sub_8D9610
// Address: 0x8d9610
//
_BOOL8 __fastcall sub_8D9610(__int64 a1, _BYTE *a2)
{
  bool v2; // zf

  byte_4F60594 = 0;
  sub_8D9600(a1, (__int64 (__fastcall *)(__int64, unsigned int *))sub_8D0FB0, 0x41Bu);
  v2 = byte_4F60594 == 0;
  *a2 = byte_4F60594;
  return !v2;
}
