// Function: sub_15A1020
// Address: 0x15a1020
//
__int64 __fastcall sub_15A1020(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al

  v4 = a1[16];
  switch ( v4 )
  {
    case 10:
      return sub_15A06D0(**(__int64 ****)(*(_QWORD *)a1 + 16LL), a2, a3, a4);
    case 12:
      return sub_15A0FF0((__int64)a1);
    case 8:
      return sub_1594B20((__int64)a1);
  }
  return 0;
}
