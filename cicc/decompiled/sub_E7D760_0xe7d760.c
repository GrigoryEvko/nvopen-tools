// Function: sub_E7D760
// Address: 0xe7d760
//
__int64 __fastcall sub_E7D760(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d

  v6 = a4;
  sub_E5CB20(a1[37], a2, a3, a4, a5, a6);
  sub_EA1710(a2, 0);
  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD))(*a1 + 480))(a1, a2, a3, v6);
}
