// Function: sub_195E3D0
// Address: 0x195e3d0
//
__int64 (__fastcall **__fastcall sub_195E3D0(_QWORD *a1))()
{
  __int64 v1; // rsi
  __int64 (__fastcall **result)(); // rax

  v1 = a1[9];
  result = off_49F3B68;
  *a1 = off_49F3B68;
  if ( v1 )
    return (__int64 (__fastcall **)())sub_161E7C0((__int64)(a1 + 9), v1);
  return result;
}
