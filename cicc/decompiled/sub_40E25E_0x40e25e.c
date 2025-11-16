// Function: sub_40E25E
// Address: 0x40e25e
//
__int64 __fastcall sub_40E25E(__int64 (__fastcall *a1)(__int64, _BYTE *), __int64 a2, char *a3, __int64 a4)
{
  __int64 (__fastcall *v4)(__int64, _BYTE *); // rbx
  _BYTE v6[4128]; // [rsp+0h] [rbp-1020h] BYREF

  v4 = a1;
  if ( !a1 )
  {
    v4 = (__int64 (__fastcall *)(__int64, _BYTE *))unk_505F9C0;
    if ( !unk_505F9C0 )
      v4 = (__int64 (__fastcall *)(__int64, _BYTE *))&sub_130AA00;
  }
  sub_40D5CA((__int64)v6, 0x1000u, a3, a4);
  return v4(a2, v6);
}
