// Function: sub_33CEB60
// Address: 0x33ceb60
//
__int64 sub_33CEB60()
{
  char v0; // cl
  unsigned __int8 v1; // r8
  __int64 v2; // r9
  _QWORD *v3; // r10
  unsigned int v4; // r13d
  _BYTE v6[16]; // [rsp+0h] [rbp-60h] BYREF
  __int64 (__fastcall *v7)(_QWORD *, __int64, int); // [rsp+10h] [rbp-50h]
  bool (__fastcall *v8)(__int64, __int64); // [rsp+18h] [rbp-48h]
  _QWORD v9[2]; // [rsp+20h] [rbp-40h] BYREF
  void (__fastcall *v10)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-30h]
  bool (__fastcall *v11)(__int64, __int64); // [rsp+38h] [rbp-28h]

  v8 = sub_33C8610;
  v7 = sub_33C7E20;
  v10 = 0;
  sub_33C7E20(v9, (__int64)v6, 2);
  v11 = v8;
  v10 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v7;
  v4 = sub_33CAAD0(v3, v2, (__int64)v9, v0, v1);
  if ( v10 )
    v10(v9, v9, 3);
  if ( v7 )
    v7(v6, (__int64)v6, 3);
  return v4;
}
