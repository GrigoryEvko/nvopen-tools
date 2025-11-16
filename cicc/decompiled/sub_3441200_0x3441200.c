// Function: sub_3441200
// Address: 0x3441200
//
__int64 __fastcall sub_3441200(__int64 a1, __int64 a2, int a3)
{
  char v3; // cl
  unsigned __int8 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // r10
  unsigned int v7; // r13d
  _DWORD v9[4]; // [rsp+0h] [rbp-60h] BYREF
  __int64 (__fastcall *v10)(_QWORD *, _DWORD *, int); // [rsp+10h] [rbp-50h]
  bool (__fastcall *v11)(unsigned int *, __int64); // [rsp+18h] [rbp-48h]
  _QWORD v12[2]; // [rsp+20h] [rbp-40h] BYREF
  void (__fastcall *v13)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-30h]
  bool (__fastcall *v14)(unsigned int *, __int64); // [rsp+38h] [rbp-28h]

  v9[0] = a3;
  v11 = sub_343F3C0;
  v10 = sub_343F670;
  v13 = 0;
  sub_343F670(v12, v9, 2);
  v14 = v11;
  v13 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v10;
  v7 = sub_33CA8D0(v6, v5, (__int64)v12, v3, v4);
  if ( v13 )
    v13(v12, v12, 3);
  if ( v10 )
    v10(v9, v9, 3);
  return v7;
}
