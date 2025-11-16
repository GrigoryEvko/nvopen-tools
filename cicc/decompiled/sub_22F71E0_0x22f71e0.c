// Function: sub_22F71E0
// Address: 0x22f71e0
//
__int64 __fastcall sub_22F71E0(__int64 a1, __int64 a2, char *a3, char *a4, int a5, int a6, char a7)
{
  __int64 result; // rax
  _DWORD v8[4]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v9)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v10)(_DWORD *, __int64, __int64, __int64, unsigned int); // [rsp+18h] [rbp-18h]

  v8[0] = a5;
  v8[1] = a6 & 0xFFFFFFFE;
  v10 = sub_22F4E30;
  v9 = sub_22F4F00;
  sub_22F5D60(a1, a2, a3, a4, (a6 & 1) == 0, a7, (__int64)v8, 0);
  result = (__int64)v9;
  if ( v9 )
    return v9(v8, v8, 3);
  return result;
}
