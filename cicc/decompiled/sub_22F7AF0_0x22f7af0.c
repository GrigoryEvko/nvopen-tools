// Function: sub_22F7AF0
// Address: 0x22f7af0
//
__int64 __fastcall sub_22F7AF0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, unsigned int *a5, _DWORD *a6, int a7)
{
  _DWORD v8[4]; // [rsp+8h] [rbp-30h] BYREF
  __int64 (__fastcall *v9)(_QWORD *, _DWORD *, int); // [rsp+18h] [rbp-20h]
  bool (__fastcall *v10)(_DWORD *, __int64); // [rsp+20h] [rbp-18h]

  v8[0] = a7;
  v10 = sub_22F4FE0;
  v9 = sub_22F4ED0;
  sub_22F7880(a1, a2, a3, a4, a5, a6, (__int64)v8);
  if ( v9 )
    v9(v8, v8, 3);
  return a1;
}
