// Function: sub_2F5B270
// Address: 0x2f5b270
//
__int64 __fastcall sub_2F5B270(__int64 a1, __int64 a2)
{
  void (__fastcall *v2)(_BYTE *, __int64, __int64); // rax
  unsigned int v3; // r12d
  _BYTE v5[16]; // [rsp+0h] [rbp-7260h] BYREF
  void (__fastcall *v6)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-7250h]
  __int64 v7; // [rsp+18h] [rbp-7248h]
  __int64 v8[14]; // [rsp+20h] [rbp-7240h] BYREF
  _QWORD v9[3642]; // [rsp+90h] [rbp-71D0h] BYREF

  sub_2F4FDC0(v8, a1);
  v2 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a1 + 216);
  v6 = 0;
  if ( v2 )
  {
    v2(v5, a1 + 200, 2);
    v7 = *(_QWORD *)(a1 + 224);
    v6 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a1 + 216);
  }
  sub_2F4F660((__int64)v9, v8, (__int64)v5);
  if ( v6 )
    v6(v5, v5, 3);
  v3 = sub_2F5A640(v9, a2);
  sub_2F4E350((__int64)v9);
  return v3;
}
