// Function: sub_34E6930
// Address: 0x34e6930
//
void __fastcall sub_34E6930(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  void (__fastcall *v4)(__int64 *, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD); // rax
  __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  _BYTE *v6; // [rsp+10h] [rbp-20h]
  __int64 v7; // [rsp+18h] [rbp-18h]
  _BYTE v8[8]; // [rsp+20h] [rbp-10h] BYREF

  v6 = v8;
  v3 = *a3;
  v5 = 0;
  v4 = *(void (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(v3 + 368);
  v7 = 0;
  v4(a3, a1, a2, 0, v8, 0, &v5, 0);
  if ( v6 != v8 )
    _libc_free((unsigned __int64)v6);
  if ( v5 )
    sub_B91220((__int64)&v5, v5);
}
