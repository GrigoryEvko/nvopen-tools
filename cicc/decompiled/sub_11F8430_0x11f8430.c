// Function: sub_11F8430
// Address: 0x11f8430
//
__int64 *__fastcall sub_11F8430(
        __int64 *a1,
        __int64 a2,
        _QWORD *a3,
        void (__fastcall *a4)(__int64, _QWORD *, __int64),
        __int64 a5,
        __int64 a6,
        void (__fastcall *a7)(__int64, __int64 *, unsigned int *, __int64),
        __int64 a8)
{
  _QWORD *v9; // rax
  __int64 v10; // r15
  _QWORD *v11; // [rsp+8h] [rbp-A8h] BYREF
  _QWORD v12[20]; // [rsp+10h] [rbp-A0h] BYREF

  if ( a4 )
  {
    sub_11F3DF0(a1, a2, a4, a5, a7, a8);
  }
  else
  {
    memset(v12, 0, 0x78u);
    if ( a3 )
    {
      v9 = (_QWORD *)sub_11F3890(a3);
    }
    else
    {
      v10 = sub_AA4B30(a2);
      if ( LOBYTE(v12[14]) )
      {
        LOBYTE(v12[14]) = 0;
        sub_A55520(v12, a2);
      }
      sub_A558A0((__int64)v12, v10, 1);
      LOBYTE(v12[14]) = 1;
      v9 = v12;
    }
    v11 = v9;
    sub_11F3DF0(a1, a2, (void (__fastcall *)(__int64, _QWORD *, __int64))sub_11F3530, (__int64)&v11, a7, a8);
    if ( LOBYTE(v12[14]) )
    {
      LOBYTE(v12[14]) = 0;
      sub_A55520(v12, a2);
    }
  }
  return a1;
}
