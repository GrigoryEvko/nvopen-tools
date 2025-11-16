// Function: sub_2165A20
// Address: 0x2165a20
//
__int64 __fastcall sub_2165A20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v4)(); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v5)(); // [rsp+18h] [rbp-18h]

  v5 = sub_2165CB0;
  v4 = sub_2165BA0;
  v3[0] = a1;
  sub_394BD90(a2, 0, v3);
  result = (__int64)v4;
  if ( v4 )
    return ((__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))v4)(v3, v3, 3);
  return result;
}
