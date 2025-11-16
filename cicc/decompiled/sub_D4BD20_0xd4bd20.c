// Function: sub_D4BD20
// Address: 0xd4bd20
//
_QWORD *__fastcall sub_D4BD20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // rsi
  unsigned __int8 *v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_D4B950(&v8, a2, a3, a4, a5, a6);
  v6 = v8;
  *a1 = v8;
  if ( v6 )
    sub_B96E90((__int64)a1, (__int64)v6, 1);
  if ( v9[0] )
    sub_B91220((__int64)v9, v9[0]);
  if ( v8 )
    sub_B91220((__int64)&v8, (__int64)v8);
  return a1;
}
