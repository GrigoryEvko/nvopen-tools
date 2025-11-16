// Function: sub_29E11F0
// Address: 0x29e11f0
//
_QWORD *__fastcall sub_29E11F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  unsigned int v6; // ebx
  int v7; // eax
  _QWORD *v8; // rax
  __int64 v10; // [rsp+0h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_B118D0(v12, a2, a3, a4, a5);
  v10 = sub_B10CD0((__int64)v12);
  v11 = sub_B10D00(a2);
  v6 = sub_B10CF0(a2);
  v7 = sub_B10CE0(a2);
  v8 = sub_B01860(a4, v7, v6, v11, v10, 0, 0, 1);
  sub_B10CB0(a1, (__int64)v8);
  if ( v12[0] )
    sub_B91220((__int64)v12, v12[0]);
  return a1;
}
