// Function: sub_16C2450
// Address: 0x16c2450
//
_QWORD *__fastcall sub_16C2450(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  _QWORD v9[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v10; // [rsp+10h] [rbp-50h] BYREF
  __int16 v11; // [rsp+20h] [rbp-40h]

  v11 = 261;
  v9[0] = a4;
  v9[1] = a5;
  v10 = v9;
  v6 = (_QWORD *)sub_16C2200(24, (__int64)&v10);
  v7 = v6;
  if ( v6 )
  {
    *v6 = off_4985050;
    sub_16C2440((__int64)v6, a2, a2 + a3);
  }
  *a1 = v7;
  return a1;
}
