// Function: sub_2A0CF40
// Address: 0x2a0cf40
//
__int64 __fastcall sub_2A0CF40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v5[2]; // [rsp+8h] [rbp-58h] BYREF
  __int64 v6; // [rsp+18h] [rbp-48h]
  char v7[56]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a2;
  v5[0] = 6;
  v5[1] = 0;
  v6 = a3;
  if ( a3 != 0 && a3 != -4096 && a3 != -8192 )
    sub_BD73F0((__int64)v5);
  sub_F621C0((__int64)v7, a1, &v4);
  result = v6;
  if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
    return sub_BD60C0(v5);
  return result;
}
