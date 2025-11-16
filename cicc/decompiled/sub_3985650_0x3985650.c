// Function: sub_3985650
// Address: 0x3985650
//
__int64 *__fastcall sub_3985650(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 *v5; // r12
  __int64 v7; // r15
  __int64 *v8; // r13
  __int64 v9; // r11
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = (a2 - (__int64)a1) >> 4;
  v5 = a1;
  v11[0] = a4;
  while ( v4 > 0 )
  {
    while ( 1 )
    {
      v7 = v4 >> 1;
      v8 = &v5[2 * (v4 >> 1)];
      if ( !(unsigned __int8)sub_3985080((__int64)v11, *v8, a3) )
        break;
      v5 = v8 + 2;
      v4 = v9 - v7 - 1;
      if ( v4 <= 0 )
        return v5;
    }
    v4 = v7;
  }
  return v5;
}
