// Function: sub_A6E4A0
// Address: 0xa6e4a0
//
__int64 __fastcall sub_A6E4A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  _QWORD v5[3]; // [rsp+0h] [rbp-60h] BYREF
  int v6; // [rsp+18h] [rbp-48h] BYREF
  _QWORD *v7; // [rsp+20h] [rbp-40h]
  int *v8; // [rsp+28h] [rbp-38h]
  int *v9; // [rsp+30h] [rbp-30h]
  __int64 v10; // [rsp+38h] [rbp-28h]

  result = sub_B2DC10();
  if ( (_BYTE)result )
  {
    v5[0] = 0;
    v6 = 0;
    v7 = 0;
    v8 = &v6;
    v9 = &v6;
    v10 = 0;
    v5[1] = 224;
    if ( (unsigned __int8)sub_B2D610(a2, 70) )
    {
      sub_B2D4F0(a1, v5);
      v4 = 70;
      sub_B2CD30(a1, 70);
    }
    else if ( !(unsigned __int8)sub_B2D610(a2, 71) || (unsigned __int8)sub_B2D610(a1, 70) )
    {
      v4 = 69;
      if ( (unsigned __int8)sub_B2D610(a2, 69) )
      {
        v4 = 70;
        if ( !(unsigned __int8)sub_B2D610(a1, 70) )
        {
          v4 = 71;
          if ( !(unsigned __int8)sub_B2D610(a1, 71) )
          {
            v4 = 69;
            sub_B2CD30(a1, 69);
          }
        }
      }
    }
    else
    {
      sub_B2D4F0(a1, v5);
      v4 = 71;
      sub_B2CD30(a1, 71);
    }
    return sub_A6E200(v7, v4);
  }
  return result;
}
