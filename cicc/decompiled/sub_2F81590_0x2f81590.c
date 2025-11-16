// Function: sub_2F81590
// Address: 0x2f81590
//
_BYTE *__fastcall sub_2F81590(__int64 **a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  _BYTE *result; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // [rsp+8h] [rbp-68h]
  unsigned __int64 v11; // [rsp+8h] [rbp-68h]
  unsigned __int64 v12[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v13; // [rsp+30h] [rbp-40h]

  result = sub_BA8CB0((__int64)a1, a3, a4);
  if ( !result )
  {
    v12[0] = a3;
    v13 = 261;
    v12[1] = a4;
    v9 = sub_B2C660(a2, 0, (__int64)v12, (__int64)a1);
    if ( a5 )
    {
      v10 = v9;
      sub_B2EC90(v9, a5);
      v9 = v10;
    }
    v12[0] = v9;
    v11 = v9;
    sub_2A41DC0(a1, v12, 1);
    return (_BYTE *)v11;
  }
  return result;
}
