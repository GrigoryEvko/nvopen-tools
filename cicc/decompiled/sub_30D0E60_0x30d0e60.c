// Function: sub_30D0E60
// Address: 0x30d0e60
//
__int64 __fastcall sub_30D0E60(__int64 **a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v4; // [rsp+Fh] [rbp-51h]
  _BYTE v5[24]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v6; // [rsp+28h] [rbp-38h]
  unsigned int v7; // [rsp+30h] [rbp-30h]
  unsigned __int64 v8; // [rsp+38h] [rbp-28h]
  unsigned int v9; // [rsp+40h] [rbp-20h]
  char v10; // [rsp+48h] [rbp-18h]
  unsigned __int8 v11; // [rsp+50h] [rbp-10h]

  sub_30D08B0((__int64)v5, a2, **a1, (__int64)(*a1 + 1));
  result = v11;
  if ( v11 )
  {
    v11 = 0;
    if ( v10 )
    {
      v10 = 0;
      if ( v9 > 0x40 && v8 )
      {
        v3 = result;
        j_j___libc_free_0_0(v8);
        result = v3;
      }
      if ( v7 > 0x40 )
      {
        if ( v6 )
        {
          v4 = result;
          j_j___libc_free_0_0(v6);
          return v4;
        }
      }
    }
  }
  return result;
}
