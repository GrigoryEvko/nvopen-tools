// Function: sub_26C3E80
// Address: 0x26c3e80
//
bool __fastcall sub_26C3E80(__int64 a1, __int64 a2)
{
  bool result; // al
  bool v3; // [rsp+Fh] [rbp-51h]
  bool v4; // [rsp+Fh] [rbp-51h]
  int v5; // [rsp+10h] [rbp-50h] BYREF
  int v6; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v7; // [rsp+28h] [rbp-38h]
  unsigned int v8; // [rsp+30h] [rbp-30h]
  unsigned __int64 v9; // [rsp+38h] [rbp-28h]
  unsigned int v10; // [rsp+40h] [rbp-20h]
  char v11; // [rsp+48h] [rbp-18h]
  bool v12; // [rsp+50h] [rbp-10h]

  sub_26C3D10((__int64)&v5, a1, a2);
  result = v12;
  if ( v12 )
  {
    v12 = 0;
    result = v6 > v5;
    if ( v11 )
    {
      v11 = 0;
      if ( v10 > 0x40 && v9 )
      {
        v3 = v6 > v5;
        j_j___libc_free_0_0(v9);
        result = v3;
      }
      if ( v8 > 0x40 )
      {
        if ( v7 )
        {
          v4 = result;
          j_j___libc_free_0_0(v7);
          return v4;
        }
      }
    }
  }
  return result;
}
