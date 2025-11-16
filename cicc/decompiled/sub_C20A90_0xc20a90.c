// Function: sub_C20A90
// Address: 0xc20a90
//
__int64 __fastcall sub_C20A90(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // [rsp+0h] [rbp-70h] BYREF
  __int64 v3; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v5[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v6[32]; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int8 v7; // [rsp+50h] [rbp-20h]
  _BYTE *v8; // [rsp+60h] [rbp-10h]
  __int64 v9; // [rsp+68h] [rbp-8h]

  sub_C7C840(v6, a1, 1, 35);
  result = v7;
  if ( v7 )
  {
    result = 0;
    if ( *v8 != 32 )
    {
      v4[0] = 0;
      v4[1] = 0;
      v5[0] = (__int64)v8;
      v5[1] = v9;
      if ( *v8 != 32 )
        return sub_C1F5F0(v5, v4, &v2, &v3);
    }
  }
  return result;
}
