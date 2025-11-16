// Function: sub_291C360
// Address: 0x291c360
//
__int64 __fastcall sub_291C360(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-78h]
  unsigned __int64 v7; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-68h]
  __int64 v9[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v10; // [rsp+40h] [rbp-40h]

  v4 = a1[14] - a1[5];
  v10 = 257;
  v8 = sub_AE43F0(*a1, a3);
  if ( v8 > 0x40 )
    sub_C43690((__int64)&v7, v4, 0);
  else
    v7 = v4;
  result = sub_291C070(a2, a1[4], (__int64)&v7, a3, v9);
  if ( v8 > 0x40 )
  {
    if ( v7 )
    {
      v6 = result;
      j_j___libc_free_0_0(v7);
      return v6;
    }
  }
  return result;
}
