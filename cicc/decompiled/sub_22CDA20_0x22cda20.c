// Function: sub_22CDA20
// Address: 0x22cda20
//
__int64 __fastcall sub_22CDA20(unsigned __int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v4; // r15d
  _BYTE v6[8]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v7; // [rsp+8h] [rbp-58h]
  unsigned int v8; // [rsp+10h] [rbp-50h]
  unsigned __int64 v9; // [rsp+18h] [rbp-48h]
  unsigned int v10; // [rsp+20h] [rbp-40h]
  unsigned __int8 v11; // [rsp+28h] [rbp-38h]

  sub_22CD7F0((__int64)v6, a1, a2, a3);
  v4 = v11;
  if ( v11 )
  {
    sub_22C5240(a1, (__int64)a2, a3, (__int64)v6);
    if ( v11 )
    {
      v11 = 0;
      if ( (unsigned int)v6[0] - 4 <= 1 )
      {
        if ( v10 > 0x40 && v9 )
          j_j___libc_free_0_0(v9);
        if ( v8 > 0x40 && v7 )
          j_j___libc_free_0_0(v7);
      }
    }
  }
  return v4;
}
