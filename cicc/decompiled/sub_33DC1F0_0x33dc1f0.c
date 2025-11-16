// Function: sub_33DC1F0
// Address: 0x33dc1f0
//
__int64 __fastcall sub_33DC1F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // r12d
  unsigned int v6; // ebx
  unsigned __int64 v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-38h]
  unsigned __int64 v10; // [rsp+10h] [rbp-30h]
  unsigned int v11; // [rsp+18h] [rbp-28h]

  sub_33D4EF0((__int64)&v8, a1, a2, a3, a4, a5);
  v6 = v9;
  if ( v9 )
  {
    if ( v9 > 0x40 )
    {
      LOBYTE(v5) = v6 == (unsigned int)sub_C445E0((__int64)&v8);
      if ( v11 <= 0x40 )
      {
LABEL_7:
        if ( v8 )
          j_j___libc_free_0_0(v8);
        return v5;
      }
LABEL_4:
      if ( v10 )
      {
        j_j___libc_free_0_0(v10);
        v6 = v9;
      }
      if ( v6 <= 0x40 )
        return v5;
      goto LABEL_7;
    }
    LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == v8;
  }
  else
  {
    v5 = 1;
  }
  if ( v11 > 0x40 )
    goto LABEL_4;
  return v5;
}
