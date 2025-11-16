// Function: sub_1CCAE30
// Address: 0x1ccae30
//
__int64 __fastcall sub_1CCAE30(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // [rsp+0h] [rbp-70h] BYREF
  __int16 v4; // [rsp+10h] [rbp-60h]
  __int64 *v5; // [rsp+20h] [rbp-50h] BYREF
  __int64 v6; // [rsp+30h] [rbp-40h] BYREF
  int v7; // [rsp+4Ch] [rbp-24h]

  v3 = a1 + 240;
  v4 = 260;
  sub_16E1010((__int64)&v5, (__int64)&v3);
  LOBYTE(v1) = v7 == 23;
  if ( v5 != &v6 )
    j_j___libc_free_0(v5, v6 + 1);
  return v1;
}
