// Function: sub_1D8DC30
// Address: 0x1d8dc30
//
__int64 __fastcall sub_1D8DC30(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  _BYTE v4[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v5; // [rsp+8h] [rbp-38h]
  int v6; // [rsp+10h] [rbp-30h]

  v2 = 0;
  sub_20C9140(v4);
  if ( v6 )
  {
    v2 = 1;
    sub_1D8D810((unsigned __int64 *)(a2 + 320), (__int64)v4);
  }
  j___libc_free_0(v5);
  return v2;
}
