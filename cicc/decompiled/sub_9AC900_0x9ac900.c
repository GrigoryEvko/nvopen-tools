// Function: sub_9AC900
// Address: 0x9ac900
//
__int64 __fastcall sub_9AC900(__int64 *a1, __int64 *a2, __m128i *a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v7; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-58h]
  __int64 v9; // [rsp+10h] [rbp-50h]
  unsigned int v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-38h]
  __int64 v13; // [rsp+30h] [rbp-30h]
  unsigned int v14; // [rsp+38h] [rbp-28h]

  sub_9AC780((__int64)&v7, a1, 0, a3);
  sub_9AC780((__int64)&v11, a2, 0, a3);
  v4 = sub_ABD960(&v7, &v11);
  if ( v4 > 3 )
    BUG();
  v5 = v4;
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return v5;
}
