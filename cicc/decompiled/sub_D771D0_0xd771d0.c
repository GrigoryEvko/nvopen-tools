// Function: sub_D771D0
// Address: 0xd771d0
//
__int64 __fastcall sub_D771D0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v4; // [rsp+10h] [rbp-20h] BYREF

  sub_B2F930(v3, a1);
  v1 = sub_B2F650(v3[0], v3[1]);
  if ( (__int64 *)v3[0] != &v4 )
    j_j___libc_free_0(v3[0], v4 + 1);
  return v1;
}
