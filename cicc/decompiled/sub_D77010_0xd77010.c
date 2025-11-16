// Function: sub_D77010
// Address: 0xd77010
//
__int64 __fastcall sub_D77010(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  __int64 result; // rax
  __int64 v5[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF

  v2 = *a1;
  sub_B2F930(v5, a2);
  v3 = sub_B2F650(v5[0], v5[1]);
  if ( (__int64 *)v5[0] != &v6 )
    j_j___libc_free_0(v5[0], v6 + 1);
  result = sub_BAEEF0(v2, v3);
  *(_BYTE *)(result + 12) |= 0x80u;
  return result;
}
