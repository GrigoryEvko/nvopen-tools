// Function: sub_11552A0
// Address: 0x11552a0
//
__int64 __fastcall sub_11552A0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *v3; // rax
  unsigned int v4; // ebx
  unsigned int v5; // r14d
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-28h]

  v2 = &v9;
  v3 = *a1;
  v8 = 1;
  v7 = 0;
  v10 = 1;
  v9 = 0;
  sub_C4C400(a2, *v3, (__int64)&v7, (__int64)&v9);
  v4 = v10;
  if ( v10 <= 0x40 )
    LOBYTE(v2) = v9 == 0;
  else
    LOBYTE(v2) = v4 == (unsigned int)sub_C444A0((__int64)&v9);
  if ( (_BYTE)v2 )
  {
    v5 = v8;
    LODWORD(v2) = 0;
    if ( v8 )
    {
      if ( v8 <= 0x40 )
        LOBYTE(v2) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) == v7;
      else
        LOBYTE(v2) = v5 == (unsigned int)sub_C445E0((__int64)&v7);
      LODWORD(v2) = (unsigned int)v2 ^ 1;
    }
  }
  if ( v4 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return (unsigned int)v2;
}
