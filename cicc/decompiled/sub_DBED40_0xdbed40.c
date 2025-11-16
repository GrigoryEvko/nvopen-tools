// Function: sub_DBED40
// Address: 0xdbed40
//
__int64 __fastcall sub_DBED40(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-18h]

  v3 = sub_DBB9F0(a1, a2, 1u, 0);
  sub_AB14C0((__int64)&v6, v3);
  v4 = 1LL << ((unsigned __int8)v7 - 1);
  if ( v7 <= 0x40 )
  {
    LOBYTE(v2) = (v6 & v4) == 0;
    return v2;
  }
  LOBYTE(v2) = (*(_QWORD *)(v6 + 8LL * ((v7 - 1) >> 6)) & v4) == 0;
  if ( !v6 )
    return v2;
  j_j___libc_free_0_0(v6);
  return v2;
}
