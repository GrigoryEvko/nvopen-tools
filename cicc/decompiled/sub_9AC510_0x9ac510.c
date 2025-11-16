// Function: sub_9AC510
// Address: 0x9ac510
//
__int64 __fastcall sub_9AC510(__int64 a1, __m128i *a2, unsigned int a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-28h]
  __int64 v8; // [rsp+10h] [rbp-20h]
  unsigned int v9; // [rsp+18h] [rbp-18h]

  sub_9AC330((__int64)&v6, a1, a3, a2);
  v4 = 1LL << ((unsigned __int8)v9 - 1);
  if ( v9 > 0x40 )
  {
    LOBYTE(v3) = (*(_QWORD *)(v8 + 8LL * ((v9 - 1) >> 6)) & v4) != 0;
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  else
  {
    LOBYTE(v3) = (v8 & v4) != 0;
  }
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return v3;
}
