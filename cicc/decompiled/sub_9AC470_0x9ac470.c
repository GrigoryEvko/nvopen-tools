// Function: sub_9AC470
// Address: 0x9ac470
//
__int64 __fastcall sub_9AC470(__int64 a1, __m128i *a2, unsigned int a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]
  unsigned int v10; // [rsp+18h] [rbp-18h]

  sub_9AC330((__int64)&v7, a1, a3, a2);
  v4 = 1LL << ((unsigned __int8)v8 - 1);
  if ( v8 > 0x40 )
  {
    LOBYTE(v3) = (*(_QWORD *)(v7 + 8LL * ((v8 - 1) >> 6)) & v4) != 0;
    if ( v10 <= 0x40 )
      goto LABEL_8;
    v5 = v9;
    if ( !v9 )
      goto LABEL_8;
  }
  else
  {
    LOBYTE(v3) = (v7 & v4) != 0;
    if ( v10 <= 0x40 )
      return v3;
    v5 = v9;
    if ( !v9 )
      return v3;
  }
  j_j___libc_free_0_0(v5);
  if ( v8 <= 0x40 )
    return v3;
LABEL_8:
  if ( !v7 )
    return v3;
  j_j___libc_free_0_0(v7);
  return v3;
}
