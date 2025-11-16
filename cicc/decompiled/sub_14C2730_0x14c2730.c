// Function: sub_14C2730
// Address: 0x14c2730
//
__int64 __fastcall sub_14C2730(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-28h]
  __int64 v12; // [rsp+10h] [rbp-20h]
  unsigned int v13; // [rsp+18h] [rbp-18h]

  sub_14C2530((__int64)&v10, a1, a2, a3, a4, a5, a6, 0);
  v7 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 > 0x40 )
  {
    LOBYTE(v6) = (*(_QWORD *)(v10 + 8LL * ((v11 - 1) >> 6)) & v7) != 0;
    if ( v13 <= 0x40 )
      goto LABEL_8;
    v8 = v12;
    if ( !v12 )
      goto LABEL_8;
  }
  else
  {
    LOBYTE(v6) = (v10 & v7) != 0;
    if ( v13 <= 0x40 )
      return v6;
    v8 = v12;
    if ( !v12 )
      return v6;
  }
  j_j___libc_free_0_0(v8);
  if ( v11 <= 0x40 )
    return v6;
LABEL_8:
  if ( !v10 )
    return v6;
  j_j___libc_free_0_0(v10);
  return v6;
}
