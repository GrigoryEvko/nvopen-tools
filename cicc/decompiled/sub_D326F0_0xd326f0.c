// Function: sub_D326F0
// Address: 0xd326f0
//
__int64 __fastcall sub_D326F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]
  char v9; // [rsp+10h] [rbp-20h]

  v3 = a1;
  v4 = a2;
  sub_DC06D0(&v7, a3, a2, a1);
  if ( !v9 )
    return 0;
  v5 = 1LL << ((unsigned __int8)v8 - 1);
  if ( v8 > 0x40 )
  {
    if ( (*(_QWORD *)(v7 + 8LL * ((v8 - 1) >> 6)) & v5) != 0 )
      v3 = a2;
    if ( v7 )
      j_j___libc_free_0_0(v7);
    return v3;
  }
  else if ( (v5 & v7) == 0 )
  {
    return a1;
  }
  return v4;
}
