// Function: sub_1477A90
// Address: 0x1477a90
//
__int64 __fastcall sub_1477A90(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *v3; // rax
  unsigned int v4; // ebx
  __int64 v5; // rax
  __int64 v7; // r13
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-28h]

  v2 = &v8;
  v3 = sub_1477920(a1, a2, 1u);
  sub_158ABC0(&v8, v3);
  v4 = v9;
  v5 = 1LL << ((unsigned __int8)v9 - 1);
  if ( v9 > 0x40 )
  {
    v7 = v8;
    if ( (*(_QWORD *)(v8 + 8LL * ((v9 - 1) >> 6)) & v5) != 0 )
      LODWORD(v2) = 1;
    else
      LOBYTE(v2) = v4 == (unsigned int)sub_16A57B0(&v8);
    if ( !v7 )
      return (unsigned int)v2;
    j_j___libc_free_0_0(v7);
    return (unsigned int)v2;
  }
  else
  {
    LODWORD(v2) = 1;
    if ( (v8 & v5) != 0 )
      return (unsigned int)v2;
    LOBYTE(v2) = v8 == 0;
    return (unsigned int)v2;
  }
}
