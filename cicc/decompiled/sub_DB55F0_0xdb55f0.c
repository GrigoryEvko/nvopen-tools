// Function: sub_DB55F0
// Address: 0xdb55f0
//
__int64 __fastcall sub_DB55F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // ebx
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v6; // r12d
  unsigned int v9; // r12d
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-28h]

  v2 = sub_D95540(a2);
  v3 = sub_D97050(a1, v2);
  sub_DB4FC0((__int64)&v11, a1, a2);
  v4 = v12;
  if ( v12 <= 0x40 )
  {
    _RDX = v11;
    v9 = 64;
    __asm { tzcnt   rcx, rdx }
    if ( v11 )
      v9 = _RCX;
    if ( v3 <= v12 )
      v4 = v3;
    if ( v4 <= v9 )
      return v4;
    return v9;
  }
  else
  {
    v5 = sub_C44590((__int64)&v11);
    if ( v3 <= v5 )
      v5 = v3;
    v6 = v5;
    if ( v11 )
      j_j___libc_free_0_0(v11);
    return v6;
  }
}
