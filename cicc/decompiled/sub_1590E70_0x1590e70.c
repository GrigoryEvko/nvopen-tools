// Function: sub_1590E70
// Address: 0x1590e70
//
__int64 __fastcall sub_1590E70(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v4; // [rsp+8h] [rbp-38h]
  __int64 v5; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-28h]

  if ( sub_158A0B0(a2) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( !sub_158A120(a2) )
  {
    v6 = *(_DWORD *)(a2 + 8);
    if ( v6 > 0x40 )
      sub_16A4FD0(&v5, a2);
    else
      v5 = *(_QWORD *)a2;
    v4 = *(_DWORD *)(a2 + 24);
    if ( v4 > 0x40 )
      sub_16A4FD0(&v3, a2 + 16);
    else
      v3 = *(_QWORD *)(a2 + 16);
    sub_15898E0(a1, (__int64)&v3, &v5);
    if ( v4 > 0x40 && v3 )
      j_j___libc_free_0_0(v3);
    if ( v6 > 0x40 )
    {
      if ( v5 )
        j_j___libc_free_0_0(v5);
    }
    return a1;
  }
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  return a1;
}
