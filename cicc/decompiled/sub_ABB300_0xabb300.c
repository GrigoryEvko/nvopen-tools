// Function: sub_ABB300
// Address: 0xabb300
//
__int64 __fastcall sub_ABB300(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v4; // [rsp+8h] [rbp-38h]
  __int64 v5; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-28h]

  if ( sub_AAF760(a2) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( !sub_AAF7D0(a2) )
  {
    v6 = *(_DWORD *)(a2 + 8);
    if ( v6 > 0x40 )
      sub_C43780(&v5, a2);
    else
      v5 = *(_QWORD *)a2;
    v4 = *(_DWORD *)(a2 + 24);
    if ( v4 > 0x40 )
      sub_C43780(&v3, a2 + 16);
    else
      v3 = *(_QWORD *)(a2 + 16);
    sub_AADC30(a1, (__int64)&v3, &v5);
    if ( v4 > 0x40 && v3 )
      j_j___libc_free_0_0(v3);
    if ( v6 > 0x40 )
    {
      if ( v5 )
        j_j___libc_free_0_0(v5);
    }
    return a1;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
  return a1;
}
