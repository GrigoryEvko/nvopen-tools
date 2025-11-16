// Function: sub_33C9430
// Address: 0x33c9430
//
__int64 __fastcall sub_33C9430(unsigned int *a1, __int64 a2)
{
  unsigned __int64 *v2; // r12
  unsigned __int64 v4; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-18h]

  v2 = &v4;
  sub_C44AB0((__int64)&v4, *(_QWORD *)(*(_QWORD *)a2 + 96LL) + 24LL, *a1);
  if ( v5 <= 0x40 )
  {
    LODWORD(v2) = 0;
    if ( v4 )
      LOBYTE(v2) = (v4 & (v4 - 1)) == 0;
    return (unsigned int)v2;
  }
  LOBYTE(v2) = (unsigned int)sub_C44630((__int64)&v4) == 1;
  if ( !v4 )
    return (unsigned int)v2;
  j_j___libc_free_0_0(v4);
  return (unsigned int)v2;
}
