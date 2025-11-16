// Function: sub_FFF170
// Address: 0xfff170
//
__int64 __fastcall sub_FFF170(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  unsigned int v3; // ebx
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-18h]

  v2 = &v5;
  sub_C4B490((__int64)&v5, a2, *a1);
  v3 = v6;
  if ( v6 <= 0x40 )
  {
    LOBYTE(v2) = v5 == 0;
    return (unsigned int)v2;
  }
  else
  {
    LOBYTE(v2) = v3 == (unsigned int)sub_C444A0((__int64)&v5);
    if ( v5 )
      j_j___libc_free_0_0(v5);
    return (unsigned int)v2;
  }
}
