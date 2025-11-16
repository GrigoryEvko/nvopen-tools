// Function: sub_AAFB30
// Address: 0xaafb30
//
__int64 __fastcall sub_AAFB30(__int64 a1, int *a2, __int64 a3)
{
  __int64 *v3; // r12
  unsigned int v4; // ebx
  __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-18h]

  v3 = &v6;
  v7 = 1;
  v6 = 0;
  sub_AAF830(a1, a2, a3, &v6);
  v4 = v7;
  if ( v7 <= 0x40 )
  {
    LOBYTE(v3) = v6 == 0;
    return (unsigned int)v3;
  }
  else
  {
    LOBYTE(v3) = v4 == (unsigned int)sub_C444A0(&v6);
    if ( v6 )
      j_j___libc_free_0_0(v6);
    return (unsigned int)v3;
  }
}
