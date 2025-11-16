// Function: sub_1682740
// Address: 0x1682740
//
__int64 __fastcall sub_1682740(__int64 *a1, char *a2)
{
  __int64 v2; // r12

  if ( !a1 )
    return 4;
  v2 = *a1;
  if ( !*a1 )
    return 4;
  sub_16823C0(*a1, a2);
  j_j___libc_free_0(v2, 296);
  *a1 = 0;
  return 0;
}
