// Function: sub_1517310
// Address: 0x1517310
//
__int64 *__fastcall sub_1517310(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r13

  v2 = *a2;
  *a2 = 0;
  v3 = *a1;
  *a1 = v2;
  if ( v3 )
  {
    sub_1516EE0(v3);
    j_j___libc_free_0(v3, 1016);
  }
  return a1;
}
