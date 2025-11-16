// Function: sub_A04970
// Address: 0xa04970
//
__int64 *__fastcall sub_A04970(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r13

  v2 = *a2;
  *a2 = 0;
  v3 = *a1;
  *a1 = v2;
  if ( v3 )
  {
    sub_A043F0(v3);
    j_j___libc_free_0(v3, 1144);
  }
  return a1;
}
