// Function: sub_1D13970
// Address: 0x1d13970
//
__int64 __fastcall sub_1D13970(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = *a1;
    if ( v2 )
      j_j___libc_free_0(v2, a1[2] - v2);
    return j_j___libc_free_0(a1, 24);
  }
  return result;
}
