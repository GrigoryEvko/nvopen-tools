// Function: sub_13A50C0
// Address: 0x13a50c0
//
__int64 __fastcall sub_13A50C0(unsigned __int64 **a1)
{
  unsigned __int64 *v1; // r12
  __int64 result; // rax

  v1 = *a1;
  if ( ((unsigned __int8)*a1 & 1) == 0 )
  {
    if ( v1 )
    {
      _libc_free(*v1);
      return j_j___libc_free_0(v1, 24);
    }
  }
  return result;
}
