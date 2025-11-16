// Function: sub_228BF40
// Address: 0x228bf40
//
void __fastcall sub_228BF40(unsigned __int64 **a1)
{
  unsigned __int64 *v1; // r12

  v1 = *a1;
  if ( ((unsigned __int8)*a1 & 1) == 0 && v1 )
  {
    if ( (unsigned __int64 *)*v1 != v1 + 2 )
      _libc_free(*v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
