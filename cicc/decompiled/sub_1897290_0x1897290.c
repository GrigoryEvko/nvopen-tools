// Function: sub_1897290
// Address: 0x1897290
//
__int64 __fastcall sub_1897290(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = *(unsigned __int64 **)a1;
  v2 = (unsigned __int64 *)(*(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v2 )
  {
    do
    {
      v2 -= 13;
      if ( (unsigned __int64 *)*v2 != v2 + 2 )
        _libc_free(*v2);
    }
    while ( v1 != v2 );
    v2 = *(unsigned __int64 **)a1;
  }
  if ( v2 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v2);
  return j_j___libc_free_0(a1, 432);
}
