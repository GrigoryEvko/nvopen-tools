// Function: sub_26AB870
// Address: 0x26ab870
//
void __fastcall sub_26AB870(unsigned __int64 a1)
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
  j_j___libc_free_0(a1);
}
