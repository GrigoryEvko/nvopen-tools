// Function: sub_23FD590
// Address: 0x23fd590
//
void __fastcall sub_23FD590(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(unsigned __int64 **)a1;
  v3 = (unsigned __int64 *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v3 )
  {
    do
    {
      v3 -= 10;
      v4 = v3[4];
      if ( (unsigned __int64 *)v4 != v3 + 6 )
        j_j___libc_free_0(v4);
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3);
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)a1;
  }
  if ( v3 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v3);
}
