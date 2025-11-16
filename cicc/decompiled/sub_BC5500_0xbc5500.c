// Function: sub_BC5500
// Address: 0xbc5500
//
void __fastcall sub_BC5500(unsigned int *a1, __int64 a2)
{
  unsigned int *v2; // r13
  unsigned int *v3; // r12

  v2 = *(unsigned int **)a1;
  v3 = (unsigned int *)(*(_QWORD *)a1 + 32LL * a1[2]);
  if ( *(unsigned int **)a1 != v3 )
  {
    do
    {
      v3 -= 8;
      if ( *(unsigned int **)v3 != v3 + 4 )
      {
        a2 = *((_QWORD *)v3 + 2) + 1LL;
        j_j___libc_free_0(*(_QWORD *)v3, a2);
      }
    }
    while ( v3 != v2 );
    v3 = *(unsigned int **)a1;
  }
  if ( v3 != a1 + 4 )
    _libc_free(v3, a2);
}
