// Function: sub_23C6910
// Address: 0x23c6910
//
void __fastcall sub_23C6910(_QWORD *a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = (unsigned __int64 *)a1[7];
  v2 = (unsigned __int64 *)a1[6];
  if ( v1 != v2 )
  {
    do
    {
      if ( (unsigned __int64 *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2);
      v2 += 4;
    }
    while ( v1 != v2 );
    v2 = (unsigned __int64 *)a1[6];
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
}
