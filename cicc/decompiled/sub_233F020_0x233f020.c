// Function: sub_233F020
// Address: 0x233f020
//
void __fastcall sub_233F020(unsigned __int64 **a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = a1[1];
  v2 = *a1;
  if ( v1 != *a1 )
  {
    do
    {
      if ( (unsigned __int64 *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2);
      v2 += 4;
    }
    while ( v1 != v2 );
    v2 = *a1;
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
}
