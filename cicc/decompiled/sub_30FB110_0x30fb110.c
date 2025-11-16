// Function: sub_30FB110
// Address: 0x30fb110
//
void __fastcall sub_30FB110(unsigned __int64 **a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi

  v2 = a1[1];
  v3 = *a1;
  if ( v2 != *a1 )
  {
    do
    {
      v4 = v3[5];
      if ( v4 )
        j_j___libc_free_0(v4);
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3);
      v3 += 10;
    }
    while ( v2 != v3 );
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
}
