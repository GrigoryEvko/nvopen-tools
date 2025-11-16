// Function: sub_2767770
// Address: 0x2767770
//
void __fastcall sub_2767770(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi

  v2 = *a1;
  if ( v2 )
  {
    v3 = (unsigned __int64 *)a1[5];
    v4 = a1[9] + 8;
    if ( v4 > (unsigned __int64)v3 )
    {
      do
      {
        v5 = *v3++;
        j_j___libc_free_0(v5);
      }
      while ( v4 > (unsigned __int64)v3 );
      v2 = *a1;
    }
    j_j___libc_free_0(v2);
  }
}
