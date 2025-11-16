// Function: sub_27677F0
// Address: 0x27677f0
//
void __fastcall sub_27677F0(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // rdi

  v2 = a1[1];
  v3 = *a1;
  if ( v2 != *a1 )
  {
    do
    {
      v4 = (unsigned __int64 *)v3;
      v3 += 80LL;
      sub_2767770(v4);
    }
    while ( v2 != v3 );
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3);
}
