// Function: sub_30FB2C0
// Address: 0x30fb2c0
//
void __fastcall sub_30FB2C0(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rdi

  v2 = a1[5];
  if ( v2 )
    j_j___libc_free_0(v2);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    j_j___libc_free_0(*a1);
}
