// Function: sub_310CFB0
// Address: 0x310cfb0
//
void __fastcall sub_310CFB0(unsigned __int64 *a1)
{
  unsigned __int64 *i; // rbx

  for ( i = a1 + 40; ; i -= 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
    if ( i == a1 )
      break;
  }
}
