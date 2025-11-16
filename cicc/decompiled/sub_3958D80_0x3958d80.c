// Function: sub_3958D80
// Address: 0x3958d80
//
void sub_3958D80()
{
  char v0; // al
  char *v1; // rdi

  v1 = aPrhodwRpfifyMc;
  if ( aPrhodwRpfifyMc != &aPrhodwRpfifyMc[27] )
  {
    v0 = 0;
    do
    {
      *v1++ ^= v0;
      v0 += 3;
    }
    while ( &aPrhodwRpfifyMc[27] != v1 );
  }
}
