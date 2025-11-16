// Function: sub_21892C0
// Address: 0x21892c0
//
void sub_21892C0()
{
  char v0; // al
  _BYTE *v1; // rdi

  v1 = &unk_4CD4B20;
  if ( &unk_4CD4B20 != &unk_4CD4D97 )
  {
    v0 = 0;
    do
    {
      *v1++ ^= v0;
      v0 += 3;
    }
    while ( &unk_4CD4D97 != (_UNKNOWN *)v1 );
  }
}
