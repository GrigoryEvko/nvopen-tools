// Function: sub_21892A0
// Address: 0x21892a0
//
void sub_21892A0()
{
  char v0; // al
  _BYTE *v1; // rdi

  v1 = &unk_4CD4DA0;
  if ( &unk_4CD4DA0 != &unk_4CDFA70 )
  {
    v0 = 0;
    do
    {
      *v1++ ^= v0;
      v0 += 3;
    }
    while ( &unk_4CDFA70 != (_UNKNOWN *)v1 );
  }
}
