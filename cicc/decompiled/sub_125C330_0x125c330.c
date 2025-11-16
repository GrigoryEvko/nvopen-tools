// Function: sub_125C330
// Address: 0x125c330
//
void sub_125C330()
{
  _BYTE *v0; // rdi
  char v1; // al

  v0 = &unk_4C5D580;
  if ( &unk_4C5D580 != &unk_4C5DDE0 )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( &unk_4C5DDE0 != (_UNKNOWN *)v0 );
  }
}
