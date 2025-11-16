// Function: sub_125C350
// Address: 0x125c350
//
void sub_125C350()
{
  _BYTE *v0; // rdi
  char v1; // al

  v0 = &unk_4C5D1A0;
  if ( &unk_4C5D1A0 != &unk_4C5D578 )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( &unk_4C5D578 != (_UNKNOWN *)v0 );
  }
}
