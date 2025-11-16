// Function: sub_15DE590
// Address: 0x15de590
//
void sub_15DE590()
{
  _BYTE *v0; // rdi
  char v1; // al

  v0 = &unk_4C7E1E0;
  if ( &unk_4C7E1E0 != &unk_4CD28BD )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( &unk_4CD28BD != (_UNKNOWN *)v0 );
  }
}
