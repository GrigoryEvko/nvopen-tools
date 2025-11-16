// Function: sub_8FC410
// Address: 0x8fc410
//
void sub_8FC410()
{
  _BYTE *v0; // rdi
  char v1; // al

  v0 = &unk_4B7F2A0;
  if ( &unk_4B7F2A0 != &unk_4B7F678 )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( &unk_4B7F678 != (_UNKNOWN *)v0 );
  }
}
