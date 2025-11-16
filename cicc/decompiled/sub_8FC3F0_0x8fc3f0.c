// Function: sub_8FC3F0
// Address: 0x8fc3f0
//
void sub_8FC3F0()
{
  _BYTE *v0; // rdi
  char v1; // al

  v0 = &unk_4B7F680;
  if ( &unk_4B7F680 != &unk_4B7FEE0 )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( &unk_4B7FEE0 != (_UNKNOWN *)v0 );
  }
}
