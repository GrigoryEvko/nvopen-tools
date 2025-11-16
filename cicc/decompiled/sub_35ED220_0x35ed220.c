// Function: sub_35ED220
// Address: 0x35ed220
//
void sub_35ED220()
{
  char v0; // al
  _BYTE *v1; // rdi

  v1 = &unk_4CE00A0;
  if ( &unk_4CE00A0 != &unk_4CE0317 )
  {
    v0 = 0;
    do
    {
      *v1++ ^= v0;
      v0 += 3;
    }
    while ( &unk_4CE0317 != (_UNKNOWN *)v1 );
  }
}
