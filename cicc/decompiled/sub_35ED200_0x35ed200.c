// Function: sub_35ED200
// Address: 0x35ed200
//
void sub_35ED200()
{
  char v0; // al
  _BYTE *v1; // rdi

  v1 = &unk_4CE0320;
  if ( &unk_4CE0320 != &unk_4CF6D5F )
  {
    v0 = 0;
    do
    {
      *v1++ ^= v0;
      v0 += 3;
    }
    while ( &unk_4CF6D5F != (_UNKNOWN *)v1 );
  }
}
