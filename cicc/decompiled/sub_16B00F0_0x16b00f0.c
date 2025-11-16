// Function: sub_16B00F0
// Address: 0x16b00f0
//
void __fastcall sub_16B00F0(_QWORD *a1)
{
  if ( a1[12] != a1[11] )
    _libc_free(a1[12]);
}
