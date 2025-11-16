// Function: sub_16B0050
// Address: 0x16b0050
//
void __fastcall sub_16B0050(_QWORD *a1)
{
  if ( a1[12] != a1[11] )
    _libc_free(a1[12]);
}
