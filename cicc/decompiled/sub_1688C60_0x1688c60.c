// Function: sub_1688C60
// Address: 0x1688c60
//
void __fastcall sub_1688C60(__int64 a1, char a2)
{
  if ( a1 )
  {
    if ( !a2 )
      sub_1688BB0(*(_QWORD *)(a1 - 8));
    _libc_free(a1 - 8);
  }
}
