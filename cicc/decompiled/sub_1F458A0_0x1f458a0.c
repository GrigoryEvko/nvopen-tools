// Function: sub_1F458A0
// Address: 0x1f458a0
//
void __fastcall sub_1F458A0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = (_QWORD *)a1[27];
  unk_4FCB770 = 0;
  if ( v2 != a1 + 29 )
    _libc_free((unsigned __int64)v2);
  v3 = a1[12];
  if ( v3 != a1[11] )
    _libc_free(v3);
}
