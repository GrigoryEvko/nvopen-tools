// Function: sub_C4FC50
// Address: 0xc4fc50
//
void __fastcall sub_C4FC50(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi

  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v3 = (_QWORD *)a1[9];
  if ( v3 != a1 + 11 )
    _libc_free(v3, a2);
}
