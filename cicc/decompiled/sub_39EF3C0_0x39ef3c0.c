// Function: sub_39EF3C0
// Address: 0x39ef3c0
//
__int64 __fastcall sub_39EF3C0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A40B50;
  v2 = a1[41];
  if ( (_QWORD *)v2 != a1 + 43 )
    _libc_free(v2);
  return sub_38D40E0(a1);
}
