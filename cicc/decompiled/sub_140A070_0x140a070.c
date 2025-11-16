// Function: sub_140A070
// Address: 0x140a070
//
__int64 __fastcall sub_140A070(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1[28];
  if ( v2 != a1[27] )
    _libc_free(v2);
  v3 = a1[20];
  if ( (_QWORD *)v3 != a1 + 22 )
    _libc_free(v3);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
