// Function: sub_1E723E0
// Address: 0x1e723e0
//
void __fastcall sub_1E723E0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  _QWORD *v3; // rdi
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rdi

  *a1 = &unk_49FCA80;
  v2 = a1[63];
  if ( (_QWORD *)v2 != a1 + 65 )
    _libc_free(v2);
  v3 = a1 + 17;
  v4 = a1 + 8;
  sub_1E72310(v3);
  v5 = *(v4 - 2);
  if ( (_QWORD *)v5 != v4 )
    _libc_free(v5);
}
