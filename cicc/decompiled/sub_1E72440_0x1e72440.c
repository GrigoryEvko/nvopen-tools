// Function: sub_1E72440
// Address: 0x1e72440
//
void __fastcall sub_1E72440(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1;
  v2 = (_QWORD *)(a1 + 512);
  *(v2 - 64) = &unk_49FC9E0;
  sub_1E72310(v2);
  v3 = (_QWORD *)(v1 + 144);
  v1 += 64;
  sub_1E72310(v3);
  v4 = *(_QWORD *)(v1 - 16);
  if ( v4 != v1 )
    _libc_free(v4);
}
