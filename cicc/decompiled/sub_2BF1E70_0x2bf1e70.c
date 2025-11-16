// Function: sub_2BF1E70
// Address: 0x2bf1e70
//
void __fastcall sub_2BF1E70(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A239A8;
  v2 = a1[6];
  if ( v2 )
    sub_2BF1C60((__int64 *)(v2 + 16), (__int64)a1);
  v3 = a1[2];
  if ( (_QWORD *)v3 != a1 + 4 )
    _libc_free(v3);
}
