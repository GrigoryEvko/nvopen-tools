// Function: sub_20EAE80
// Address: 0x20eae80
//
void __fastcall sub_20EAE80(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = (__int64)(a1 + 57);
  *(_QWORD *)(v2 - 456) = off_4A00A30;
  sub_20EA5C0(v2);
  v3 = a1[47];
  if ( (_QWORD *)v3 != a1 + 49 )
    _libc_free(v3);
  v4 = a1[36];
  if ( v4 != a1[35] )
    _libc_free(v4);
  v5 = a1[23];
  if ( v5 != a1[22] )
    _libc_free(v5);
  v6 = a1[15];
  if ( (_QWORD *)v6 != a1 + 17 )
    _libc_free(v6);
  nullsub_774();
}
