// Function: sub_34F3A60
// Address: 0x34f3a60
//
__int64 __fastcall sub_34F3A60(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = off_4A38638;
  v2 = a1[41];
  if ( (_QWORD *)v2 != a1 + 43 )
    _libc_free(v2);
  v3 = a1[37];
  while ( v3 )
  {
    sub_34F3890(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  v5 = a1[29];
  if ( (_QWORD *)v5 != a1 + 31 )
    _libc_free(v5);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
