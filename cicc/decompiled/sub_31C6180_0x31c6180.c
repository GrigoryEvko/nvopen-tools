// Function: sub_31C6180
// Address: 0x31c6180
//
void __fastcall sub_31C6180(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1[28];
  *a1 = off_4A34BD8;
  while ( v2 )
  {
    sub_31C5F30(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = a1[23];
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
