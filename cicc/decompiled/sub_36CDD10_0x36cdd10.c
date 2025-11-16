// Function: sub_36CDD10
// Address: 0x36cdd10
//
void __fastcall sub_36CDD10(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A3B1C0;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  sub_BB9280((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
