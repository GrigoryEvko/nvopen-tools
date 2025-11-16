// Function: sub_384D060
// Address: 0x384d060
//
void __fastcall sub_384D060(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A3DCA0;
  v2 = a1[20];
  if ( (_QWORD *)v2 != a1 + 22 )
    j_j___libc_free_0(v2);
  *a1 = &unk_4A3DD58;
  sub_16366C0(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
