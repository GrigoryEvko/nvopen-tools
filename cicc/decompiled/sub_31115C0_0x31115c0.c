// Function: sub_31115C0
// Address: 0x31115c0
//
void __fastcall sub_31115C0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A32A78;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
