// Function: sub_31F4EB0
// Address: 0x31f4eb0
//
void __fastcall sub_31F4EB0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A35300;
  v2 = a1[1];
  if ( v2 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
