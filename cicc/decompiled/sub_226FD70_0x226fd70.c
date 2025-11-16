// Function: sub_226FD70
// Address: 0x226fd70
//
void __fastcall sub_226FD70(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A08AE8;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
