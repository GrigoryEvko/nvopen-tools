// Function: sub_23051C0
// Address: 0x23051c0
//
void __fastcall sub_23051C0(_QWORD *a1)
{
  unsigned __int64 v1; // r13

  v1 = a1[1];
  *a1 = &unk_4A0B010;
  if ( v1 )
  {
    sub_2E81F20(v1);
    j_j___libc_free_0(v1);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
