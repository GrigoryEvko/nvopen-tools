// Function: sub_2305210
// Address: 0x2305210
//
void __fastcall sub_2305210(_QWORD *a1)
{
  unsigned __int64 v1; // r13

  v1 = a1[1];
  *a1 = &unk_4A0AF20;
  if ( v1 )
  {
    sub_103C970(v1);
    j_j___libc_free_0(v1);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
