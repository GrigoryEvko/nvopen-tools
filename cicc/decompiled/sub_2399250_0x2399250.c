// Function: sub_2399250
// Address: 0x2399250
//
void __fastcall sub_2399250(_QWORD *a1)
{
  __int64 v1; // rbx

  v1 = a1[1];
  *a1 = &unk_4A0AB10;
  if ( v1 )
  {
    sub_2398D90(v1 + 64);
    sub_2398F30(v1 + 32);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
