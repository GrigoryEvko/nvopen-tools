// Function: sub_23992A0
// Address: 0x23992a0
//
void __fastcall sub_23992A0(_QWORD *a1)
{
  __int64 v1; // rbx

  v1 = a1[1];
  *a1 = &unk_4A0AAE8;
  if ( v1 )
  {
    sub_2398D90(v1 + 64);
    sub_2398F30(v1 + 32);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
