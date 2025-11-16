// Function: sub_3702B10
// Address: 0x3702b10
//
void __fastcall sub_3702B10(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A3C600;
  v2 = a1[1];
  if ( v2 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
