// Function: sub_3502F80
// Address: 0x3502f80
//
void __fastcall sub_3502F80(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A38790;
  v2 = a1[6];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = a1[3];
  if ( v3 )
    j_j___libc_free_0(v3);
  j_j___libc_free_0((unsigned __int64)a1);
}
