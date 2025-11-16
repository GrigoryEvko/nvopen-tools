// Function: sub_228A8F0
// Address: 0x228a8f0
//
void __fastcall sub_228A8F0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A08E50;
  v2 = a1[6];
  if ( v2 )
    j_j___libc_free_0_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
