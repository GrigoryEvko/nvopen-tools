// Function: sub_2306090
// Address: 0x2306090
//
void __fastcall sub_2306090(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // r12

  v1 = a1[5];
  *a1 = &unk_4A0E178;
  if ( v1 )
  {
    sub_23C6FB0(v1);
    j_j___libc_free_0(v1);
  }
  v2 = a1[4];
  if ( v2 )
  {
    sub_23C6FB0(a1[4]);
    j_j___libc_free_0(v2);
  }
}
