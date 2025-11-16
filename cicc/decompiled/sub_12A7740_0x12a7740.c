// Function: sub_12A7740
// Address: 0x12a7740
//
void __fastcall sub_12A7740(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi

  v1 = a1[2];
  while ( v1 )
  {
    sub_12A7570(*(_QWORD *)(v1 + 24));
    v2 = v1;
    v1 = *(_QWORD *)(v1 + 16);
    j_j___libc_free_0(v2, 48);
  }
}
