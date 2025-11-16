// Function: sub_1824F90
// Address: 0x1824f90
//
void *__fastcall sub_1824F90(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi

  v2 = a1[51];
  *a1 = off_49F0A40;
  while ( v2 )
  {
    sub_1824DC0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  return sub_1636790(a1);
}
