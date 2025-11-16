// Function: sub_356E260
// Address: 0x356e260
//
void __fastcall sub_356E260(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r12

  v1 = *a1;
  if ( *a1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 8), 16LL * *(unsigned int *)(v1 + 24), 8);
    j_j___libc_free_0(v1);
  }
}
