// Function: sub_22A6420
// Address: 0x22a6420
//
void __fastcall sub_22A6420(__int64 a1)
{
  unsigned __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 56), 16LL * *(unsigned int *)(v1 + 72), 8);
    if ( *(_QWORD *)v1 != v1 + 16 )
      _libc_free(*(_QWORD *)v1);
    j_j___libc_free_0(v1);
  }
}
