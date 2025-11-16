// Function: sub_22A6490
// Address: 0x22a6490
//
__int64 __fastcall sub_22A6490(_QWORD *a1)
{
  unsigned __int64 v1; // r13

  v1 = a1[22];
  *a1 = &unk_4A09B60;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 56), 16LL * *(unsigned int *)(v1 + 72), 8);
    if ( *(_QWORD *)v1 != v1 + 16 )
      _libc_free(*(_QWORD *)v1);
    j_j___libc_free_0(v1);
  }
  return sub_BB9260((__int64)a1);
}
