// Function: sub_11F34E0
// Address: 0x11f34e0
//
__int64 __fastcall sub_11F34E0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49E64B0;
  v2 = (_QWORD *)a1[1];
  if ( v2 != a1 + 3 )
    j_j___libc_free_0(v2, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 40);
}
