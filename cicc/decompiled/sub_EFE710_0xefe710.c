// Function: sub_EFE710
// Address: 0xefe710
//
__int64 __fastcall sub_EFE710(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf

  v6 = *(_BYTE *)(a1 + 1824) == 0;
  *(_QWORD *)a1 = &unk_49E4EE8;
  if ( !v6 )
  {
    *(_BYTE *)(a1 + 1824) = 0;
    sub_EFDF60(a1 + 16, a2, a3, a4, a5, a6);
  }
  return j_j___libc_free_0(a1, 1880);
}
