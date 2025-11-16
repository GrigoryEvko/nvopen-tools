// Function: sub_142D660
// Address: 0x142d660
//
__int64 __fastcall sub_142D660(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 552) == 0;
  *(_QWORD *)a1 = &unk_49EB5E8;
  if ( !v1 )
    sub_142D3E0(a1 + 160);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 560);
}
