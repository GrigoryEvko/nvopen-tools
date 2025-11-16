// Function: sub_142D600
// Address: 0x142d600
//
__int64 __fastcall sub_142D600(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 552) == 0;
  *(_QWORD *)a1 = &unk_49EB5E8;
  if ( !v1 )
    sub_142D3E0(a1 + 160);
  return sub_1636790(a1);
}
