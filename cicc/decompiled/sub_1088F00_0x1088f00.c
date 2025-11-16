// Function: sub_1088F00
// Address: 0x1088f00
//
__int64 __fastcall sub_1088F00(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdi

  *(_BYTE *)(a1 + 128) = 0;
  sub_10887F0(*(__int64 **)(a1 + 112), a2);
  v3 = *(__int64 **)(a1 + 120);
  if ( v3 )
    sub_10887F0(v3, a2);
  return sub_E8EB90(a1);
}
