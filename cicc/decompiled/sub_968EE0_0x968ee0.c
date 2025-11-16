// Function: sub_968EE0
// Address: 0x968ee0
//
__int64 __fastcall sub_968EE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax

  if ( !a1 )
    return 771;
  if ( !*(_QWORD *)(a1 + 40) || !((__int64 (*)(void))sub_B43CB0)() )
    return 771;
  v2 = sub_B43CB0(a1);
  v3 = sub_BCAC60(a2);
  return sub_B2DB90(v2, v3);
}
