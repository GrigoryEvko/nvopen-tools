// Function: sub_D4A4C0
// Address: 0xd4a4c0
//
__int64 __fastcall sub_D4A4C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  v1 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 72LL);
  if ( (unsigned __int8)sub_B2D610(v1, 19) || (unsigned __int8)sub_B2D610(v1, 76) )
    return 1;
  else
    return sub_D4A4A0(a1, 76, v3, v4, v5, v6);
}
