// Function: sub_1456E10
// Address: 0x1456e10
//
__int64 __fastcall sub_1456E10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  if ( *(_BYTE *)(a2 + 8) == 11 )
    return a2;
  v2 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
  return sub_15A9650(v2, a2, v3, v4, v5, v6);
}
