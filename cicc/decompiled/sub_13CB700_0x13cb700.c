// Function: sub_13CB700
// Address: 0x13cb700
//
char __fastcall sub_13CB700(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rax

  if ( *(_BYTE *)(a1 + 16) <= 0x17u )
    return 1;
  if ( !*(_QWORD *)(a1 + 40) || !*(_QWORD *)(a2 + 40) || !((__int64 (*)(void))sub_15F2060)() )
    return 0;
  if ( a3 )
    return sub_15CCEE0(a3, a1, a2);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)(sub_15F2060(a1) + 80);
  if ( v6 )
    v6 -= 24;
  return v5 == v6 && *(_BYTE *)(a1 + 16) != 29;
}
