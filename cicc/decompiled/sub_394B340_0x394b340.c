// Function: sub_394B340
// Address: 0x394b340
//
unsigned __int64 __fastcall sub_394B340(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  int v4; // edx
  __int64 v6; // r13
  __int64 v7; // rax

  v4 = a3 & 0x70;
  if ( !v4 )
    return a2;
  if ( v4 != 16 )
    sub_16BD130("We do not support this DWARF encoding yet!", 1u);
  v6 = sub_38BFA60(*(_QWORD *)(a1 + 760), 1);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a4 + 176LL))(a4, v6, 0);
  v7 = sub_38CF310(v6, 0, *(_QWORD *)(a1 + 760), 0);
  return sub_38CB1F0(17, a2, v7, *(_QWORD *)(a1 + 760), 0);
}
