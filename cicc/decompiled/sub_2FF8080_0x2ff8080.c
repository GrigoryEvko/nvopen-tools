// Function: sub_2FF8080
// Address: 0x2ff8080
//
__int64 __fastcall sub_2FF8080(__int64 a1, __int64 a2, char a3)
{
  char v6; // al
  _WORD *v7; // rsi

  if ( *(_BYTE *)(a1 + 72) )
    return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 1136LL))(*(_QWORD *)(a1 + 200));
  if ( sub_2FF7B90(a1) )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 200) + 1160LL))(
             *(_QWORD *)(a1 + 200),
             a1 + 80,
             a2,
             0);
  if ( *(_WORD *)(a2 + 68) == 21 )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 200) + 1160LL))(
             *(_QWORD *)(a1 + 200),
             a1 + 80,
             a2,
             0);
  v6 = sub_2FF7B70(a1);
  if ( !a3 && !v6 )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 200) + 1160LL))(
             *(_QWORD *)(a1 + 200),
             a1 + 80,
             a2,
             0);
  if ( !sub_2FF7B70(a1) )
    return sub_2FE09D0(*(_QWORD *)(a1 + 200), a1, a2);
  v7 = sub_2FF7DB0(a1, a2);
  if ( (*v7 & 0x1FFF) == 0x1FFF )
    return sub_2FE09D0(*(_QWORD *)(a1 + 200), a1, a2);
  return sub_2FF8060(a1, (__int64)v7);
}
