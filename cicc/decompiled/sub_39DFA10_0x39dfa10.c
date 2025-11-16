// Function: sub_39DFA10
// Address: 0x39dfa10
//
__int64 __fastcall sub_39DFA10(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 280) + 358LL) )
    return sub_16E7AB0(*(_QWORD *)(a1 + 272), a2);
  v2 = sub_38D71A0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 24LL), a2);
  if ( v2 == -1 )
    return sub_16E7AB0(*(_QWORD *)(a1 + 272), a2);
  else
    return (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 288) + 24LL))(
             *(_QWORD *)(a1 + 288),
             *(_QWORD *)(a1 + 272),
             v2);
}
