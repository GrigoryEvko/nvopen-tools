// Function: sub_1F4BF20
// Address: 0x1f4bf20
//
__int64 __fastcall sub_1F4BF20(__int64 a1, __int64 a2, char a3)
{
  if ( sub_1F4B690(a1) || **(_WORD **)(a2 + 16) == 16 || !sub_1F4B670(a1) && !a3 )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 184) + 848LL))(
             *(_QWORD *)(a1 + 184),
             a1 + 72,
             a2,
             0);
  if ( sub_1F4B670(a1) && (*sub_1F4B8B0(a1, a2) & 0x3FFF) != 0x3FFF )
    return sub_1F4BE60(a1);
  return sub_1F3BC50(*(_QWORD *)(a1 + 184), a1, a2);
}
