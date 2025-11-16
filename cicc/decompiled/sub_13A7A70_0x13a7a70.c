// Function: sub_13A7A70
// Address: 0x13a7a70
//
__int64 __fastcall sub_13A7A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a3 + 16) != 56 )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  if ( !(unsigned __int8)sub_15FA300(a3) )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  if ( *(_WORD *)(a2 + 24) != 7 )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  if ( *(_QWORD *)(a2 + 40) != 2 )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  if ( !(unsigned __int8)sub_1477BC0(*(_QWORD *)(a1 + 8), **(_QWORD **)(a2 + 32)) )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  result = sub_1477BC0(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  if ( !(_BYTE)result )
    return sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  return result;
}
