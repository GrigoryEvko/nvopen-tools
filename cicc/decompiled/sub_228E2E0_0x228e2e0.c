// Function: sub_228E2E0
// Address: 0x228e2e0
//
__int64 __fastcall sub_228E2E0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 result; // rax

  if ( *a3 != 63 )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  if ( !sub_B4DE30((__int64)a3) )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  if ( *(_WORD *)(a2 + 24) != 8 )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  if ( *(_QWORD *)(a2 + 40) != 2 )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  if ( !(unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 8), **(_QWORD **)(a2 + 32)) )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  result = sub_DBED40(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  if ( !(_BYTE)result )
    return sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  return result;
}
