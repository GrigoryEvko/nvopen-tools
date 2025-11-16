// Function: sub_1CEBF10
// Address: 0x1cebf10
//
__int64 __fastcall sub_1CEBF10(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12

  if ( !sub_1456C80(*a1, *a2) )
    return 0;
  v2 = *a1;
  v3 = sub_146F1B0(*a1, (__int64)a2);
  v4 = sub_1456F20(v2, v3);
  if ( *(_WORD *)(v4 + 24) != 10 )
    return 0;
  v5 = *(_QWORD *)(v4 - 8);
  if ( *(_BYTE *)(v5 + 16) || (unsigned __int8)sub_1C2F070(*(_QWORD *)(v4 - 8)) )
    return 0;
  return v5;
}
