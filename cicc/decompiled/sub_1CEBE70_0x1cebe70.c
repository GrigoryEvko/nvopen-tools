// Function: sub_1CEBE70
// Address: 0x1cebe70
//
__int64 __fastcall sub_1CEBE70(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int8 v6; // al

  if ( !sub_1456C80(*a1, *a2) )
    return 0;
  v2 = *a1;
  v3 = sub_146F1B0(*a1, (__int64)a2);
  v4 = sub_1456F20(v2, v3);
  if ( *(_WORD *)(v4 + 24) != 10 )
    return 0;
  v5 = *(_QWORD *)(v4 - 8);
  v6 = *(_BYTE *)(v5 + 16);
  if ( v6 <= 0x17u || v6 != 53 && (v6 != 72 || !sub_1CCB220(v5) && !sub_1CCB280(v5)) )
    return 0;
  return v5;
}
