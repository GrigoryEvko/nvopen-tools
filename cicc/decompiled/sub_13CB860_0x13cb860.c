// Function: sub_13CB860
// Address: 0x13cb860
//
__int64 __fastcall sub_13CB860(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v6; // rbx
  __int64 v7; // r13

  if ( *(_BYTE *)(a1 + 16) != 79 )
    return 0;
  v4 = *(_QWORD *)(a1 - 72);
  if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 75) > 1u )
    return 0;
  v6 = *(_QWORD *)(v4 - 48);
  v7 = *(_QWORD *)(v4 - 24);
  if ( (a2 != (*(_WORD *)(v4 + 18) & 0x7FFF) || a3 != v6 || a4 != v7)
    && (a2 != (unsigned int)sub_15FF5D0() || a3 != v7 || a4 != v6) )
  {
    return 0;
  }
  return v4;
}
