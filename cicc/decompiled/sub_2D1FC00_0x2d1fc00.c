// Function: sub_2D1FC00
// Address: 0x2d1fc00
//
_BYTE *__fastcall sub_2D1FC00(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rax
  _BYTE *v5; // r12

  if ( !sub_D97040(*a1, *(_QWORD *)(a2 + 8)) )
    return 0;
  v2 = *a1;
  v3 = sub_DD8400(*a1, a2);
  v4 = sub_D97190(v2, (__int64)v3);
  if ( *(_WORD *)(v4 + 24) != 15 )
    return 0;
  v5 = *(_BYTE **)(v4 - 8);
  if ( *v5 || (unsigned __int8)sub_CE9220(*(_QWORD *)(v4 - 8)) )
    return 0;
  return v5;
}
