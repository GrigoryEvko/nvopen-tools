// Function: sub_93FAB0
// Address: 0x93fab0
//
_QWORD *__fastcall sub_93FAB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // rcx

  v3 = *(_BYTE *)(a3 + 140);
  if ( (unsigned __int8)(v3 - 9) > 2u )
  {
    v4 = 0;
    if ( v3 == 12 )
      v4 = **(_QWORD **)(a3 + 168);
  }
  else
  {
    v4 = *(_QWORD *)(*(_QWORD *)(a3 + 168) + 168LL);
  }
  sub_93F6E0(a1, a2, a3, v4);
  return a1;
}
