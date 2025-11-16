// Function: sub_96AEE0
// Address: 0x96aee0
//
__int64 __fastcall sub_96AEE0(__int64 a1, __int64 a2, char a3)
{
  __int64 *v3; // r14
  __int64 v6; // rdi
  char v7; // al
  __int16 v9; // ax
  char v10; // dl

  v3 = (__int64 *)(a1 + 24);
  v6 = a1 + 24;
  if ( *(_QWORD *)(a1 + 24) == sub_C33340() )
    v7 = sub_C40310(v6);
  else
    v7 = sub_C33940(v6);
  if ( !v7 )
    return a1;
  v9 = sub_968EE0(a2, *(_QWORD *)(a1 + 8));
  v10 = v9;
  if ( !a3 )
    v10 = HIBYTE(v9);
  return sub_96AC80(*(_QWORD **)(a1 + 8), v3, v10);
}
