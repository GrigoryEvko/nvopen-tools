// Function: sub_829160
// Address: 0x829160
//
__int64 __fastcall sub_829160(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  char v3; // al
  bool v4; // zf
  char v5; // al
  unsigned __int8 v6; // al

  if ( !a1 )
    return 0;
  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
  v3 = *(_BYTE *)(a1 + 80);
  if ( v3 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v3 = *(_BYTE *)(a1 + 80);
  }
  v4 = v3 == 24;
  v5 = *(_BYTE *)(v2 + 80);
  if ( v4 )
    a1 = *(_QWORD *)(a1 + 88);
  if ( v5 == 16 )
  {
    v2 = **(_QWORD **)(v2 + 88);
    v5 = *(_BYTE *)(v2 + 80);
  }
  if ( v5 == 24 )
    v2 = *(_QWORD *)(v2 + 88);
  v6 = *(_BYTE *)(*(_QWORD *)(a1 + 88) + 195LL);
  if ( ((v6 ^ *(_BYTE *)(*(_QWORD *)(v2 + 88) + 195LL)) & 2) != 0 )
    return (v6 & 2) == 0 ? -1 : 1;
  else
    return sub_6F3270(a1, v2, 0);
}
