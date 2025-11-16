// Function: sub_5D0610
// Address: 0x5d0610
//
__int64 __fastcall sub_5D0610(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v5; // r13d
  _QWORD *v6; // rdi
  __int64 v7; // rdx
  char v8; // si
  char v9; // cl
  __int64 i; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax

  v2 = *(_QWORD *)(a1 + 160);
  if ( !v2 )
    return 1;
  v5 = sub_8D2A90(*(_QWORD *)(v2 + 120));
  if ( v5 )
  {
    sub_685330(1890, a2, a1);
    return 0;
  }
  if ( qword_4F077A8 > 0x9C3Fu && (*(_BYTE *)(v2 + 144) & 4) != 0 )
  {
    sub_685330(1891, a2, a1);
    return v5;
  }
  v6 = *(_QWORD **)(v2 + 112);
  if ( !v6 )
    return 1;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 120) + 140LL) == 12 )
    {
      v7 = *(_QWORD *)(v2 + 120);
      do
      {
        v7 = *(_QWORD *)(v7 + 160);
        v8 = *(_BYTE *)(v7 + 140);
      }
      while ( v8 == 12 );
    }
    else
    {
      v8 = *(_BYTE *)(*(_QWORD *)(v2 + 120) + 140LL);
      v7 = *(_QWORD *)(v2 + 120);
    }
    v9 = *(_BYTE *)(v6[15] + 140LL);
    for ( i = v6[15]; v9 == 12; v9 = *(_BYTE *)(i + 140) )
      i = *(_QWORD *)(i + 160);
    v11 = *(_QWORD *)(v7 + 128);
    v12 = *(_QWORD *)(i + 128);
    if ( v11 != v12 && (v8 != 2 || v8 != v9 || v11 <= v12) )
      break;
    v6 = (_QWORD *)v6[14];
    if ( !v6 )
      return 1;
  }
  if ( *v6 && v6[1] )
  {
    sub_686870(1109, a2, *v6, a1);
    return v5;
  }
  sub_686180(1110, a2, a1, v6[15]);
  return v5;
}
