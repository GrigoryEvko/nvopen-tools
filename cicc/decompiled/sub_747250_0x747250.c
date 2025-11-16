// Function: sub_747250
// Address: 0x747250
//
__int64 __fastcall sub_747250(__int64 a1, unsigned int a2, __int64 a3)
{
  char v5; // al
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // r8

  v5 = *(_BYTE *)(a1 + 173);
  if ( v5 == 6 )
  {
    if ( (*(_BYTE *)(a1 + 168) & 8) == 0 )
      return sub_74E040(a1, 1, a2, a3);
    if ( *(_BYTE *)(a1 + 176) != 2 )
      return sub_74E040(a1, 1, a2, a3);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 173LL) != 2 )
      return sub_74E040(a1, 1, a2, a3);
    if ( *(_QWORD *)(a1 + 192) )
      return sub_74E040(a1, 1, a2, a3);
    if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(a1 + 128)) )
      return sub_74E040(a1, 1, a2, a3);
    v7 = sub_8D46C0(*(_QWORD *)(a1 + 128));
    v9 = sub_8D4050(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 128LL));
    if ( v7 != v9 && !(unsigned int)sub_8D97D0(v7, v9, 0, v8, v10) )
      return sub_74E040(a1, 1, a2, a3);
    v5 = *(_BYTE *)(a1 + 173);
  }
  if ( v5 == 12 && !*(_BYTE *)(a1 + 176) )
    return sub_74C550(a1, 2, a3);
  else
    return sub_748000(a1, a2, a3);
}
