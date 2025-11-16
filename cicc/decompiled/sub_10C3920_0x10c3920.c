// Function: sub_10C3920
// Address: 0x10c3920
//
__int64 __fastcall sub_10C3920(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v5; // al
  unsigned __int8 *v6; // r9
  unsigned __int8 *v7; // rax
  _BYTE *v8; // r10
  _BYTE *v9; // r9
  _BYTE *v10; // rsi
  _BYTE *v11; // r11
  _BYTE *v12; // r10
  _BYTE *v13; // rsi

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 59 )
    return 0;
  v5 = sub_995B10((_QWORD **)a1, *(_QWORD *)(a2 - 64));
  v6 = *(unsigned __int8 **)(a2 - 32);
  if ( v5 )
  {
    if ( v6 )
    {
      **(_QWORD **)(a1 + 8) = v6;
      if ( *v6 == *(_DWORD *)(a1 + 40) + 29 )
      {
        v11 = (_BYTE *)*((_QWORD *)v6 - 8);
        v12 = *(_BYTE **)(a1 + 16);
        v13 = (_BYTE *)*((_QWORD *)v6 - 4);
        if ( v11 == v12 && *v13 == 59 && (unsigned __int8)sub_10B8260((__int64 *)(a1 + 24), (__int64)v13) )
          return 1;
        if ( v13 == v12 && *v11 == 59 && (unsigned __int8)sub_10B8260((__int64 *)(a1 + 24), (__int64)v11) )
          return 1;
      }
    }
  }
  if ( !(unsigned __int8)sub_995B10((_QWORD **)a1, (__int64)v6) )
    return 0;
  v7 = *(unsigned __int8 **)(a2 - 64);
  if ( !v7 )
    return 0;
  **(_QWORD **)(a1 + 8) = v7;
  if ( *v7 != *(_DWORD *)(a1 + 40) + 29 )
    return 0;
  v8 = (_BYTE *)*((_QWORD *)v7 - 8);
  v9 = *(_BYTE **)(a1 + 16);
  v10 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( v8 != v9 || *v10 != 59 || !(unsigned __int8)sub_10B8260((__int64 *)(a1 + 24), (__int64)v10) )
  {
    if ( v10 == v9 && *v8 == 59 )
      return sub_10B8260((__int64 *)(a1 + 24), (__int64)v8);
    return 0;
  }
  return 1;
}
