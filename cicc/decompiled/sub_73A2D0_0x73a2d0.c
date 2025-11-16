// Function: sub_73A2D0
// Address: 0x73a2d0
//
__int64 __fastcall sub_73A2D0(__int64 a1, _UNKNOWN *__ptr32 *a2, __int64 a3, __int64 a4)
{
  char v4; // al
  _UNKNOWN *__ptr32 *v5; // r8
  __int64 v7; // rsi
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // r12

  v4 = *(_BYTE *)(a1 + 24);
  v5 = a2;
  if ( v4 == 2 )
  {
    v7 = *(_QWORD *)(a1 + 56);
    if ( *(_BYTE *)(v7 + 173) == 12 )
    {
      if ( v5 )
        return sub_73A2C0((__int64)v5, v7, a3, a4, v5);
      else
        return 1;
    }
    return 0;
  }
  if ( a2 || v4 != 1 )
    return 0;
  v8 = *(_QWORD *)(a1 + 72);
  if ( !v8 )
    return 0;
  while ( 1 )
  {
    v9 = *(_BYTE *)(v8 + 24);
    if ( v9 != 2 )
      break;
    if ( *(_BYTE *)(*(_QWORD *)(v8 + 56) + 173LL) == 12 )
      return 1;
LABEL_11:
    v8 = *(_QWORD *)(v8 + 16);
    if ( !v8 )
      return 0;
  }
  if ( v9 != 1 )
    goto LABEL_11;
  v10 = *(_QWORD *)(v8 + 72);
  if ( !v10 )
    goto LABEL_11;
  while ( !(unsigned int)sub_73A2D0(v10, 0, a3, a4, v5) )
  {
    v10 = *(_QWORD *)(v10 + 16);
    if ( !v10 )
      goto LABEL_11;
  }
  return 1;
}
