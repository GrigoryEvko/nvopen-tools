// Function: sub_7D06D0
// Address: 0x7d06d0
//
_BOOL8 __fastcall sub_7D06D0(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  __int64 v6; // r12
  char v7; // al
  char v8; // al
  _BOOL8 result; // rax
  __int64 v10; // rsi
  __int64 v11; // rbx
  char v12; // al
  __int64 v13; // rsi

  v6 = a2;
  v7 = *(_BYTE *)(a2 + 80);
  if ( v7 == 16 )
  {
    v6 = **(_QWORD **)(a2 + 88);
    v7 = *(_BYTE *)(v6 + 80);
  }
  if ( v7 == 24 )
    v6 = *(_QWORD *)(v6 + 88);
  if ( !a1 )
    return 0;
  v8 = *(_BYTE *)(a1 + 80);
  if ( v8 == 24 )
  {
    v10 = *(_QWORD *)(a1 + 88);
    result = 1;
    if ( v6 != v10 )
      return (unsigned int)sub_7D0550(v6, v10, a3, a4) != 0;
  }
  else
  {
    if ( v8 != 17 )
    {
      result = 1;
      if ( a1 != v6 )
        return (unsigned int)sub_7D0550(a1, v6, a3, a4) != 0;
      return result;
    }
    v11 = *(_QWORD *)(a1 + 88);
    if ( !v11 )
      return 0;
    while ( 1 )
    {
      v12 = *(_BYTE *)(v11 + 80);
      v13 = v11;
      if ( v12 == 16 )
      {
        v13 = **(_QWORD **)(v11 + 88);
        v12 = *(_BYTE *)(v13 + 80);
      }
      if ( v12 == 24 )
        v13 = *(_QWORD *)(v13 + 88);
      if ( v6 == v13 || (unsigned int)sub_7D0550(v6, v13, a3, a4) )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        return 0;
    }
    return 1;
  }
  return result;
}
