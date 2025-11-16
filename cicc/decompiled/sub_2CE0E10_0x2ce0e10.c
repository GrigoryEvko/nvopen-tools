// Function: sub_2CE0E10
// Address: 0x2ce0e10
//
_QWORD *__fastcall sub_2CE0E10(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v4; // r9
  _QWORD *v5; // rax
  unsigned __int64 v6; // rsi
  _QWORD *v7; // r8
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rdx
  bool v10; // cl

  v4 = (_QWORD *)(a1 + 8);
  v5 = *(_QWORD **)(a1 + 16);
  if ( !v5 )
    return v4;
  v6 = *a2;
  v7 = v4;
  while ( 1 )
  {
    v9 = v5[4];
    v10 = v9 < v6;
    if ( v9 == v6 )
      v10 = v5[5] < a2[1];
    v8 = (_QWORD *)v5[3];
    if ( !v10 )
    {
      v8 = (_QWORD *)v5[2];
      v7 = v5;
    }
    if ( !v8 )
      break;
    v5 = v8;
  }
  if ( v4 == v7 )
    return v7;
  if ( v7[4] != v6 )
  {
    if ( v7[4] > v6 )
      return v4;
    return v7;
  }
  if ( a2[1] < v7[5] )
    return v4;
  return v7;
}
