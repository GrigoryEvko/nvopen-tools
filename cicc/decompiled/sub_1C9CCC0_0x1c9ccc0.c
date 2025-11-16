// Function: sub_1C9CCC0
// Address: 0x1c9ccc0
//
_QWORD *__fastcall sub_1C9CCC0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cl
  __int64 v9; // rax

  v3 = *(_QWORD **)(a1 + 16);
  if ( !v3 )
  {
    v3 = (_QWORD *)(a1 + 8);
    goto LABEL_14;
  }
  v4 = *a2;
  while ( 1 )
  {
    v6 = v3[4];
    v7 = v6 > v4;
    if ( v6 == v4 )
      v7 = v3[5] > a2[1];
    v5 = (_QWORD *)v3[3];
    if ( v7 )
      v5 = (_QWORD *)v3[2];
    if ( !v5 )
      break;
    v3 = v5;
  }
  if ( v7 )
  {
LABEL_14:
    if ( v3 == *(_QWORD **)(a1 + 24) )
      return 0;
    v9 = sub_220EF80(v3);
    v4 = *a2;
    v6 = *(_QWORD *)(v9 + 32);
    v3 = (_QWORD *)v9;
    if ( *a2 != v6 )
      goto LABEL_11;
LABEL_16:
    if ( v3[5] < a2[1] )
      return 0;
    return v3;
  }
  if ( v4 == v6 )
    goto LABEL_16;
LABEL_11:
  if ( v6 < v4 )
    return 0;
  return v3;
}
