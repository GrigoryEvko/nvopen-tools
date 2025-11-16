// Function: sub_1C9B0F0
// Address: 0x1c9b0f0
//
_QWORD *__fastcall sub_1C9B0F0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *result; // rax
  _QWORD *v4; // r8
  unsigned __int64 v5; // rdi
  bool v6; // cl
  _QWORD *v7; // rdx
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rsi
  _QWORD *v10; // rcx
  unsigned __int64 v11; // rcx
  bool v12; // r9
  _QWORD *v13; // rcx
  unsigned __int64 v14; // rcx
  bool v15; // si

  result = *(_QWORD **)(a1 + 16);
  v4 = (_QWORD *)(a1 + 8);
  if ( !result )
    return v4;
  v5 = *a2;
  while ( 1 )
  {
    v8 = result[4];
    if ( v5 == v8 )
      break;
    if ( v8 >= v5 )
    {
      v6 = v8 > v5;
      goto LABEL_4;
    }
LABEL_9:
    v7 = (_QWORD *)result[3];
    if ( !v7 )
      return v4;
LABEL_6:
    result = v7;
  }
  if ( result[5] < a2[1] )
    goto LABEL_9;
  v6 = a2[1] < result[5];
LABEL_4:
  v7 = (_QWORD *)result[2];
  if ( v6 )
  {
    v4 = result;
    if ( !v7 )
      return v4;
    goto LABEL_6;
  }
  v9 = (_QWORD *)result[3];
  if ( v9 )
  {
    while ( 1 )
    {
      v11 = v9[4];
      v12 = v11 > v5;
      if ( v5 == v11 )
        v12 = a2[1] < v9[5];
      v10 = (_QWORD *)v9[3];
      if ( v12 )
        v10 = (_QWORD *)v9[2];
      if ( !v10 )
        break;
      v9 = v10;
    }
  }
  if ( v7 )
  {
    while ( 1 )
    {
      v14 = v7[4];
      v15 = v14 < v5;
      if ( v5 == v14 )
        v15 = v7[5] < a2[1];
      v13 = (_QWORD *)v7[3];
      if ( !v15 )
      {
        v13 = (_QWORD *)v7[2];
        result = v7;
      }
      if ( !v13 )
        break;
      v7 = v13;
    }
  }
  return result;
}
