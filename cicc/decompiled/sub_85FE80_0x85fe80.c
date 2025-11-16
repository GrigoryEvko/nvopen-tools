// Function: sub_85FE80
// Address: 0x85fe80
//
_QWORD *__fastcall sub_85FE80(int a1, int a2, unsigned int a3)
{
  _QWORD *result; // rax
  __int64 v5; // r10
  __int64 v7; // rcx
  __int64 v8; // rax
  bool v9; // r11
  char v10; // dl
  unsigned int v11; // r9d
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int v14; // edi
  __int64 v15; // rdx
  __int64 v16; // rcx

  result = qword_4F04C68;
  v5 = qword_4F04C68[0];
  v7 = qword_4F04C68[0] + 776LL * a1;
  if ( a2 )
  {
    result = (_QWORD *)(qword_4F04C68[0] + 776LL * a1);
    if ( !v7 )
      return result;
    do
    {
      result[68] = 0;
      v8 = *((int *)result + 138);
      if ( (_DWORD)v8 == -1 )
        break;
      result = (_QWORD *)(v5 + 776 * v8);
    }
    while ( result );
  }
  else if ( !v7 )
  {
    return result;
  }
  v9 = a3 != 0 && unk_4D047BC == 0;
  do
  {
    for ( result = *(_QWORD **)(v7 + 536); result; result = (_QWORD *)*result )
    {
      while ( !a2 )
      {
        *(_DWORD *)(result[3] + 168LL) = 0;
        result = (_QWORD *)*result;
        if ( !result )
          goto LABEL_19;
      }
      v10 = *(_BYTE *)(v7 + 4);
      v11 = *((_DWORD *)result + 9);
      if ( v10 == 2 || v10 == 17 || !v9 || v11 <= a3 )
      {
        v12 = result[3];
        v13 = *((int *)result + 8);
        v14 = *(_DWORD *)(v12 + 168);
        if ( !v14 || v14 > v11 )
          *(_DWORD *)(v12 + 168) = v11;
        v15 = v5 + 776 * v13;
        result[1] = *(_QWORD *)(v15 + 544);
        *(_QWORD *)(v15 + 544) = result;
      }
    }
LABEL_19:
    if ( !a2 )
      *(_QWORD *)(v7 + 544) = 0;
    v16 = *(int *)(v7 + 552);
    if ( (_DWORD)v16 == -1 )
      break;
    v7 = v5 + 776 * v16;
  }
  while ( v7 );
  return result;
}
