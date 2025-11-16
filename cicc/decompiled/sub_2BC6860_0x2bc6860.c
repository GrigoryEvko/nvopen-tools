// Function: sub_2BC6860
// Address: 0x2bc6860
//
__int64 __fastcall sub_2BC6860(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(_QWORD, __int64))
{
  __int64 v5; // r13
  __int64 i; // r15
  __int64 v7; // rbx
  _QWORD *v8; // r12
  __int64 v9; // r13
  _QWORD *v10; // r15
  __int64 v12; // rcx
  __int64 v14; // [rsp+10h] [rbp-50h]

  v5 = (a3 - 1) / 2;
  v14 = a3 & 1;
  if ( a2 >= v5 )
  {
    v8 = (_QWORD *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v7 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = (_QWORD *)(a1 + 16 * (i + 1));
    if ( a5(*v8, *(v8 - 1)) )
    {
      --v7;
      v8 = (_QWORD *)(a1 + 8 * v7);
    }
    *(_QWORD *)(a1 + 8 * i) = *v8;
    if ( v7 >= v5 )
      break;
  }
  if ( !v14 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v7 )
    {
      v12 = 2 * v7 + 2;
      v7 = 2 * v7 + 1;
      *v8 = *(_QWORD *)(a1 + 8 * v12 - 8);
      v8 = (_QWORD *)(a1 + 8 * v7);
    }
  }
  v9 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v10 = (_QWORD *)(a1 + 8 * v9);
      v8 = (_QWORD *)(a1 + 8 * v7);
      if ( !a5(*v10, a4) )
        break;
      v7 = v9;
      *v8 = *v10;
      if ( a2 >= v9 )
      {
        v8 = (_QWORD *)(a1 + 8 * v9);
        break;
      }
      v9 = (v9 - 1) / 2;
    }
  }
LABEL_13:
  *v8 = a4;
  return a4;
}
