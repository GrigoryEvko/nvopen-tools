// Function: sub_1CBECF0
// Address: 0x1cbecf0
//
__int64 __fastcall sub_1CBECF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(_QWORD, __int64))
{
  __int64 i; // r15
  __int64 v6; // rbx
  _QWORD *v7; // r12
  __int64 v8; // r13
  _QWORD *v9; // r15
  __int64 v12; // [rsp+10h] [rbp-60h]
  __int64 v13; // [rsp+18h] [rbp-58h]

  v13 = (a3 - 1) / 2;
  v12 = a3 & 1;
  if ( a2 >= v13 )
  {
    v7 = (_QWORD *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v6 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v7 = (_QWORD *)(a1 + 16 * (i + 1));
    if ( a5(*v7, *(_QWORD *)(a1 + 8 * (v6 - 1))) )
      v7 = (_QWORD *)(a1 + 8 * --v6);
    *(_QWORD *)(a1 + 8 * i) = *v7;
    if ( v6 >= v13 )
      break;
  }
  if ( !v12 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v6 )
    {
      v6 = 2 * v6 + 1;
      *v7 = *(_QWORD *)(a1 + 8 * v6);
      v7 = (_QWORD *)(a1 + 8 * v6);
    }
  }
  v8 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v9 = (_QWORD *)(a1 + 8 * v8);
      v7 = (_QWORD *)(a1 + 8 * v6);
      if ( !a5(*v9, a4) )
        break;
      v6 = v8;
      *v7 = *v9;
      if ( a2 >= v8 )
      {
        v7 = (_QWORD *)(a1 + 8 * v8);
        break;
      }
      v8 = (v8 - 1) / 2;
    }
  }
LABEL_13:
  *v7 = a4;
  return a4;
}
