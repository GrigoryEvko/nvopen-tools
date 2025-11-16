// Function: sub_30A7F20
// Address: 0x30a7f20
//
_QWORD *__fastcall sub_30A7F20(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64),
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  _QWORD *v8; // rax
  __int64 v9; // r8
  _QWORD *v10; // rsi
  _QWORD *result; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 i; // rbx

  if ( !a4 )
    return (_QWORD *)sub_30A6FB0(a1, (__int64)a2, a3, 0, a5, a6);
  v8 = sub_30A7C70(a1, a4);
  v9 = a1 + 104;
  v10 = v8;
  result = *(_QWORD **)(a1 + 112);
  if ( result )
  {
    v12 = (_QWORD *)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v13 = result[2];
        v14 = result[3];
        if ( (unsigned __int64)v10 <= result[4] )
          break;
        result = (_QWORD *)result[3];
        if ( !v14 )
          goto LABEL_7;
      }
      v12 = result;
      result = (_QWORD *)result[2];
    }
    while ( v13 );
LABEL_7:
    if ( (_QWORD *)v9 != v12 && (unsigned __int64)v10 >= v12[4] )
      v9 = (__int64)v12;
  }
  for ( i = *(_QWORD *)(v9 + 88); i; i = *(_QWORD *)(i + 8) )
    result = (_QWORD *)a2(a3, i);
  return result;
}
