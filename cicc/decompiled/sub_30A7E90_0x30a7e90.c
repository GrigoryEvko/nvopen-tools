// Function: sub_30A7E90
// Address: 0x30a7e90
//
_QWORD *__fastcall sub_30A7E90(__int64 a1, __int64 (__fastcall *a2)(__int64, __int64), __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax
  __int64 v7; // r8
  _QWORD *v8; // rsi
  _QWORD *result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 i; // rbx

  v6 = sub_30A7C70(a1, a4);
  v7 = a1 + 104;
  v8 = v6;
  result = *(_QWORD **)(a1 + 112);
  if ( result )
  {
    v10 = (_QWORD *)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v11 = result[2];
        v12 = result[3];
        if ( (unsigned __int64)v8 <= result[4] )
          break;
        result = (_QWORD *)result[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v10 = result;
      result = (_QWORD *)result[2];
    }
    while ( v11 );
LABEL_6:
    if ( (_QWORD *)v7 != v10 && (unsigned __int64)v8 >= v10[4] )
      v7 = (__int64)v10;
  }
  for ( i = *(_QWORD *)(v7 + 88); i; i = *(_QWORD *)(i + 8) )
    result = (_QWORD *)a2(a3, i);
  return result;
}
