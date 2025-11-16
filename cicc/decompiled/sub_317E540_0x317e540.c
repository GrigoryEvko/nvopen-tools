// Function: sub_317E540
// Address: 0x317e540
//
_QWORD *__fastcall sub_317E540(__int64 a1, _DWORD *a2, int *a3, size_t a4)
{
  _QWORD *result; // rax
  __int64 v6; // rbx
  size_t v8; // r12
  size_t v9; // rdx
  _QWORD *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rcx
  size_t v13[2]; // [rsp+0h] [rbp-E0h] BYREF
  int v14[52]; // [rsp+10h] [rbp-D0h] BYREF

  if ( !a4 )
    return (_QWORD *)sub_317E480(a1, a2);
  v6 = *(_QWORD *)a2;
  v8 = a4;
  if ( a3 )
  {
    sub_C7D030(v14);
    sub_C7D280(v14, a3, v8);
    sub_C7D290(v14, v13);
    v8 = v13[0];
  }
  result = *(_QWORD **)(a1 + 16);
  v9 = v8 + 33 * v6;
  if ( result )
  {
    v10 = (_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v11 = result[2];
        v12 = result[3];
        if ( result[4] >= v9 )
          break;
        result = (_QWORD *)result[3];
        if ( !v12 )
          goto LABEL_10;
      }
      v10 = result;
      result = (_QWORD *)result[2];
    }
    while ( v11 );
LABEL_10:
    if ( (_QWORD *)(a1 + 8) != v10 && v10[4] <= v9 )
      return v10 + 5;
  }
  return result;
}
