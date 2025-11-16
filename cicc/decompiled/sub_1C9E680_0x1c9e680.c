// Function: sub_1C9E680
// Address: 0x1c9e680
//
_QWORD *__fastcall sub_1C9E680(_QWORD *a1, unsigned __int64 a2, int a3, _QWORD *a4)
{
  _QWORD *v5; // r9
  _QWORD *v8; // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *result; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // [rsp+8h] [rbp-28h] BYREF
  unsigned __int64 *v19; // [rsp+18h] [rbp-18h] BYREF

  v5 = a1 + 1;
  v8 = (_QWORD *)a1[2];
  v18 = a2;
  if ( !v8 )
  {
    v10 = (__int64)(a1 + 1);
LABEL_17:
    v19 = &v18;
    v10 = sub_1C9E550(a1, (_QWORD *)v10, &v19);
    goto LABEL_8;
  }
  v10 = (__int64)v5;
  do
  {
    while ( 1 )
    {
      v11 = v8[2];
      v12 = v8[3];
      if ( v8[4] >= a2 )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v12 )
        goto LABEL_6;
    }
    v10 = (__int64)v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v11 );
LABEL_6:
  if ( v5 == (_QWORD *)v10 || *(_QWORD *)(v10 + 32) > a2 )
    goto LABEL_17;
LABEL_8:
  *(_DWORD *)(v10 + 40) = a3;
  result = (_QWORD *)a4[2];
  if ( result )
  {
    v14 = a4 + 1;
    do
    {
      while ( 1 )
      {
        v15 = result[2];
        v16 = result[3];
        if ( result[4] >= v18 )
          break;
        result = (_QWORD *)result[3];
        if ( !v16 )
          goto LABEL_13;
      }
      v14 = result;
      result = (_QWORD *)result[2];
    }
    while ( v15 );
LABEL_13:
    if ( a4 + 1 != v14 && v14[4] <= v18 )
    {
      v17 = sub_220F330(v14, a4 + 1);
      result = (_QWORD *)j_j___libc_free_0(v17, 40);
      --a4[5];
    }
  }
  return result;
}
