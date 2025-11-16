// Function: sub_EFC940
// Address: 0xefc940
//
_QWORD *__fastcall sub_EFC940(__int64 a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v5; // r14
  _QWORD *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  _BOOL8 v13; // rdi
  _QWORD *v14; // rdi

  *(_QWORD *)(a1 + 96) += a2;
  if ( a2 > *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = a2;
  result = *(_QWORD **)(a1 + 16);
  v5 = (_QWORD *)(a1 + 8);
  ++*(_DWORD *)(a1 + 120);
  v6 = (_QWORD *)(a1 + 8);
  if ( !result )
    goto LABEL_13;
  do
  {
    while ( 1 )
    {
      v7 = result[2];
      v8 = result[3];
      if ( a2 >= result[4] )
        break;
      result = (_QWORD *)result[3];
      if ( !v8 )
        goto LABEL_8;
    }
    v6 = result;
    result = (_QWORD *)result[2];
  }
  while ( v7 );
LABEL_8:
  if ( v6 == v5 || a2 > v6[4] )
  {
LABEL_13:
    v9 = sub_22077B0(48);
    v10 = v6;
    *(_QWORD *)(v9 + 32) = a2;
    v6 = (_QWORD *)v9;
    *(_DWORD *)(v9 + 40) = 0;
    v11 = sub_EFBD70((_QWORD *)a1, v10, (unsigned __int64 *)(v9 + 32));
    if ( v12 )
    {
      v13 = v5 == v12 || v11 || a2 > v12[4];
      result = (_QWORD *)sub_220F040(v13, v6, v12, a1 + 8);
      ++*(_QWORD *)(a1 + 40);
    }
    else
    {
      v14 = v6;
      v6 = v11;
      result = (_QWORD *)j_j___libc_free_0(v14, 48);
    }
  }
  ++*((_DWORD *)v6 + 10);
  if ( *(_QWORD *)(a1 + 128) < a2 )
    *(_QWORD *)(a1 + 128) = a2;
  return result;
}
