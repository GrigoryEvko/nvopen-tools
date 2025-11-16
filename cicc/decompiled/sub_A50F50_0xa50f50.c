// Function: sub_A50F50
// Address: 0xa50f50
//
_BYTE *__fastcall sub_A50F50(__int64 *a1, __int64 **a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 *v5; // r14
  __int64 *v6; // rbx
  char v7; // r13
  _WORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rdi
  _BYTE *result; // rax

  v3 = *a1;
  v4 = *(_QWORD *)(v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 6 )
  {
    sub_CB6200(v3, "args: (", 7);
  }
  else
  {
    *(_DWORD *)v4 = 1936159329;
    *(_WORD *)(v4 + 4) = 8250;
    *(_BYTE *)(v4 + 6) = 40;
    *(_QWORD *)(v3 + 32) += 7LL;
  }
  v5 = a2[1];
  v6 = *a2;
  v7 = 1;
  while ( v5 != v6 )
  {
    v10 = *v6;
    v9 = *a1;
    if ( v7 )
    {
      v7 = 0;
    }
    else
    {
      v8 = *(_WORD **)(v9 + 32);
      if ( *(_QWORD *)(v9 + 24) - (_QWORD)v8 <= 1u )
      {
        sub_CB6200(v9, ", ", 2);
      }
      else
      {
        *v8 = 8236;
        *(_QWORD *)(v9 + 32) += 2LL;
      }
      v9 = *a1;
    }
    ++v6;
    sub_CB59D0(v9, v10);
  }
  v11 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == result )
    return (_BYTE *)sub_CB6200(v11, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v11 + 32);
  return result;
}
