// Function: sub_11F3530
// Address: 0x11f3530
//
unsigned __int64 __fastcall sub_11F3530(__int64 *a1, __int64 a2, __int64 a3)
{
  _WORD *v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // rbx
  __int64 i; // r13
  __int64 v9; // rdi

  sub_A5C020((_BYTE *)a3, a2, 0, *a1);
  v5 = *(_WORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 1u )
  {
    result = sub_CB6200(a2, (unsigned __int8 *)":\n", 2u);
  }
  else
  {
    result = 2618;
    *v5 = 2618;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  v7 = *(_QWORD *)(a3 + 56);
  for ( i = a3 + 48; i != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    while ( 1 )
    {
      v9 = v7 - 24;
      if ( !v7 )
        v9 = 0;
      sub_A693B0(v9, (_BYTE *)a2, *a1, 0);
      result = *(_QWORD *)(a2 + 32);
      if ( result >= *(_QWORD *)(a2 + 24) )
        break;
      *(_QWORD *)(a2 + 32) = result + 1;
      *(_BYTE *)result = 10;
      v7 = *(_QWORD *)(v7 + 8);
      if ( i == v7 )
        return result;
    }
    result = sub_CB5D20(a2, 10);
  }
  return result;
}
