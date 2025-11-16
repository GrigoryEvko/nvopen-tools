// Function: sub_2291F00
// Address: 0x2291f00
//
_QWORD *__fastcall sub_2291F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned int v10; // [rsp+Ch] [rbp-34h]

  if ( *(_WORD *)(a2 + 24) != 8 )
    return (_QWORD *)a2;
  v4 = *(_QWORD *)(a2 + 48);
  if ( a3 == v4 )
    return **(_QWORD ***)(a2 + 32);
  v5 = *(_QWORD *)(a1 + 8);
  v10 = *(_WORD *)(a2 + 28) & 7;
  v6 = sub_D33D80((_QWORD *)a2, v5, a3, a4, v10);
  v7 = sub_2291F00(a1, **(_QWORD **)(a2 + 32), a3);
  return sub_DC1960(v5, v7, v6, v4, v10);
}
