// Function: sub_8C6FA0
// Address: 0x8c6fa0
//
_QWORD *__fastcall sub_8C6FA0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _UNKNOWN *__ptr32 *a5)
{
  __int64 v5; // r13
  _QWORD *v6; // rbx
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rsi

  v5 = *(_QWORD *)(a2 + 240);
  if ( !a1 )
    return 0;
  v6 = a1;
  while ( 1 )
  {
    v8 = *(_QWORD *)(a2 + 152);
    v9 = *(_QWORD *)(v6[1] + 88LL);
    v10 = *(_QWORD *)(v9 + 152);
    if ( (v8 == v10 || (unsigned int)sub_8D97D0(v8, v10, 0, a4, a5)) && sub_89AB40(*(_QWORD *)(v9 + 240), v5, 2, a4, a5) )
      break;
    v6 = (_QWORD *)*v6;
    if ( !v6 )
      return 0;
  }
  return v6;
}
