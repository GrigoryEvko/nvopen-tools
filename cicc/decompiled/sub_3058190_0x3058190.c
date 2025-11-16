// Function: sub_3058190
// Address: 0x3058190
//
_BYTE *__fastcall sub_3058190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdx
  _BYTE *result; // rax

  v7 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 7u )
  {
    sub_CB6200(a2, "generic(", 8u);
  }
  else
  {
    *v7 = 0x28636972656E6567LL;
    *(_QWORD *)(a2 + 32) += 8LL;
  }
  sub_E7FAD0(*(unsigned int **)(a1 + 24), a2, a3, 0, a5, a6);
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)")", 1u);
  *result = 41;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
