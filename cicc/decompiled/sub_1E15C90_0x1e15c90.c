// Function: sub_1E15C90
// Address: 0x1e15c90
//
_QWORD *__fastcall sub_1E15C90(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v4; // r14
  char v5; // r15
  __int64 v6; // r12
  _QWORD *result; // rax
  _QWORD *v8; // rcx

  v4 = *(const void **)(a1 + 56);
  v5 = *(_BYTE *)(a1 + 49) + 1;
  v6 = *(unsigned __int8 *)(a1 + 49);
  result = (_QWORD *)sub_1E0A240(a2, v6 + 1);
  v8 = result;
  if ( 8 * v6 )
  {
    result = memmove(result, v4, 8 * v6);
    v8 = result;
  }
  v8[v6] = a3;
  *(_BYTE *)(a1 + 49) = v5;
  *(_QWORD *)(a1 + 56) = v8;
  return result;
}
