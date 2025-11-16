// Function: sub_3244E70
// Address: 0x3244e70
//
__int64 *__fastcall sub_3244E70(__int64 a1, unsigned __int8 a2)
{
  _QWORD **v2; // rbx
  __int64 *result; // rax
  _QWORD **i; // r14
  _QWORD *v6; // rsi

  v2 = *(_QWORD ***)(a1 + 152);
  result = (__int64 *)*(unsigned int *)(a1 + 160);
  for ( i = &v2[(_QWORD)result]; i != v2; result = sub_3244DB0((__int64 *)a1, v6, a2) )
    v6 = *v2++;
  return result;
}
