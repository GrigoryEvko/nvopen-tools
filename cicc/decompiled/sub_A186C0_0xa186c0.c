// Function: sub_A186C0
// Address: 0xa186c0
//
_QWORD *__fastcall sub_A186C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *result; // rax

  v4 = *(unsigned int *)(a1 + 8);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v4 + 1, 16);
    v4 = *(unsigned int *)(a1 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)a1 + 16 * v4);
  *result = a2;
  result[1] = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
