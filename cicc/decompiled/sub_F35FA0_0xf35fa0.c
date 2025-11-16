// Function: sub_F35FA0
// Address: 0xf35fa0
//
_QWORD *__fastcall sub_F35FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  _QWORD *result; // rax

  v7 = *(unsigned int *)(a1 + 8);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + 1, 0x10u, a5, a6);
    v7 = *(unsigned int *)(a1 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)a1 + 16 * v7);
  *result = a2;
  result[1] = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
