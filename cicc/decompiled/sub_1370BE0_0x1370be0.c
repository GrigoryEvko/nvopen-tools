// Function: sub_1370BE0
// Address: 0x1370be0
//
unsigned __int64 *__fastcall sub_1370BE0(__int64 a1, unsigned int *a2, unsigned __int64 a3, unsigned int a4)
{
  unsigned __int64 v5; // rax
  bool v6; // dl
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 *result; // rax

  v5 = *(_QWORD *)(a1 + 80) + a3;
  v6 = __CFADD__(*(_QWORD *)(a1 + 80), a3);
  *(_QWORD *)(a1 + 80) = v5;
  *(_BYTE *)(a1 + 88) |= v6;
  v7 = ((unsigned __int64)*a2 << 32) | a4;
  v8 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 16);
    v8 = *(unsigned int *)(a1 + 8);
  }
  result = (unsigned __int64 *)(*(_QWORD *)a1 + 16 * v8);
  *result = v7;
  result[1] = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
