// Function: sub_FE8630
// Address: 0xfe8630
//
unsigned __int64 *__fastcall sub_FE8630(
        __int64 a1,
        unsigned int *a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v7; // rax
  bool v8; // dl
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 *result; // rax

  v7 = *(_QWORD *)(a1 + 80) + a3;
  v8 = __CFADD__(*(_QWORD *)(a1 + 80), a3);
  *(_QWORD *)(a1 + 80) = v7;
  *(_BYTE *)(a1 + 88) |= v8;
  v9 = ((unsigned __int64)*a2 << 32) | a4;
  v10 = *(unsigned int *)(a1 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v10 + 1, 0x10u, a5, a6);
    v10 = *(unsigned int *)(a1 + 8);
  }
  result = (unsigned __int64 *)(*(_QWORD *)a1 + 16 * v10);
  *result = v9;
  result[1] = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
