// Function: sub_1B054F0
// Address: 0x1b054f0
//
_QWORD *__fastcall sub_1B054F0(__int64 a1, unsigned __int8 *a2, _QWORD *a3, _QWORD *a4, int a5, int a6)
{
  unsigned int v8; // edx
  _QWORD *result; // rax
  unsigned __int64 v10; // rdx

  v8 = *(_DWORD *)(a1 + 8);
  if ( v8 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, a5, a6);
    v8 = *(_DWORD *)(a1 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)a1 + 16LL * v8);
  if ( result )
  {
    v10 = (4LL * *a2) | *a4 & 0xFFFFFFFFFFFFFFFBLL;
    *result = *a3;
    result[1] = v10;
    v8 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v8 + 1;
  return result;
}
