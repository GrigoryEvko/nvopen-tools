// Function: sub_2A19030
// Address: 0x2a19030
//
_QWORD *__fastcall sub_2A19030(__int64 a1, unsigned __int8 *a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r9
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rcx
  int v10; // edx
  _QWORD *result; // rax
  unsigned __int64 v12; // rbx

  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 12);
  v7 = *a2;
  v8 = *a3;
  v9 = *a4;
  if ( v5 >= v6 )
  {
    v12 = v9 & 0xFFFFFFFFFFFFFFFBLL | (4 * v7);
    if ( v6 < v5 + 1 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v5 + 1, 0x10u, a5, v6);
      v5 = *(unsigned int *)(a1 + 8);
    }
    result = (_QWORD *)(*(_QWORD *)a1 + 16 * v5);
    *result = v8;
    result[1] = v12;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 8);
    result = (_QWORD *)(*(_QWORD *)a1 + 16 * v5);
    if ( result )
    {
      *result = v8;
      result[1] = (4 * v7) | v9 & 0xFFFFFFFFFFFFFFFBLL;
      v10 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v10 + 1;
  }
  return result;
}
