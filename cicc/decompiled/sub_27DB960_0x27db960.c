// Function: sub_27DB960
// Address: 0x27db960
//
__int64 *__fastcall sub_27DB960(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r8
  int v9; // edx
  __int64 *result; // rax
  __int64 v11; // rbx
  __int64 v12; // r12

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 12);
  if ( v6 >= v7 )
  {
    v11 = *a3;
    v12 = *a2;
    if ( v7 < v6 + 1 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v6 + 1, 0x10u, v7, a6);
      v6 = *(unsigned int *)(a1 + 8);
    }
    result = (__int64 *)(*(_QWORD *)a1 + 16 * v6);
    *result = v12;
    result[1] = v11;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 8);
    result = (__int64 *)(*(_QWORD *)a1 + 16 * v6);
    if ( result )
    {
      *result = *a2;
      result[1] = *a3;
      v9 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v9 + 1;
  }
  return result;
}
