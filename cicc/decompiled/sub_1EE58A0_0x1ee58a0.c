// Function: sub_1EE58A0
// Address: 0x1ee58a0
//
_DWORD *__fastcall sub_1EE58A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _DWORD *v5; // rdi
  int v6; // edx
  __int64 v7; // rsi
  _DWORD *result; // rax
  unsigned int v9; // r8d
  int v10; // r9d

  v3 = *(unsigned int *)(a1 + 8);
  v5 = *(_DWORD **)a1;
  v6 = a2;
  v7 = (__int64)&v5[2 * v3];
  result = sub_1EE52A0(v5, v7, v6);
  if ( (_DWORD *)v7 == result )
  {
    if ( v9 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v9, v10);
      result = (_DWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    *(_QWORD *)result = a2;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    result[1] |= HIDWORD(a2);
  }
  return result;
}
