// Function: sub_CB0910
// Address: 0xcb0910
//
__int64 __fastcall sub_CB0910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 result; // rax
  int v8; // edx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rsi

  v6 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(a1 + 40);
  v8 = *(_DWORD *)(v6 + 4 * result - 4);
  if ( v8 )
  {
    if ( v8 == 2 )
    {
      result = (unsigned int)(result - 1);
      v10 = *(unsigned int *)(a1 + 44);
      *(_DWORD *)(a1 + 40) = result;
      if ( result + 1 > v10 )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 4u, a5, a6);
        v6 = *(_QWORD *)(a1 + 32);
        result = *(unsigned int *)(a1 + 40);
      }
      *(_DWORD *)(v6 + 4 * result) = 3;
      ++*(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    result = (unsigned int)(result - 1);
    v9 = *(unsigned int *)(a1 + 44);
    *(_DWORD *)(a1 + 40) = result;
    if ( result + 1 > v9 )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 4u, a5, a6);
      v6 = *(_QWORD *)(a1 + 32);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_DWORD *)(v6 + 4 * result) = 1;
    ++*(_DWORD *)(a1 + 40);
  }
  return result;
}
