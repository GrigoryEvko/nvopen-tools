// Function: sub_16CCE60
// Address: 0x16cce60
//
__int64 __fastcall sub_16CCE60(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 result; // rax
  void *v7; // rdi
  size_t v8; // rdx

  v5 = *(_QWORD *)(a3 + 16);
  if ( v5 == *(_QWORD *)(a3 + 8) )
  {
    v7 = *(void **)(a1 + 8);
    *(_QWORD *)(a1 + 16) = v7;
    v8 = 8LL * *(unsigned int *)(a3 + 28);
    if ( v8 )
      memmove(v7, *(const void **)(a3 + 16), v8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = v5;
    *(_QWORD *)(a3 + 16) = *(_QWORD *)(a3 + 8);
  }
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a3 + 24);
  *(_DWORD *)(a1 + 28) = *(_DWORD *)(a3 + 28);
  result = *(unsigned int *)(a3 + 32);
  *(_DWORD *)(a1 + 32) = result;
  *(_DWORD *)(a3 + 24) = a2;
  *(_QWORD *)(a3 + 28) = 0;
  return result;
}
