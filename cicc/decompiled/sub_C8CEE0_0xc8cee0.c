// Function: sub_C8CEE0
// Address: 0xc8cee0
//
__int64 __fastcall sub_C8CEE0(__int64 a1, void *a2, int a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  size_t v8; // rdx

  if ( *(_BYTE *)(a5 + 28) )
  {
    *(_QWORD *)(a1 + 8) = a2;
    v8 = 8LL * *(unsigned int *)(a5 + 20);
    if ( v8 )
      memmove(a2, *(const void **)(a5 + 8), v8);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a5 + 8);
    *(_QWORD *)(a5 + 8) = a4;
  }
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a5 + 16);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a5 + 20);
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a5 + 24);
  result = *(unsigned __int8 *)(a5 + 28);
  *(_BYTE *)(a1 + 28) = result;
  *(_DWORD *)(a5 + 16) = a3;
  *(_QWORD *)(a5 + 20) = 0;
  *(_BYTE *)(a5 + 28) = 1;
  return result;
}
