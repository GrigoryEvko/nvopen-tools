// Function: sub_C8CD20
// Address: 0xc8cd20
//
__int64 __fastcall sub_C8CD20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool v4; // zf
  char *v5; // rsi
  __int64 result; // rax

  v2 = *(unsigned int *)(a2 + 16);
  *(_DWORD *)(a1 + 16) = v2;
  v4 = *(_BYTE *)(a2 + 28) == 0;
  v5 = *(char **)(a2 + 8);
  if ( !v4 )
    v2 = *(unsigned int *)(a2 + 20);
  if ( v5 != &v5[8 * v2] )
    memmove(*(void **)(a1 + 8), v5, 8 * v2);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  result = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = result;
  return result;
}
