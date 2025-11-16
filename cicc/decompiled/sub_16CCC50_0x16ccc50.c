// Function: sub_16CCC50
// Address: 0x16ccc50
//
__int64 __fastcall sub_16CCC50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char *v4; // rsi
  char *v5; // rdx
  __int64 result; // rax

  v2 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v2;
  v4 = *(char **)(a2 + 16);
  v5 = &v4[8 * v2];
  if ( v4 == *(char **)(a2 + 8) )
    v5 = &v4[8 * *(unsigned int *)(a2 + 28)];
  if ( v4 != v5 )
    memmove(*(void **)(a1 + 16), v4, v5 - v4);
  *(_DWORD *)(a1 + 28) = *(_DWORD *)(a2 + 28);
  result = *(unsigned int *)(a2 + 32);
  *(_DWORD *)(a1 + 32) = result;
  return result;
}
