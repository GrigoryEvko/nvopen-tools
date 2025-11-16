// Function: sub_2E32230
// Address: 0x2e32230
//
char *__fastcall sub_2E32230(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rsi
  char *result; // rax
  int v7; // r8d
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(unsigned int *)(a1 + 72);
  v4 = *(_QWORD **)(a1 + 64);
  v8 = a2;
  v5 = (__int64)&v4[v3];
  result = (char *)sub_2E2FDA0(v4, v5, &v8);
  if ( result + 8 != (char *)v5 )
  {
    result = (char *)memmove(result, result + 8, v5 - (_QWORD)(result + 8));
    v7 = *(_DWORD *)(a1 + 72);
  }
  *(_DWORD *)(a1 + 72) = v7 - 1;
  return result;
}
