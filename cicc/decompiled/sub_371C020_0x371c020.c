// Function: sub_371C020
// Address: 0x371c020
//
unsigned __int64 __fastcall sub_371C020(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 result; // rax
  _QWORD *v7; // r12
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD **)(a1 + 48);
  v4 = *(unsigned int *)(a1 + 56);
  v10[0] = a2;
  v5 = (__int64)&v3[v4];
  result = (unsigned __int64)sub_371B710(v3, v5, v10);
  if ( v5 != result )
  {
    v7 = (_QWORD *)result;
    sub_371C000(a1, v10[0]);
    v8 = *(_QWORD *)(a1 + 48) + 8LL * *(unsigned int *)(a1 + 56);
    v9 = *(_DWORD *)(a1 + 56);
    if ( (_QWORD *)v8 != v7 + 1 )
    {
      memmove(v7, v7 + 1, v8 - (_QWORD)(v7 + 1));
      v9 = *(_DWORD *)(a1 + 56);
    }
    result = (unsigned int)(v9 - 1);
    *(_DWORD *)(a1 + 56) = result;
  }
  return result;
}
