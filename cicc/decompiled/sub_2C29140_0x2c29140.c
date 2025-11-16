// Function: sub_2C29140
// Address: 0x2c29140
//
char *__fastcall sub_2C29140(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // r8d
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  char *result; // rax
  int v12; // r8d
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(_QWORD **)(a1 + 80);
  v13[0] = a2;
  v6 = (__int64)&v5[v4];
  v7 = sub_2C25750(v5, v6, v13);
  if ( v7 + 1 != (_QWORD *)v6 )
  {
    memmove(v7, v7 + 1, v6 - (_QWORD)(v7 + 1));
    v8 = *(_DWORD *)(a1 + 88);
  }
  v13[0] = a1;
  *(_DWORD *)(a1 + 88) = v8 - 1;
  v9 = *(_QWORD **)(a2 + 56);
  v10 = (__int64)&v9[*(unsigned int *)(a2 + 64)];
  result = (char *)sub_2C25750(v9, v10, v13);
  if ( result + 8 != (char *)v10 )
  {
    result = (char *)memmove(result, result + 8, v10 - (_QWORD)(result + 8));
    v12 = *(_DWORD *)(a2 + 64);
  }
  *(_DWORD *)(a2 + 64) = v12 - 1;
  return result;
}
