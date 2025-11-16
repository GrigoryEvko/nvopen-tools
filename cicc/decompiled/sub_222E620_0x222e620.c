// Function: sub_222E620
// Address: 0x222e620
//
__int64 *__fastcall sub_222E620(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  char *v10; // r9
  __int64 v11; // rdi
  char v13; // [rsp+Bh] [rbp-15h] BYREF
  int v14[5]; // [rsp+Ch] [rbp-14h] BYREF

  sub_222E2D0(&v13, a1, 0, a4);
  if ( !v13 )
    return a1;
  v9 = *a1;
  v14[0] = 0;
  v10 = (char *)a1 + *(_QWORD *)(v9 - 24);
  v11 = *((_QWORD *)v10 + 32);
  if ( !v11 )
    sub_426219(0, a1, v7, v8);
  (*(void (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, char *, int *, __int64))(*(_QWORD *)v11 + 40LL))(
    v11,
    *((_QWORD *)v10 + 29),
    0xFFFFFFFFLL,
    0,
    0xFFFFFFFFLL,
    v10,
    v14,
    a2);
  if ( !v14[0] )
    return a1;
  sub_222DC80((__int64)a1 + *(_QWORD *)(*a1 - 24), *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 32) | v14[0]);
  return a1;
}
