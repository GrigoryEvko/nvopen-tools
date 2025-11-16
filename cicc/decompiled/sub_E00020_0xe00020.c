// Function: sub_E00020
// Address: 0xe00020
//
_QWORD *__fastcall sub_E00020(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx

  *a1 = 0;
  v6 = *(__int64 **)(a3 + 16);
  a1[1] = 0;
  v7 = sub_BA6CD0(*(_QWORD *)(a2 + 16), v6);
  v8 = *(_QWORD *)(a3 + 24);
  v9 = *(_QWORD *)(a2 + 24);
  a1[2] = v7;
  a1[3] = sub_BA74A0(v9, v8, v10, v11);
  return a1;
}
