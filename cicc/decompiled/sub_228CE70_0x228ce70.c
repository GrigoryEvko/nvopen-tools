// Function: sub_228CE70
// Address: 0x228ce70
//
__int64 *__fastcall sub_228CE70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 *v8; // rdi
  __int64 *v9; // rax
  __int64 *v10; // rdi
  __int64 *result; // rax

  v5 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)a1 = 2;
  v6 = sub_D95540(a2);
  v7 = sub_DA2C50(v5, v6, 1, 0);
  v8 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = v7;
  v9 = sub_DCAF50(v8, (__int64)v7, 0);
  v10 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 24) = v9;
  result = sub_DCAF50(v10, a2, 0);
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 32) = result;
  return result;
}
