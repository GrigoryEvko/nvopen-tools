// Function: sub_1688290
// Address: 0x1688290
//
_QWORD *__fastcall sub_1688290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  _QWORD *result; // rax
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  char v10; // [rsp+0h] [rbp-20h]

  v4 = *(_QWORD *)(sub_1689050(a1, a2, a3) + 24);
  result = sub_1685080(v4, 40);
  if ( !result )
  {
    sub_1683C30(v4, 40, v6, v7, v8, v9, v10);
    result = 0;
  }
  *result = a1;
  *(_OWORD *)(result + 3) = 0;
  *(_OWORD *)(result + 1) = 0;
  result[3] = result + 2;
  return result;
}
