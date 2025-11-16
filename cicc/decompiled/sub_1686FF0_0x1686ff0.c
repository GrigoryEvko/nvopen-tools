// Function: sub_1686FF0
// Address: 0x1686ff0
//
_QWORD *__fastcall sub_1686FF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  int v5; // edx
  int v6; // ecx
  int v7; // r8d
  int v8; // r9d
  char v9; // [rsp+0h] [rbp-10h]

  v3 = *(_QWORD *)(sub_1689050(a1, a2, a3) + 24);
  result = sub_1685080(v3, 8);
  if ( result )
  {
    *result = 0;
  }
  else
  {
    sub_1683C30(v3, 8, v5, v6, v7, v8, v9);
    MEMORY[0] = 0;
    return 0;
  }
  return result;
}
