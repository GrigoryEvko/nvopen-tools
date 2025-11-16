// Function: sub_16875A0
// Address: 0x16875a0
//
_QWORD *__fastcall sub_16875A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  _QWORD *result; // rax
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  char v10; // [rsp-28h] [rbp-28h]

  if ( !*(_QWORD *)(a1 + 48) )
    return 0;
  v4 = *(_QWORD *)(sub_1689050(a1, a2, a3) + 24);
  result = sub_1685080(v4, 16);
  if ( !result )
  {
    sub_1683C30(v4, 16, v6, v7, v8, v9, v10);
    result = 0;
  }
  result[1] = 0;
  *result = a1;
  *((_DWORD *)result + 3) = **(_DWORD **)(a1 + 96);
  return result;
}
