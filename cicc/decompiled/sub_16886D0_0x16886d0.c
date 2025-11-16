// Function: sub_16886D0
// Address: 0x16886d0
//
_QWORD *__fastcall sub_16886D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // ebx
  __int64 v4; // rdi
  _QWORD *result; // rax
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  char v10; // [rsp+0h] [rbp-20h]

  v3 = a1;
  v4 = *(_QWORD *)(sub_1689050(a1, a2, a3) + 24);
  result = sub_1685080(v4, 16);
  if ( !result )
  {
    sub_1683C30(v4, 16, v6, v7, v8, v9, v10);
    result = 0;
  }
  *(_OWORD *)result = 0;
  *(_DWORD *)result = v3;
  *((_DWORD *)result + 2) = 1;
  *((_BYTE *)result + 12) = ~(_BYTE)v3;
  return result;
}
