// Function: sub_1F044A0
// Address: 0x1f044a0
//
char __fastcall sub_1F044A0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  char result; // al
  __int64 v7; // rcx
  int v8; // r8d
  unsigned __int64 v9; // r9
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]
  int v12; // [rsp+Ch] [rbp-24h]

  result = sub_1E17260(*(_QWORD *)(a2 + 8), *(_QWORD *)(a1 + 1976), *(_QWORD *)(a3 + 8), byte_4FCA640);
  if ( result )
  {
    v10 = a2 | 6;
    v12 = a4;
    v11 = 1;
    return sub_1F01A00(a3, (__int64)&v10, 1, v7, v8, v9);
  }
  return result;
}
