// Function: sub_3265110
// Address: 0x3265110
//
char __fastcall sub_3265110(__int64 a1, __int64 a2, unsigned int a3)
{
  char result; // al
  __int64 v4; // rdx
  __int64 v5; // [rsp+8h] [rbp-40h] BYREF
  int v6; // [rsp+10h] [rbp-38h]
  __int64 v7; // [rsp+18h] [rbp-30h] BYREF
  int v8; // [rsp+20h] [rbp-28h]
  __int64 v9; // [rsp+28h] [rbp-20h] BYREF
  int v10; // [rsp+30h] [rbp-18h]

  v5 = 0;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  result = sub_3264EF0(a1, a2, a3, (__int64)&v5, (__int64)&v7, (__int64)&v9, 0);
  if ( result )
  {
    v4 = *(_QWORD *)(a2 + 56);
    result = 0;
    if ( v4 )
      return *(_QWORD *)(v4 + 32) == 0;
  }
  return result;
}
