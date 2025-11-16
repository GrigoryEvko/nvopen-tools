// Function: sub_1F6E6F0
// Address: 0x1f6e6f0
//
char __fastcall sub_1F6E6F0(__int64 a1, __int64 a2, unsigned int a3)
{
  char result; // al
  __int64 v4; // rdx
  __int64 v5; // [rsp+0h] [rbp-40h] BYREF
  int v6; // [rsp+8h] [rbp-38h]
  __int64 v7; // [rsp+10h] [rbp-30h] BYREF
  int v8; // [rsp+18h] [rbp-28h]
  __int64 v9; // [rsp+20h] [rbp-20h] BYREF
  int v10; // [rsp+28h] [rbp-18h]

  v5 = 0;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  result = sub_1F6E520(a1, a2, a3, (__int64)&v5, (__int64)&v7, (__int64)&v9);
  if ( result )
  {
    v4 = *(_QWORD *)(a2 + 48);
    result = 0;
    if ( v4 )
      return *(_QWORD *)(v4 + 32) == 0;
  }
  return result;
}
